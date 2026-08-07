# VisDom: Sparse Novel View Synthesis with Visible Domain Constraint

Code release for our GCPR 2026 paper. Two pipelines:

- **[`zipnerf/`](zipnerf/README.md)** — Zip-NeRF-based reconstruction. See [`zipnerf/README.md`](zipnerf/README.md) for install + how to reproduce our results.
- **[`3dgs-go/`](3dgs-go/README.md)** — 3D Gaussian Splatting-based reconstruction. See [`3dgs-go/README.md`](3dgs-go/README.md) for install + how to reproduce our results.

Both pipelines need converted datasets — see [Dataset preparation](#dataset-preparation) below. See [Citation](#citation) for the paper and the datasets/codebases this release builds on.

Released for **non-commercial research and evaluation use only** — see [`LICENSE.md`](LICENSE.md). `zipnerf/` and `3dgs-go/` carry their own component licenses ([`zipnerf/LICENSE`](zipnerf/LICENSE), [`3dgs-go/LICENSE.md`](3dgs-go/LICENSE.md)).

## Dataset preparation

Our two pipelines expect different dataset layouts:

- **`zipnerf/`** expects the **ActorsHQ** layout: `calibration.csv` + `rgbs/CamXXX/` + `masks/CamXXX/`.
- **`3dgs-go/`** expects a **COLMAP** dataset: `images/` + `masks/` + `sparse/0/{cameras,images,points3D}.bin`.

We evaluate on three sources, none of which natively matches both formats, so we convert:

| Source | Native format | Needed by | Conversion |
|---|---|---|---|
| [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) | COLMAP | zipnerf | `colmap_to_actorshq.py` |
| [OmniObject3D](https://omniobject3d.github.io/) (as prepared by [GaussianObject](https://github.com/GaussianObject/GaussianObject)) | COLMAP | zipnerf | `colmap_to_actorshq.py` |
| [ActorsHQ](https://www.actors-hq.com/) | ActorsHQ csv | 3dgs-go | `actorshq_to_colmap.py` |

Download the raw datasets from their respective sources; the splits used for train/eval cameras per scene are in [`datasets/dataset_splits/`](datasets/dataset_splits/) (`mipnerf360/`, `omni3d/`, `exp_acthq_Actor/`).

### COLMAP -> ActorsHQ (`colmap_to_actorshq.py`)

Converts a COLMAP sparse model (Mip-NeRF 360, OmniObject3D/GaussianObject) into the ActorsHQ `calibration.csv` / `rgbs/` / `masks/` layout consumed by zipnerf.

Expects `<colmap_dir>/sparse/0/{images,cameras}.bin`, an image subdirectory (default `images_2`), and a `masks/` subdirectory with same-named masks.

```bash
python datasets/colmap_to_actorshq.py <colmap_dir> <outdir> \
    --image_subdir images_2 \
    --mask_subdir masks
```

zipnerf's `ActorsHQ` dataset loader expects the output at `<Config.data_dir>/<scene>/Sequence1/4x/`, so `<outdir>` should be e.g. `datasets/zipnerf-dataset/bonsai/Sequence1/4x` with `Config.data_dir` set to `datasets/zipnerf-dataset`. Two things to get right:

- **`--image_subdir`**: which resolution to train on (`images`, `images_2`, `images_4`, ...) — this is baked into the output permanently, there's no runtime downsampling in the ActorsHQ loader. Check what resolution the target experiment actually used (e.g. by comparing image dimensions against any existing converted copy) rather than assuming from directory naming conventions elsewhere.
- **`--timestamp`**: must match the frame index recorded in the corresponding `dataset_splits/<scene>/<camcount>/splits/{train,test}.txt` file (e.g. `12`), since that's the filename suffix (`CamXXX_rgbNNNNNN.jpg`) the loader looks for. The default of `0` only works if the split file also says `0`.

Worked example — Mip-NeRF 360 bonsai, matching the existing `mipnerf360/bonsai/004_00` split (`images_4`, timestamp `12`):

```bash
python datasets/colmap_to_actorshq.py \
    <mip360_bonsai_colmap_dir> \
    datasets/zipnerf-dataset/bonsai/Sequence1/4x \
    --image_subdir images_4 --timestamp 12
```

then, from `zipnerf/`:

```bash
./runone_vhull_4.sh mipnerf360/bonsai/004_00 1 <path-to>/datasets/zipnerf-dataset
```

(`1` here is `deg_view`; see `run_exp_vhull_mip360.sh` for per-scene values.)

### ActorsHQ -> COLMAP (`actorshq_to_colmap.py`)

Converts an ActorsHQ sequence (`calibration.csv` + `rgbs/` + `masks/`) into a COLMAP dataset for 3dgs-go: composites images onto a white background using the masks, and writes `sparse/0/{cameras,images,points3D}.txt` directly from the known ActorsHQ camera calibration.

This does **not** invoke the `colmap` binary. Camera poses/intrinsics come straight from the ActorsHQ calibration, so there's nothing for COLMAP's feature matching + triangulation to refine — `points3D.txt` is left empty, since 3dgs-go initializes gaussians from a visual hull (`visual_hull.py`), not from COLMAP-triangulated points. No `colmap` install is required for this step.

`--timestamp` picks which frame of the ActorsHQ sequence to extract (default `0`); the released `exp_acthq_Actor` splits/checkpoints were captured at frame `12`, e.g.:

```bash
python datasets/actorshq_to_colmap.py --data <actorshq_actor_dir>/Sequence1/4x --outdir <outdir> --timestamp 12
```

Depends on `camera_actorshq.py` (ActorsHQ calibration.csv reader).

#### Sparse-view splits (`conf_to_sparse_splits.py`)

3dgs-go picks its N-camera train/test split from `sparse_<N>.txt` / `sparse_test.txt` files sitting next to `sparse/0/` in the COLMAP dataset — 0-indexed positions into the camera list sorted alphabetically by filename. zipnerf's splits (`dataset_splits/.../conf.json`) list the same cameras but as 1-indexed `CamNNN` numbers. Rather than maintaining two independently-curated split files per scene, derive the 3dgs-go one from the single `dataset_splits` source of truth:

```bash
python datasets/conf_to_sparse_splits.py datasets/dataset_splits/exp_acthq_Actor/Actor01/005_00/conf.json datasets/3dgs-go-dataset/Actor01
```

(verified to reproduce the exact same camera sets as the reference `sparse_N.txt`/`sparse_test.txt` files, for both the ActorsHQ and Mip-NeRF 360 splits — order and trailing newline can differ, the selected cameras don't.)

## Citation

Accepted to GCPR 2026. The camera-ready citation isn't available yet — please cite the arXiv version for now; we'll update this with the proceedings BibTeX once it's out.

If you use this code or these datasets, please consider citing our work as well as the datasets and codebases it builds on:

```bibtex
@misc{gladkova2026visdomsparsenovelview,
  title={VisDom: Sparse Novel View Synthesis with Visible Domain Constraint},
  author={Mariia Gladkova and Tarun Yenamandra and Edmond Boyer and Robert Maier and Tony Tung and Daniel Cremers},
  year={2026},
  eprint={2606.20531},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.20531}
}

@article{isik2023humanrf,
  title = {HumanRF: High-Fidelity Neural Radiance Fields for Humans in Motion},
  author = {I\c{s}{\i}k, Mustafa and R{\"u}nz, Martin and Georgopoulos, Markos and Khakhulin, Taras
    and Starck, Jonathan and Agapito, Lourdes and Nie{\ss}ner, Matthias},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {42},
  number = {4},
  pages = {1--12},
  year = {2023},
  publisher = {ACM New York, NY, USA},
  doi = {10.1145/3592415},
  url = {https://doi.org/10.1145/3592415}
}

@article{yang2024gaussianobject,
  title   = {GaussianObject: High-Quality 3D Object Reconstruction from Four Views with Gaussian Splatting},
  author  = {Chen Yang and Sikuang Li and Jiemin Fang and Ruofan Liang and
             Lingxi Xie and Xiaopeng Zhang and Wei Shen and Qi Tian},
  journal = {ACM Transactions on Graphics},
  year    = {2024}
}

@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@misc{barron2023zipnerf,
      title={Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields},
      author={Jonathan T. Barron and Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
      year={2023},
      eprint={2304.06706},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

(`zipnerf/` in this release is built on [zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch); `3dgs-go/` is built on [GaussianObject](https://github.com/chensjtu/GaussianObject) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting); dataset conversion targets [ActorsHQ](https://actors-hq.com/) and Mip-NeRF 360/OmniObject3D as prepared by GaussianObject.)
