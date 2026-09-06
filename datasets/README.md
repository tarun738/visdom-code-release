# Dataset preparation

Conversion scripts for the two dataset layouts the pipelines expect. Run these from this
`datasets/` directory.

Our two pipelines expect different dataset layouts:

- **`zipnerf/`** expects the **ActorsHQ** layout: `calibration.csv` + `rgbs/CamXXX/` + `masks/CamXXX/`.
- **`3dgs-go/`** expects a **COLMAP** dataset: `images/` + `masks/` + `sparse/0/{cameras,images,points3D}.bin`.

We evaluate on three sources, none of which natively matches both formats, so we convert:

| Source | Native format | Needed by | Conversion |
|---|---|---|---|
| [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) | COLMAP | zipnerf | `colmap_to_actorshq.py` |
| [OmniObject3D](https://omniobject3d.github.io/) (as prepared by [GaussianObject](https://github.com/GaussianObject/GaussianObject)) | COLMAP | zipnerf | `colmap_to_actorshq.py` |
| [ActorsHQ](https://www.actors-hq.com/) | ActorsHQ csv | 3dgs-go | `actorshq_to_colmap.py` |

Download the raw datasets from their respective sources; the splits used for train/eval cameras per scene are in [`dataset_splits/`](dataset_splits/) (`mipnerf360/`, `omni3d/`, `exp_acthq_Actor/`).

### COLMAP -> ActorsHQ (`colmap_to_actorshq.py`)

Converts a COLMAP sparse model (Mip-NeRF 360, OmniObject3D/GaussianObject) into the ActorsHQ `calibration.csv` / `rgbs/` / `masks/` layout consumed by zipnerf.

Expects `<colmap_dir>/sparse/0/{images,cameras}.bin`, an image subdirectory (default `images_2`), and a `masks/` subdirectory with same-named masks.

```bash
python colmap_to_actorshq.py <colmap_dir> <outdir> \
    --image_subdir images_2 \
    --mask_subdir masks
```

zipnerf's `ActorsHQ` dataset loader expects the output at `<Config.data_dir>/<scene>/Sequence1/4x/`, so `<outdir>` should be e.g. `zipnerf-dataset/bonsai/Sequence1/4x` with `Config.data_dir` set to `zipnerf-dataset`. Two things to get right:

- **`--image_subdir`**: which resolution to train on (`images`, `images_2`, `images_4`, ...) — this is baked into the output permanently, there's no runtime downsampling in the ActorsHQ loader. Check what resolution the target experiment actually used (e.g. by comparing image dimensions against any existing converted copy) rather than assuming from directory naming conventions elsewhere.
- **`--timestamp`**: must match the frame index recorded in the corresponding `dataset_splits/<scene>/<camcount>/splits/{train,test}.txt` file (e.g. `12`), since that's the filename suffix (`CamXXX_rgbNNNNNN.jpg`) the loader looks for. The default of `0` only works if the split file also says `0`.

Worked example — Mip-NeRF 360 bonsai, matching the existing `mipnerf360/bonsai/004_00` split (`images_4`, timestamp `12`):

```bash
python colmap_to_actorshq.py \
    <mip360_bonsai_colmap_dir> \
    zipnerf-dataset/bonsai/Sequence1/4x \
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
python actorshq_to_colmap.py --data <actorshq_actor_dir>/Sequence1/4x --outdir <outdir> --timestamp 12
```

Depends on `camera_actorshq.py` (ActorsHQ calibration.csv reader).

#### Sparse-view splits (`conf_to_sparse_splits.py`)

3dgs-go picks its N-camera train/test split from `sparse_<N>.txt` / `sparse_test.txt` files sitting next to `sparse/0/` in the COLMAP dataset — 0-indexed positions into the camera list sorted alphabetically by filename. zipnerf's splits (`dataset_splits/.../conf.json`) list the same cameras but as 1-indexed `CamNNN` numbers. Rather than maintaining two independently-curated split files per scene, derive the 3dgs-go one from the single `dataset_splits` source of truth:

```bash
python conf_to_sparse_splits.py dataset_splits/exp_acthq_Actor/Actor01/005_00/conf.json 3dgs-go-dataset/Actor01
```

(verified to reproduce the exact same camera sets as the reference `sparse_N.txt`/`sparse_test.txt` files, for both the ActorsHQ and Mip-NeRF 360 splits — order and trailing newline can differ, the selected cameras don't.)
