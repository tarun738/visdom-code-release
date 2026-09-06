# VisDom: Sparse Novel View Synthesis with Visible Domain Constraint
## $\color{red}{\text{Accepted to GCPR}}$

Code release for our paper. Two pipelines:

- **[`zipnerf/`](zipnerf/README.md)** — Zip-NeRF-based reconstruction. See [`zipnerf/README.md`](zipnerf/README.md) for install + how to reproduce our results.
- **[`3dgs-go/`](3dgs-go/README.md)** — 3D Gaussian Splatting-based reconstruction. See [`3dgs-go/README.md`](3dgs-go/README.md) for install + how to reproduce our results.

Both pipelines need converted datasets — see [Dataset preparation](#dataset-preparation) below. See [Citation](#citation) for the paper and the datasets/codebases this release builds on.

Released for **non-commercial research and evaluation use only** — see [`LICENSE.md`](LICENSE.md). `zipnerf/` and `3dgs-go/` carry their own component licenses ([`zipnerf/LICENSE`](zipnerf/LICENSE), [`3dgs-go/LICENSE.md`](3dgs-go/LICENSE.md)).

## Dataset preparation

Our two pipelines expect different dataset layouts:

- **`zipnerf/`** expects the **ActorsHQ** layout: `calibration.csv` + `rgbs/CamXXX/` + `masks/CamXXX/`.
- **`3dgs-go/`** expects a **COLMAP** dataset: `images/` + `masks/` + `sparse/0/{cameras,images,points3D}.bin`.

We evaluate on three sources, none of which natively matches both formats, so we convert. See
[`datasets/README.md`](datasets/README.md) for the conversion scripts, worked examples, and the
train/eval camera splits in [`datasets/dataset_splits/`](datasets/dataset_splits/).

## Reproducing the paper's numbers

Results obtained from this code differ slightly from the numbers printed in the paper. We
advise re-running the evaluation with this codebase rather than comparing against the
published tables cell by cell.

## Citation

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
