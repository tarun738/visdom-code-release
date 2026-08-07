# 3DGS-GO (sparse-view reproduction)

This is built on [GaussianObject](https://github.com/chensjtu/GaussianObject) and
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), extended with
sparse-view, visual-hull-guided initialization used in our paper. See the original repos for the
base 3DGS implementation, general usage, and citation.

## Install

Requires an existing PyTorch + CUDA install with a matching `nvcc` toolchain (needed to build
`diff-gaussian-rasterization` and `simple-knn` from source).

```
pip install -r requirements.txt
```

(`requirements.txt` includes the local `./submodules/...` packages, so this both installs
dependencies and builds the CUDA extensions.)

## Reproducing our results

Sparse-view camera splits (which cameras go into train/val per scene) live in
`../datasets/dataset_splits/`. Unlike zipnerf, this codebase reads splits from
`sparse_<N>.txt` / `sparse_test.txt` files sitting next to `sparse/0/` in each scene's COLMAP
dataset (0-indexed positions into the camera list sorted alphabetically by filename), so derive
those first from the `dataset_splits/` conf.json — see
[`../datasets/README.md`](../datasets/README.md#sparse-view-splits-conf_to_sparse_splitspy).

Download the datasets (Mip-NeRF 360, OmniObject3D, ActorsHQ) separately, convert them with the
scripts in `../datasets/` (see [`../datasets/README.md`](../datasets/README.md)), and point each
script at your local copy by editing its `DATA_DIR=<path to dataset>` line:

```
./run_exp_vhull_mip360.sh   # bonsai/garden/kitchen at 4/6/9 cameras
./run_exp_vhull_omni3d.sh   # 17 OmniObject3D scenes at 6/9 cameras
./run_exp_vhull_actor.sh    # ActorsHQ Actor01-08 at 5/8/12 cameras
```

Each calls `runone_exp_vhull.sh`: visual hull -> coarse 3DGS -> render (first stage only, no
leave-one-out / LoRA fine-tuning / Gaussian repair), matching the settings recovered from our
original experiment logs.
