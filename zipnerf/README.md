# ZipNeRF (sparse-view reproduction)

This is a fork of [SuLvXiangXin/zipnerf-pytorch](https://github.com/SuLvXiangXin/zipnerf-pytorch), extended with
sparse-view, visual-hull-guided training used in our paper. See the original repo for the base
ZipNeRF implementation, general usage, and citation.

## Install

```
# Make the conda environment (pinned to what we used to produce our results).
conda env create -f environment.yml
conda activate zipnerf

# Install the CUDA extensions into that environment.
pip install ./gridencoder
pip install ./nvdiffrast  # optional, for textured mesh
```

## Reproducing our results

Sparse-view camera splits (which cameras go into train/val per scene) live in `../datasets/dataset_splits/`,
one `conf.json` per `<dataset>/<scene>/<camN>_00/`. `internal/configs.py` resolves `Config.meta_exp`
against that directory automatically, so no extra flags are needed for this part.

Download the datasets (Mip-NeRF 360, OmniObject3D, ActorsHQ) separately and point each script at your
local copy by editing its `DATA_DIR=<path to dataset>` line:

```
./run_exp_vhull_mip360.sh   # bonsai/garden/kitchen at 4/6/9 cameras
./run_exp_vhull_omni3d.sh   # 17 OmniObject3D scenes at 4/6/9 cameras
./run_exp_vhull_actor.sh    # ActorsHQ Actor01-08 at 5/8/12 cameras
```

Each calls `runone_vhull_4.sh` (4-camera runs; `mip360_obj_4.gin`, early-stopped at 6001 steps) or
`runone_exp_vhull.sh` (6/8/9/12-camera runs; `mip360_obj.gin`, full 25000 steps) with `Config.vhull=True`,
matching the settings recovered from our original experiment logs.
