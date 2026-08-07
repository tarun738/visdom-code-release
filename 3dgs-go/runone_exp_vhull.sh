#!/bin/bash
# Single-scene, first-stage 3DGS-GO reproduction: visual hull -> coarse 3DGS -> render.
#
# This is the "ours" (visible-domain) protocol from the original per-dataset scripts
#   GaussianObject/run_mip_vd.sh, GaussianObject/run_omni_vd.sh, GOActHQ/run_act_vd.sh
# reduced to the first stage only (no leave-one-out / LoRA fine-tuning / Gaussian repair).
#
# Note on --vd_K: GaussianObject/run_mip_vd.sh threads a --vd_K/--vd_k arg through
# visual_hull.py/train_gs.py, but its own driver (batch_mip.py) never actually supplies
# it, and this codebase (ported from GOActHQ, which never had --vd_K at all) doesn't
# support that flag. The plain (K-less) visible-domain visual hull below is what the
# reference results were actually produced with (confirmed by comparing point counts of
# the saved training input.ply against each visual_hull_*_vd*.ply candidate).
set -e

views=$1
seq=$2
outfl=$3
data_dir=$4
vhull_reso=$5
train_res=$6
sh_degree=$7
lambda_silhouette=$8
lambda_silhouette_interpCams=$9

echo "$seq views=$views vhull_reso=$vhull_reso train_res=$train_res sh_degree=$sh_degree lambda_silhouette=$lambda_silhouette lambda_silhouette_interpCams=$lambda_silhouette_interpCams data_dir=$data_dir"

python visual_hull.py \
    --sparse_id $views \
    --data_dir $data_dir/$seq \
    --reso $vhull_reso --not_vis --vd

python train_gs.py -s $data_dir/$seq \
    -m $outfl/gs_init/$seq/$views \
    -r $train_res --sparse_view_num $views --sh_degree $sh_degree \
    --init_pcd_name visual_hull_${views}_vd \
    --white_background --random_background --load_best --interpCams \
    --lambda_silhouette_interpCams $lambda_silhouette_interpCams --lambda_silhouette $lambda_silhouette

python render.py \
    -m $outfl/gs_init/$seq/$views \
    --sparse_view_num $views --sh_degree $sh_degree \
    --init_pcd_name visual_hull_${views}_vd \
    --white_background --skip_all --skip_train --load_best
