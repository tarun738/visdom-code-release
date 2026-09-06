#!/bin/bash
# OmniObject3D, first stage only.
# cam=4 settings from GaussianObject/run_omni_vd.sh:
#   vhull_reso=2, train_res=2, sh_degree=3,
#   lambda_silhouette=0.01, lambda_silhouette_interpCams=1.0
#   (reversed relative to mip360/actorshq, which use 0.1 / 0.01).
# cam=6,9 settings from GaussianObject/run_omni_vd69.sh ("om69"):
#   same vhull_reso/train_res/sh_degree, but lambda_silhouette=0.1,
#   lambda_silhouette_interpCams=0.01 (matching mip360/actorshq).
DATA_DIR="<path to dataset>"   # edit me
OUT_DIR=exps
scenes="backpack_016 box_043 broccoli_003 corn_007 dinosaur_006 flower_pot_007 gloves_009_1080 guitar_002 hamburger_012 picnic_basket_009 pineapple_013 sandwich_003 suitcase_006 timer_010_1080 toy_plane_005 toy_truck_037 vase_012"

# for s in $scenes
# do
# ./runone_exp_vhull.sh 4 $s $OUT_DIR $DATA_DIR 2 2 3 0.01 1.0
# done

for cam in 6 9
do
for s in $scenes
do
./runone_exp_vhull.sh $cam $s $OUT_DIR $DATA_DIR 2 2 3 0.1 0.01
done
done
