#!/bin/bash
# Mip-NeRF 360, first stage only. Settings from GaussianObject/run_mip_vd.sh:
#   vhull_reso=2, train_res=4, sh_degree=3,
#   lambda_silhouette=0.1, lambda_silhouette_interpCams=0.01.
DATA_DIR=<path to dataset>
OUT_DIR=exps
scenes="bonsai kitchen garden"

for cam in 4 6 9
do
for s in $scenes
do
./runone_exp_vhull.sh $cam $s $OUT_DIR $DATA_DIR 2 4 3 0.1 0.01
done
done
