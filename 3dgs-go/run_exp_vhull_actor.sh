#!/bin/bash
# ActorsHQ, first stage only. Settings from GOActHQ/run_act_vd.sh:
#   vhull_reso=1, train_res=1 (native resolution, no downsampling), sh_degree=3,
#   lambda_silhouette=0.1, lambda_silhouette_interpCams=0.01.
DATA_DIR=<path to dataset>
OUT_DIR=exps
actors="Actor01 Actor02 Actor03 Actor04 Actor05 Actor06 Actor07 Actor08"

for cam in 5 8 12
do
for a in $actors
do
./runone_exp_vhull.sh $cam $a $OUT_DIR $DATA_DIR 1 1 3 0.1 0.01
done
done
