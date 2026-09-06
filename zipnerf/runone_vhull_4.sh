#!/bin/bash
set -e
actvar=$1
deg_view=${2:-1}
data_dir=$3
echo $actvar deg_view=$deg_view data_dir=$data_dir

python train.py --gin_configs=configs/mip360_obj_4.gin --gin_bindings="Config.meta_exp='$actvar'" \
                  --gin_bindings="Config.data_dir = '${data_dir}'" \
                  --gin_bindings="Config.vhull=True" --gin_bindings="Config.earlystop=6001" \
                  --gin_bindings="PropMLP.deg_view=${deg_view}" --gin_bindings="NerfMLP.deg_view=${deg_view}" \
                  --gin_bindings="Config.exp_output_folder='exp'"

python eval.py --gin_configs=configs/mip360_obj_4.gin --gin_bindings="Config.meta_exp='$actvar'" \
                  --gin_bindings="Config.data_dir = '${data_dir}'" \
                  --gin_bindings="Config.vhull=True" --gin_bindings="Config.earlystop=6001" \
                  --gin_bindings="PropMLP.deg_view=${deg_view}" --gin_bindings="NerfMLP.deg_view=${deg_view}" \
                  --gin_bindings="Config.exp_output_folder='exp'"
