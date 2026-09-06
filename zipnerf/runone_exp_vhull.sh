#!/bin/bash
set -e
actvar=$1
data_dir=$2
echo $actvar data_dir=$data_dir

python train.py --gin_configs=configs/mip360_obj.gin --gin_bindings="Config.meta_exp='$actvar'" \
                  --gin_bindings="Config.data_dir = '${data_dir}'" \
                  --gin_bindings="Config.vhull=True" \
                  --gin_bindings="Config.exp_output_folder='exp'"

python eval.py --gin_configs=configs/mip360_obj.gin --gin_bindings="Config.meta_exp='$actvar'" \
                  --gin_bindings="Config.data_dir = '${data_dir}'" \
                  --gin_bindings="Config.vhull=True" \
                  --gin_bindings="Config.exp_output_folder='exp'"
