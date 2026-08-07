DATA_DIR=<path to dataset>

# 004_00: (earlystop=6001). garden uses deg_view=4, kitchen/bonsai use 1.
./runone_vhull_4.sh mipnerf360/garden/004_00 4 $DATA_DIR
./runone_vhull_4.sh mipnerf360/kitchen/004_00 1 $DATA_DIR
./runone_vhull_4.sh mipnerf360/bonsai/004_00 1 $DATA_DIR

# 006_00 / 009_00: full 25000 steps.
for camcount in 006_00 009_00
do
./runone_exp_vhull.sh mipnerf360/garden/$camcount $DATA_DIR
./runone_exp_vhull.sh mipnerf360/bonsai/$camcount $DATA_DIR
./runone_exp_vhull.sh mipnerf360/kitchen/$camcount $DATA_DIR
done
