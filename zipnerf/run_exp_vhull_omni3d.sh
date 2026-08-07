DATA_DIR=<path to dataset>
scenes="backpack_016 box_043 broccoli_003 corn_007 dinosaur_006 flower_pot_007 gloves_009_1080 guitar_002 hamburger_012 picnic_basket_009 pineapple_013 sandwich_003 suitcase_006 timer_010_1080 toy_plane_005 toy_truck_037 vase_012"

# 004_00: (earlystop=6001, deg_view=4).
for s in $scenes
do
./runone_vhull_4.sh omni3d/$s/004_00 4 $DATA_DIR
done

# 006_00 / 009_00: full 25000 steps.
for camcount in 006_00 009_00
do
for s in $scenes
do
./runone_exp_vhull.sh omni3d/$s/$camcount $DATA_DIR
done
done
