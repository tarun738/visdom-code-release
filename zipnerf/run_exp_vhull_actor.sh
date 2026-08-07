DATA_DIR=<path to dataset>
actors="Actor01 Actor02 Actor03 Actor04 Actor05 Actor06 Actor07 Actor08"

for camcount in 005_00 008_00 012_00
do
for a in $actors
do
./runone_exp_vhull.sh exp_acthq_Actor/$a/$camcount $DATA_DIR
done
done
