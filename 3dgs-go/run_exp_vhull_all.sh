#!/bin/bash
# Run the full first-stage vhull reproduction sweep across all three datasets:
#   Mip-NeRF 360 (3 scenes x 3 cam counts = 9 runs)
#   OmniObject3D (17 scenes x 3 cam counts = 51 runs)
#   ActorsHQ (8 actors x 3 cam counts = 24 runs)
# Total: 84 runs. Edit DATA_DIR in each run_exp_vhull_*.sh before running.
set -e
cd "$(dirname "$0")"

echo "############# MIP360 #############"
./run_exp_vhull_mip360.sh

echo "############# OMNI3D #############"
./run_exp_vhull_omni3d.sh

echo "############# ACTORSHQ #############"
./run_exp_vhull_actor.sh

echo "############# ALL DONE #############"
