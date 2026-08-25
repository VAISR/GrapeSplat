#!/bin/bash

set -euox pipefail

cd "$(dirname "$(readlink -f "$0")")"

# bulk downloads
wget -c https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
7z x multi_view_training_dslr_undistorted.7z -bsp1
rm multi_view_training_dslr_undistorted.7z

wget -c https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z
7z x multi_view_training_dslr_jpg.7z -bsp1
rm multi_view_training_dslr_jpg.7z

wget -c https://www.eth3d.net/data/multi_view_training_dslr_occlusion.7z
7z x multi_view_training_dslr_occlusion.7z -bsp1
rm multi_view_training_dslr_occlusion.7z

wget -c https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z
7z x multi_view_training_dslr_scan_eval.7z -bsp1
rm multi_view_training_dslr_scan_eval.7z

# per-scene depth downloads
scenes=("courtyard" "delivery_area" "electro" "facade" "kicker" "meadow" "office" "pipes" "playground" "relief" "relief_2" "terrace" "terrains")
for scene in "${scenes[@]}"; do
    wget -c https://www.eth3d.net/data/${scene}_dslr_depth.7z
    7z x ${scene}_dslr_depth.7z -bsp1
    rm ${scene}_dslr_depth.7z
done

