#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Step 1: download dtu.zip (~8.3 GB) via git LFS selective pull
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/depth-anything/DA3-BENCH _hfclone
(cd _hfclone && git lfs pull --include=dtu.zip)
mv _hfclone/dtu.zip ./
rm -rf _hfclone

# Step 2: unzip and clean
unzip -q dtu.zip
rm dtu.zip

# Step 3: reorganize into self-contained per-scan dirs
SCANS=(1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118)
for n in "${SCANS[@]}"; do
  scan="scan$n"
  mkdir -p "$scan/cameras"
  mv "dtu/Rectified/$scan" "$scan/images"
  mv "dtu/depth_raw/Depths/$scan" "$scan/depths"
  cp dtu/Cameras/*_cam.txt "$scan/cameras/"
  cp dtu/Cameras/pair.txt "$scan/"
  printf -v n3 "%03d" "$n"
  mv "dtu/Points/stl/stl${n3}_total.ply" "$scan/mesh.ply"
  mv "dtu/SampleSet/mvs_data/ObsMask/ObsMask${n}_10.mat" "$scan/obs_mask.mat"
  mv "dtu/SampleSet/mvs_data/ObsMask/Plane${n}.mat" "$scan/plane.mat"
done

# Step 4: drop the now-empty original tree
rm -rf dtu/
