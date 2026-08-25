# GrapeSplat

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/asherchen/grapesplat)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

GrapeSplat reconstructs a renderable 3D Gaussian scene from unposed, uncalibrated images in one forward pass. Gaussians live on a scene-level voxel grid instead of on pixels, so the grid sets the primitive count rather than the image resolution and the view count.

`NOTICE.md` lists the third-party components and what we changed in them.

## Install

Python 3.12, CUDA 12.8, [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/VAISR/GrapeSplat.git
cd GrapeSplat
MAX_JOBS=1 uv sync
```

Keep `MAX_JOBS=1`. Several packages under `gitmodules/` build CUDA extensions on the first sync, and compiling them in parallel exhausts host memory. Raising it is at your own risk.

## Storage layout

Configs reach large assets through project-relative entries you create. None of them is tracked.

- `.shared/projects` holds pretrained weights in Hugging Face `{owner}/{repo}` layout
- `.shared/datasets` holds datasets, one directory each
- `.model` holds checkpoints loaded by `ckpt=`
- `.logs` holds training logs
- `.data` holds cached split files

One link covers the first two: `ln -s <storage root> .shared`.

## Weights

Git LFS is the fastest transport, so skip the contents on clone and pull them after:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/{owner}/{repo} .shared/projects/{owner}/{repo}
cd .shared/projects/{owner}/{repo} && git lfs pull
```

- `facebook/VGGT-1B` is the geometry backbone
- `facebook/dinov3-vith16plus-pretrain-lvd1689m` is the semantic backbone, used by the ablation only
- `lhjiang/anysplat` is a baseline
- `depth-anything/DA3-GIANT-1.1` is a baseline
- `Jeasco/SplatWeaver` is a baseline
- `DazzlingSun/structsplat` is a baseline
- `HwasikJeong/2Xplat` is a baseline, file `2xplat_dl3dv_hr.pt`

Both `facebook` repositories gate on accepting their license first. Through the Hugging Face CLI instead, disable Xet or each connection caps near 1 MB/s:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1 hf download {owner}/{repo}
```

Trained GrapeSplat checkpoints live at [asherchen/grapesplat](https://huggingface.co/asherchen/grapesplat), one `.ckpt` per config name under `config/grapesplat/module/pipeline/`. Download by the LFS recipe above, then point a run at one:

```bash
uv run run_once test module/pipeline=our_gauss_clustered ckpt=.model/our_gauss_clustered.ckpt
```

- `our_gauss_clustered.ckpt` is the main paper model
- `our_voxel_affine.ckpt` and `our_voxel_peach.ckpt` are the ablation ladder below it
- `our_sem_encoded.ckpt` and `our_black_bgcolored.ckpt` are the supplementary variants
- `default.ckpt` carries the initialization state for training from scratch

## Datasets

Place each under `.shared/datasets/` with the name below; `config/grapesplat/data/*.yaml` expects them.

- `ARKitScenes` comes from the snippet below
- `Hypersim` comes from `uv run gitmodules/hypersim/code/python/tools/dataset_download_images.py`
- `ScanNetpp` comes from https://kaldir.vc.in.tum.de/scannetpp/
- `TartanAirV2` comes from `uv run gitmodules/tartanair_v2/examples/download_dips_example.py`, with `DATA_ROOT` in it pointed at your copy
- `WildRGBD` comes from `uv run gitmodules/wildrgbd/download.py --cat all`
- `NRGBD` comes from `wget https://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip`
- `SevenScenes` comes from the 7-Scenes release, prepared with the Spann3R preprocessing script
- `DTU` comes from `bash src/script/down_dtu.sh`
- `ETH3D` comes from `bash src/script/down_eth3d.sh`
- `DL3DV-Benchmark` comes from https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark by the LFS recipe above; `benchmark-meta.csv` lists the 140 test scenes

ARKitScenes reads six assets across both splits, and its metadata needs the corrected copy from this repository:

```bash
for split in Training Validation; do
  uv run gitmodules/arkitscenes/download_data.py raw \
    --split $split \
    --download_dir .shared/datasets/ARKitScenes \
    --raw_dataset_assets lowres_wide lowres_wide.traj lowres_depth confidence vga_wide vga_wide_intrinsics
done
cp doc/data/arkitscenes/metadata.csv .shared/datasets/ARKitScenes/raw/metadata.csv
```

That file is the official metadata from https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv with `sky_direction` corrected for 852 of 5071 scenes, by the classifier in `src/script/patch_arkitscenes_metadata.py` followed by manual review. Our splits do not reproduce without it.

TartanAirV2 downloads only the front-left camera; the example script already sets that.

## Run

A Hydra config name, then overrides.

```bash
uv run run_once train module/pipeline={variant}
```

Swap `train` for `test` to evaluate. `{variant}` is one of `our_voxel_affine`, `our_voxel_peach`, `our_gauss_clustered`, or `our_sem_encoded` for the ablation chain, or a `ref_*` entry to run a baseline in the same harness. `data=` picks the benchmark and `+view@data=cXnY` takes X context views out of Y frames.

Metrics over a finished sweep, then the tables and figures:

```bash
uv run run_eval
uv run run_eval_log
```

`run_once` submits through SLURM. Other schedulers need the `srun` call at the bottom of `src/script/run_once.py` adapted.

## Citation

```bibtex
@misc{grapesplat2026,
  title={GrapeSplat: Geometry-Grounded Reconstruction via Amalgamated Pose-Free Encoding for Feed-Forward 3D Gaussian Splatting},
  author={TODO(release): author list in publication order},
  year={2026},
  eprint={TODO(release): arXiv identifier},
  archivePrefix={arXiv}
}
```
