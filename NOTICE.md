# NOTICE

GrapeSplat
Copyright 2026 The GrapeSplat Authors

Original work under `src/` and `config/` is licensed under the Apache License, Version 2.0 (see `LICENSE`).

Third-party projects under `gitmodules/` keep their own licenses, each included in its own directory. Some allow non-commercial research use only, so this archive as a whole is for non-commercial research.

These copies differ from upstream:

- `gitmodules/vggt` gained gradient checkpointing in the DPT and camera heads, fp32 guards around both, a preallocated chunked DPT output, and a camera head that returns the final iteration instead of all of them
- `gitmodules/torchsparse` gained an int32 overflow fix in the implicit GEMM kernels, bf16 mma in place of fp16 with conv3d routed through autocast, a CUDA device guard for multi-GPU runs, and preserved spatial range in submanifold, transposed, and cat
- `gitmodules/anysplat` was packaged as an importable module, has AMP disabled in the Gaussian head, and passes near and far planes to the decoder
- `gitmodules/splatweaver` was packaged as an importable module, has AMP disabled in the Gaussian head, and had its CuRoPE binding restored so the extension builds
- `gitmodules/structsplat` was packaged as an importable module, gained `from_pretrained`, has AMP disabled in the Gaussian head, and loads its frozen towers lazily
- `gitmodules/twoxplat` was packaged as an importable module and runs its pose backbone in ambient precision
- `gitmodules/da3` has a narrowed AMP scope in the camera decoder and the dual DPT, and returns contiguous extrinsics
- `gitmodules/dvlt` gained shim wrappers around the model, gradient checkpointing in the encoders, and camera pose in the world-to-camera convention
- `gitmodules/lpips` gained activation checkpointing without per-layer accumulation, and a stable normalization
- `gitmodules/gsplat` accepts per-view opacities in the rasterizer
- `gitmodules/fsspec` stages the temporary file of a local write in the destination directory, so the final rename stays on one filesystem, and falls back to a static version when built outside a git checkout
- `gitmodules/fused-ssim` has a numerical stability fix in `ssim.cu`
- `gitmodules/scannetpp` gained COLMAP-based frame filtering, undistorted DSLR depth, restored depth PNG compression, and errors in place of silent skips
- `gitmodules/tartanair_v2` downloads through Hugging Face with large chunk sizes, and carries added download examples
- `gitmodules/diod` was packaged with setuptools

`doc/data/arkitscenes/metadata.csv` is derived from ARKitScenes metadata published by Apple Inc., with the `sky_direction` column corrected for 852 of 5071 scenes.
