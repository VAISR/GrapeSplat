"""Inspect module parameter sizes for VGGT-1B."""

from vggt.models.vggt import VGGT


def get_param_size(model):
    param_size = 0
    for p in model.parameters():
        param_size += p.numel()
    return param_size


def format_param_size(size):
    return f"{size / 1e6:.1f}M parameters"


def main():
    m = VGGT(enable_track=False, enable_point=False).train()
    s = get_param_size(m)
    print(f"VGGT-1B Full Module: {format_param_size(s)}")
    s = get_param_size(m.aggregator.patch_embed)
    print(f"VGGT-1B Image Encoder: {format_param_size(s)}")
    t = s
    s = get_param_size(m.aggregator)
    s -= t
    print(f"VGGT-1B Feature Aggregator: {format_param_size(s)}")
    s = get_param_size(m.camera_head)
    print(f"VGGT-1B Pose Decoder: {format_param_size(s)}")
    s = get_param_size(m.depth_head)
    print(f"VGGT-1B Depth Decoder: {format_param_size(s)}")


if __name__ == "__main__":
    main()
