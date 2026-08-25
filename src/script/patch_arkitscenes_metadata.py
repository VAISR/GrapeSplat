"""Patch ARKitScenes metadata sky_direction using diod orientation detection."""

from diod.model import get_orientation_model
from diod.utils import get_data_transforms, get_device, load_image_safely
from grapesplat.util import get_worker_count
from multiprocessing import Pool
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import tqdm
import argparse
import csv
import diod
import torch

# diod class = sky_direction mapping
# source: diod/config.py:17 ROTATIONS = {0: 0, 1: 90, 2: 180, 3: 270}
# class 0: upright 0° = Up
# class 1: rotated 90° = Left
# class 2: rotated 180° = Down
# class 3: rotated 270° = Right
DIRECTION = ("Up", "Left", "Down", "Right")


def list_scene_image_path(arg: tuple[int, Path, int]) -> list[tuple[int, Path]]:
    """List and sample image paths for one scene (runs in worker process)."""

    scene_idx, image_dir, image_per_scene = arg
    if not image_dir.exists():
        return []
    image_path_all = sorted(image_dir.iterdir())
    count = len(image_path_all)
    if count == 0:
        return []
    index = [i * count // image_per_scene for i in range(image_per_scene)]
    return [(scene_idx, image_path_all[i]) for i in index]


class SceneImageDataset(Dataset):
    """Flat dataset of (scene_index, image_tensor) for DataLoader pipelining."""

    def __init__(self, scene_list: list[dict], root_path: Path, image_per_scene: int, transform):
        self.transform = transform

        # Prefetch paths in parallel (HDD iterdir is slow)
        arg = [
            (
                scene_idx,
                root_path / "raw" / scene["fold"] / scene["video_id"] / "vga_wide",
                image_per_scene,
            )
            for scene_idx, scene in enumerate(scene_list)
        ]
        with Pool(get_worker_count()) as pool:
            result = list(
                tqdm(
                    pool.imap(list_scene_image_path, arg),
                    desc="Listing",
                    total=len(arg),
                )
            )
        self.item = [item for batch in result for item in batch]

        # Store per-scene sample paths (first, middle, last) for inspection
        self.scene_sample_path: dict[int, list[str]] = {}
        for batch in result:
            scene_idx = batch[0][0]
            path_all = [str(p) for _, p in batch]
            self.scene_sample_path[scene_idx] = [
                path_all[0],
                path_all[len(path_all) // 2],
            ]

    def __len__(self):
        return len(self.item)

    def __getitem__(self, idx):
        scene_idx, path = self.item[idx]
        img = load_image_safely(str(path))
        return scene_idx, self.transform(img)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_weight = Path(diod.__path__[0]).parent / "models" / "orientation_model_v2_0.9882.pth"

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-per-scene", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--root-path", type=Path, default=Path(".shared/datasets/ARKitScenes"))
    parser.add_argument("--weight-path", type=Path, default=default_weight)
    arg = parser.parse_args()

    # Read metadata
    metadata_path = arg.root_path / "raw" / "metadata.csv"
    with metadata_path.open() as f:
        row_all = list(csv.DictReader(f))

    scene_list = []
    for row in row_all:
        sky_direction = row["sky_direction"]
        if sky_direction not in DIRECTION:
            continue
        scene_list.append(
            {
                "sky_direction": sky_direction,
                "fold": row["fold"],
                "video_id": row["video_id"],
            }
        )

    # Load model
    device = get_device()
    model = get_orientation_model(pretrained=False)
    model.load_state_dict(torch.load(str(arg.weight_path), map_location=device))
    model.to(device)
    model.eval()

    transform = get_data_transforms()["val"]

    # Build dataset and dataloader
    dataset = SceneImageDataset(scene_list, arg.root_path, arg.image_per_scene, transform)
    loader = DataLoader(
        dataset,
        batch_size=arg.batch_size,
        multiprocessing_context="fork",
        num_workers=get_worker_count(),
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    # Predict
    pred_all = []
    progress = tqdm(total=len(loader), desc="Predicting", dynamic_ncols=True)
    with torch.inference_mode():
        for _, image_batch in loader:
            pred_all.append(model(image_batch.to(device)).argmax(dim=1).cpu())
            progress.update(1)
    progress.close()

    # Majority vote per scene
    pred_all = torch.cat(pred_all).reshape(len(scene_list), arg.image_per_scene)
    pred_dir_all = [DIRECTION[row.mode().values.item()] for row in pred_all]

    # Write delta CSV (only mismatches, sorted by confidence ascending)
    field = [
        "video_id",
        "fold",
        "sky_direction_meta",
        "sky_direction_pred",
        "confidence",
        "image_first",
        "image_middle",
    ]
    delta = []
    for scene_idx, scene in enumerate(scene_list):
        pred_dir = pred_dir_all[scene_idx]
        meta_dir = scene["sky_direction"]
        if pred_dir == meta_dir:
            continue
        vote = pred_all[scene_idx].bincount(minlength=4)
        confidence = vote.max().item() / vote.sum().item()
        sp = dataset.scene_sample_path[scene_idx]
        delta.append(
            dict(
                zip(
                    field,
                    [
                        scene["video_id"],
                        scene["fold"],
                        meta_dir,
                        pred_dir,
                        f"{confidence:.2f}",
                        sp[0],
                        sp[1],
                    ],
                )
            )
        )
    delta.sort(key=lambda r: float(r["confidence"]))

    arg.output_dir.mkdir(parents=True, exist_ok=True)
    path = arg.output_dir / "metadata_delta.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field)
        writer.writeheader()
        writer.writerows(delta)

    print(f"scenes={len(scene_list)} delta={len(delta)} match={len(scene_list) - len(delta)}")


if __name__ == "__main__":
    main()
