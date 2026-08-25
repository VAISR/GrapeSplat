"""Filter modules for Hypersim pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import HypersimDatasetAnno, HypersimSequenceAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import csv
import gzip
import h5py
import orjson
import psutil
import random


@dataclass
class HypersimFilterBuilder(FilterBuilder):
    """Filter module for Hypersim dataset annotation."""

    def __call__(self, root_path: str) -> "HypersimFilter":
        super().__call__(root_path)
        return HypersimFilter(config=self)


class HypersimFilter:
    """Filter module for Hypersim dataset annotation."""

    def __init__(self, config: HypersimFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, HypersimDatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        pool_train = load_pool(self.config.root_path, "train")
        pool_val = load_pool(self.config.root_path, "val")
        pool_test = load_pool(self.config.root_path, "test")

        intr = load_intr(self.config.root_path)

        arg_train = [
            (scene_dir, self.config.seq_len_min, intr.get(scene_dir.name))
            for scene_dir in pool_train
        ]
        arg_valtest = [
            (scene_dir, self.config.seq_len_min, intr.get(scene_dir.name))
            for scene_dir in pool_val + pool_test
        ]

        with Pool(psutil.cpu_count(logical=False)) as p:
            data_train = list(
                tqdm(
                    p.imap_unordered(load_data, arg_train),
                    desc="Filtering",
                    dynamic_ncols=True,
                    total=len(arg_train),
                )
            )
            data_valtest = list(
                tqdm(
                    p.imap_unordered(load_data, arg_valtest),
                    desc="Filtering",
                    dynamic_ncols=True,
                    total=len(arg_valtest),
                )
            )

        data_train = [seq for d in data_train for seq in d]
        data_valtest = [seq for d in data_valtest for seq in d]

        size_train_all = len(data_train)
        size_train_sub = self.config.split_size_train
        size_valtest_all = len(data_valtest)
        size_valtest_sub = self.config.split_size_val + self.config.split_size_test

        if size_train_all < size_train_sub:
            raise ValueError(
                f"Not enough sequences: expected {size_train_sub}, got {size_train_all}"
            )
        if size_valtest_all < size_valtest_sub:
            raise ValueError(
                f"Not enough sequences: expected {size_valtest_sub}, got {size_valtest_all}"
            )

        index_val = self.config.split_size_val

        def sort_tuple(x: tuple):
            return x[0]

        data_train = sorted(data_train, key=sort_tuple)
        data_valtest = sorted(data_valtest, key=sort_tuple)
        random.seed(util.get_seed())
        random.shuffle(data_train)
        random.shuffle(data_valtest)

        data_train = sorted(data_train[0:size_train_sub], key=sort_tuple)
        data_val = sorted(data_valtest[0:index_val], key=sort_tuple)
        data_test = sorted(data_valtest[index_val:size_valtest_sub], key=sort_tuple)
        data = dict(test=data_test, train=data_train, val=data_val)

        self.config.split_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.config.split_path, "wb") as f:
            f.write(orjson.dumps(dict(meta=self.config.meta, data=data)))

        return data


def load_data(args: tuple[Path, int, list[float]]) -> HypersimDatasetAnno:
    """Load data annotation per scene."""

    scene_dir, seq_len_min, intr = args
    detail_dir = scene_dir / "_detail"

    with (detail_dir / "metadata_scene.csv").open() as f:
        for row in csv.DictReader(f):
            if row["parameter_name"] == "meters_per_asset_unit":
                meters_per_asset_unit = float(row["parameter_value"])
                break

    camera_dir = [d for d in detail_dir.iterdir() if d.is_dir() and d.name.startswith("cam_")]

    data: HypersimDatasetAnno = []
    for cam_dir in camera_dir:
        with h5py.File(cam_dir / "camera_keyframe_orientations.hdf5", "r") as f:
            orientation = f["dataset"][:].tolist()
        with h5py.File(cam_dir / "camera_keyframe_positions.hdf5", "r") as f:
            position = f["dataset"][:].tolist()

        geo_dir = scene_dir / "images" / f"scene_{cam_dir.name}_geometry_hdf5"
        frame_index = sorted(
            f.name.split(".")[1] for f in geo_dir.iterdir() if f.name.endswith(".depth_meters.hdf5")
        )

        if len(frame_index) < seq_len_min:
            continue

        frame = []
        for index in frame_index:
            i = int(index)
            frame.append(
                {
                    "frame_index": index,
                    "orientation": [v for row in orientation[i] for v in row],
                    "position": [v * meters_per_asset_unit for v in position[i]],
                }
            )

        seq_name = f"{scene_dir.name}/{cam_dir.name}"
        seq_anno: HypersimSequenceAnno = {
            "M_cam_from_uv": intr,
            "frame": frame,
        }
        data.append((seq_name, seq_anno))

    return data


def load_intr(root_path: Path) -> dict[str, list[float]]:
    """Load per-scene intrinsic parameters from camera parameters CSV file."""

    intr = {}
    with (
        root_path / "evermotion_dataset" / "analysis" / "metadata_camera_parameters.csv"
    ).open() as f:
        for row in csv.DictReader(f):
            intr[row["scene_name"]] = [
                float(row["M_cam_from_uv_00"]),
                float(row["M_cam_from_uv_11"]),
                float(row["M_cam_from_uv_02"]),
                float(row["M_cam_from_uv_12"]),
            ]
    return intr


def load_pool(root_path: Path, stage: str) -> list[Path]:
    """Load pool of scene directories from split CSV file."""

    scene = set()
    dataset_dir = root_path / "evermotion_dataset"
    with (dataset_dir / "analysis" / "metadata_images_split_scene_v1.csv").open() as f:
        for row in csv.DictReader(f):
            if row["split_partition_name"] == stage:
                scene.add(row["scene_name"])

    return [dataset_dir / "scenes" / s for s in scene]
