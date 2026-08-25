"""Filter modules for DL3DV pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import DL3DVDatasetAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import csv
import gzip
import orjson
import psutil
import pycolmap
import random


@dataclass
class DL3DVFilterBuilder(FilterBuilder):
    """Filter module for DL3DV dataset annotation."""

    def __call__(self, root_path: str) -> "DL3DVFilter":
        super().__call__(root_path)
        return DL3DVFilter(config=self)


class DL3DVFilter:
    """Filter module for DL3DV dataset annotation."""

    def __init__(self, config: DL3DVFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, DL3DVDatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        pool = load_pool(self.config.root_path)
        args = [(scene_dir, self.config.seq_len_min) for scene_dir in pool]

        with Pool(psutil.cpu_count(logical=False)) as p:
            data = list(
                tqdm(
                    p.imap_unordered(load_data, args),
                    desc="Filtering",
                    dynamic_ncols=True,
                    total=len(args),
                )
            )

        data = [seq for d in data for seq in d]

        size_sub = (
            self.config.split_size_train + self.config.split_size_val + self.config.split_size_test
        )
        size_all = len(data)
        if size_all < size_sub:
            raise ValueError(f"Not enough sequences: expected {size_sub}, got {size_all}")

        index_val = self.config.split_size_train
        index_test = size_sub - self.config.split_size_test

        def sort_tuple(x: tuple):
            return x[0]

        data = sorted(data, key=sort_tuple)
        random.seed(util.get_seed())
        random.shuffle(data)

        data_train = sorted(data[0:index_val], key=sort_tuple)
        data_val = sorted(data[index_val:index_test], key=sort_tuple)
        data_test = sorted(data[index_test:size_sub], key=sort_tuple)
        data = dict(test=data_test, train=data_train, val=data_val)

        self.config.split_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.config.split_path, "wb") as f:
            f.write(orjson.dumps(dict(meta=self.config.meta, data=data)))

        return data


def load_data(args: tuple[Path, int]) -> DL3DVDatasetAnno:
    """Load data annotation per scene."""

    scene_dir, seq_len_min = args
    rec = pycolmap.Reconstruction(scene_dir / "gaussian_splat" / "sparse" / "0")

    frame = []
    for image in sorted(rec.images.values(), key=lambda image: image.name):
        image: pycolmap.Image
        if not (scene_dir / "gaussian_splat" / "images" / image.name).is_file():
            continue
        extr_w2c = image.cam_from_world()
        extr = extr_w2c.rotation.quat.tolist() + extr_w2c.translation.tolist()
        frame.append({"extr": extr, "file_name": image.name})

    if len(frame) < seq_len_min:
        return []

    intr: pycolmap.Camera = next(iter(rec.cameras.values()))
    seq_anno = {"frame": frame, "intr": intr.params.tolist()}
    return [(scene_dir.name, seq_anno)]


def load_pool(root_path: Path) -> list[Path]:
    """Load pool of scene directories from benchmark metadata CSV file."""

    with (root_path / "benchmark-meta.csv").open() as f:
        return [root_path / row["hash"] for row in csv.DictReader(f)]
