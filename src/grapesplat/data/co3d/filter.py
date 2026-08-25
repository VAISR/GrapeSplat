"""Filter modules for CO3D pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import CO3DDatasetAnno
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import orjson
import psutil
import random


@dataclass
class CO3DFilterBuilder(FilterBuilder):
    """Filter module for CO3D dataset annotation."""

    def __call__(self, root_path: str) -> "CO3DFilter":
        super().__call__(root_path)
        return CO3DFilter(config=self)


class CO3DFilter:
    """Filter module for CO3D dataset annotation."""

    def __init__(self, config: CO3DFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, CO3DDatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        pool = load_pool(self.config.root_path)
        args = [(cat_dir, self.config.seq_len_min) for cat_dir in pool]

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


def load_data(args: tuple[Path, int]) -> CO3DDatasetAnno:
    """Load data annotation per category."""

    cat_dir, seq_len_min = args
    cat_name = cat_dir.name
    anno_path = cat_dir / "frame_annotations.jgz"
    seq_id_to_anno: dict[str, list[dict]] = defaultdict(list)

    with gzip.open(anno_path, "rb") as f:
        for frame_anno in orjson.loads(f.read()):
            viewpoint = frame_anno["viewpoint"]
            frame = {
                "depth": {
                    "path": frame_anno["depth"]["path"],
                    "mask_path": frame_anno["depth"]["mask_path"],
                },
                "image": {"path": frame_anno["image"]["path"]},
                "viewpoint": {
                    "R": viewpoint["R"],
                    "T": viewpoint["T"],
                    "focal_length": viewpoint["focal_length"],
                    "principal_point": viewpoint["principal_point"],
                },
            }
            seq_id_to_anno[frame_anno["sequence_name"]].append(frame)

    return [
        (f"{cat_name}/{seq_id}", seq_anno)
        for seq_id, seq_anno in seq_id_to_anno.items()
        if len(seq_anno) >= seq_len_min
    ]


def load_pool(dir: Path) -> list[Path]:
    """Load pool of category directories from filesystem directory."""

    return list(p for p in dir.iterdir() if p.is_dir())
