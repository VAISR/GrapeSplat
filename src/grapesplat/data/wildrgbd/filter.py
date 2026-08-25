"""Filter modules for WildRGBD pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import WildRGBDDatasetAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import orjson
import psutil
import random


@dataclass
class WildRGBDFilterBuilder(FilterBuilder):
    """Filter module for WildRGBD dataset annotation."""

    def __call__(self, root_path: str) -> "WildRGBDFilter":
        super().__call__(root_path)
        return WildRGBDFilter(config=self)


class WildRGBDFilter:
    """Filter module for WildRGBD dataset annotation."""

    def __init__(self, config: WildRGBDFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, WildRGBDDatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        pool_train = load_pool(self.config.root_path, "train")
        pool_valtest = load_pool(self.config.root_path, "val")

        arg_train = [
            (self.config.root_path, seq_name, self.config.seq_len_min) for seq_name in pool_train
        ]
        arg_valtest = [
            (self.config.root_path, seq_name, self.config.seq_len_min) for seq_name in pool_valtest
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


def load_data(args: tuple[Path, str, int]) -> WildRGBDDatasetAnno:
    """Load data annotation per scene."""

    root_path, seq_name, seq_len_min = args
    scene_dir = root_path / seq_name
    extr_path = scene_dir / "cam_poses.txt"
    meta_path = scene_dir / "metadata"

    frame = []
    with extr_path.open() as f:
        for line in f:
            parts = line.split()
            file_stem = f"{int(parts[0]):05d}"
            extr = [float(v) for v in parts[1:17]]
            frame.append({"file_stem": file_stem, "extr": extr})

    if len(frame) < seq_len_min:
        return []

    meta = orjson.loads(meta_path.read_bytes())
    seq_anno = {"K": meta["K"], "frame": frame}
    return [(seq_name, seq_anno)]


def load_pool(root_path: Path, stage: str) -> list[str]:
    """Load pool of sequence names from split JSON file for NVS tasks."""

    pool = []
    for cat_dir in root_path.iterdir():
        if not cat_dir.is_dir():
            continue

        cat_name = cat_dir.name
        split = orjson.loads((cat_dir / "nvs_list.json").read_bytes())
        for seq_id in split[stage]:
            pool.append(f"{cat_name}/{seq_id}")
    return pool
