"""Filter modules for ARKitScenes pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import ARKitScenesDatasetAnno, ARKitScenesSequenceAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import csv
import gzip
import orjson
import psutil
import random


@dataclass
class ARKitScenesFilterBuilder(FilterBuilder):
    """Filter module for ARKitScenes dataset annotation."""

    def __call__(self, root_path: str) -> "ARKitScenesFilter":
        super().__call__(root_path)
        return ARKitScenesFilter(config=self)


class ARKitScenesFilter:
    """Filter module for ARKitScenes dataset annotation."""

    def __init__(self, config: ARKitScenesFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, ARKitScenesDatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        skydir = load_skydir(self.config.root_path / "raw" / "metadata.csv")
        root_path_train = self.config.root_path / "raw" / "Training"
        root_path_valtest = self.config.root_path / "raw" / "Validation"
        pool_train = load_pool(root_path_train)
        pool_valtest = load_pool(root_path_valtest)

        arg_train = [
            (root_path_train, seq_name, skydir.get(seq_name), self.config.seq_len_min)
            for seq_name in pool_train
            if skydir.get(seq_name) != "NA"
        ]
        arg_valtest = [
            (root_path_valtest, seq_name, skydir.get(seq_name), self.config.seq_len_min)
            for seq_name in pool_valtest
            if skydir.get(seq_name) != "NA"
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


def load_data(args: tuple[Path, str, str, int]) -> ARKitScenesDatasetAnno:
    """Load data annotation per scene."""

    root_path, seq_name, skydir, seq_len_min = args
    seq_dir = root_path / seq_name
    traj_path = seq_dir / "lowres_wide.traj"
    if not traj_path.exists():
        return []

    timestamp_to_extr = {}
    with traj_path.open() as f:
        for line in f:
            parts = line.split()
            timestamp_to_extr[f"{float(parts[0]):.3f}"] = [float(v) for v in parts[1:7]]

    timestamp_to_file_stem = {
        p.stem.split("_", 1)[1]: p.stem for p in (seq_dir / "vga_wide").iterdir()
    }
    timestamp_to_depth = {p.stem.split("_", 1)[1] for p in (seq_dir / "lowres_depth").iterdir()}

    # Match extr to depth and image by timestamp
    frame = []
    for timestamp, extr in sorted(timestamp_to_extr.items()):
        if not (timestamp in timestamp_to_file_stem and timestamp in timestamp_to_depth):
            continue

        file_stem = timestamp_to_file_stem[timestamp]
        intr_path = seq_dir / "vga_wide_intrinsics" / f"{file_stem}.pincam"
        intr = [float(v) for v in intr_path.read_text().split()]
        frame.append({"file_stem": file_stem, "extr": extr, "intr": intr})

    if len(frame) < seq_len_min:
        return []

    seq_anno: ARKitScenesSequenceAnno = {
        "parent_name": seq_dir.parent.name,
        "sky_direction": skydir,
        "frame": frame,
    }
    return [(seq_name, seq_anno)]


def load_pool(dir: Path) -> list[str]:
    """Load pool of sequence names from filesystem directory."""

    return list(p.name for p in dir.iterdir() if p.is_dir())


def load_skydir(path: Path) -> dict[str, str]:
    """Load sky direction per scene from metadata CSV file."""

    with path.open() as f:
        return {row["video_id"]: row["sky_direction"] for row in csv.DictReader(f)}
