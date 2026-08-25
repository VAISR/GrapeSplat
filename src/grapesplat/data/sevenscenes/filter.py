"""Filter modules for 7Scenes pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import SevenScenesDatasetAnno, SevenScenesSequenceAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import orjson
import psutil
import random


@dataclass
class SevenScenesFilterBuilder(FilterBuilder):
    """Filter module for SevenScenes dataset annotation."""

    def __call__(self, root_path: str) -> "SevenScenesFilter":
        super().__call__(root_path)
        return SevenScenesFilter(config=self)


class SevenScenesFilter:
    """Filter module for SevenScenes dataset annotation."""

    def __init__(self, config: SevenScenesFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, SevenScenesDatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        pool_trainval = load_pool(self.config.root_path, "train")
        pool_test = load_pool(self.config.root_path, "test")

        arg_trainval = [(scene_dir, self.config.seq_len_min) for scene_dir in pool_trainval]
        arg_test = [(scene_dir, self.config.seq_len_min) for scene_dir in pool_test]

        with Pool(psutil.cpu_count(logical=False)) as p:
            data_trainval = list(
                tqdm(
                    p.imap_unordered(load_data, arg_trainval),
                    desc="Filtering",
                    dynamic_ncols=True,
                    total=len(arg_trainval),
                )
            )
            data_test = list(
                tqdm(
                    p.imap_unordered(load_data, arg_test),
                    desc="Filtering",
                    dynamic_ncols=True,
                    total=len(arg_test),
                )
            )

        data_trainval = [seq for d in data_trainval for seq in d]
        data_test = [seq for d in data_test for seq in d]

        size_trainval_all = len(data_trainval)
        size_trainval_sub = self.config.split_size_train + self.config.split_size_val
        size_test_all = len(data_test)
        size_test_sub = self.config.split_size_test

        if size_trainval_all < size_trainval_sub:
            raise ValueError(
                f"Not enough sequences: expected {size_trainval_sub}, got {size_trainval_all}"
            )
        if size_test_all < size_test_sub:
            raise ValueError(f"Not enough sequences: expected {size_test_sub}, got {size_test_all}")

        index_val = self.config.split_size_train

        def sort_tuple(x: tuple):
            return x[0]

        data_trainval = sorted(data_trainval, key=sort_tuple)
        data_test = sorted(data_test, key=sort_tuple)
        random.seed(util.get_seed())
        random.shuffle(data_trainval)
        random.shuffle(data_test)

        data_train = sorted(data_trainval[0:index_val], key=sort_tuple)
        data_val = sorted(data_trainval[index_val:size_trainval_sub], key=sort_tuple)
        data_test = sorted(data_test[0:size_test_sub], key=sort_tuple)
        data = dict(test=data_test, train=data_train, val=data_val)

        self.config.split_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.config.split_path, "wb") as f:
            f.write(orjson.dumps(dict(meta=self.config.meta, data=data)))

        return data


def load_data(args: tuple[Path, int]) -> SevenScenesDatasetAnno:
    """Load data annotation per sub-sequence."""

    scene_dir, seq_len_min = args
    frame: SevenScenesSequenceAnno = []
    for extr_path in sorted(scene_dir.glob("*.pose.txt")):
        file_stem = extr_path.name.removesuffix(".pose.txt")
        extr = [float(v) for line in extr_path.read_text().splitlines() for v in line.split()]
        if any(v != v for v in extr):
            continue
        frame.append({"file_stem": file_stem, "extr": extr})

    if len(frame) < seq_len_min:
        return []

    seq_name = f"{scene_dir.parent.name}/{scene_dir.name}"
    return [(seq_name, frame)]


def load_pool(root_path: Path, stage: str) -> list[Path]:
    """Load pool of sub-sequence directories from per-scene split files."""

    split_name = f"{stage.title()}Split.txt"
    pool = []
    for scene_dir in root_path.iterdir():
        if not scene_dir.is_dir():
            continue
        split_path = scene_dir / split_name
        if not split_path.exists():
            continue
        for token in split_path.read_text().split():
            n = int(token.removeprefix("sequence"))
            seq_dir = scene_dir / f"seq-{n:02d}"
            if seq_dir.is_dir():
                pool.append(seq_dir)
    return pool
