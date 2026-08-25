"""Filter modules for DTU pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import DTUDatasetAnno, DTUSequenceAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import orjson
import psutil
import random


@dataclass
class DTUFilterBuilder(FilterBuilder):
    """Filter module for DTU dataset annotation."""

    def __call__(self, root_path: str) -> "DTUFilter":
        super().__call__(root_path)
        return DTUFilter(config=self)


class DTUFilter:
    """Filter module for DTU dataset annotation."""

    def __init__(self, config: DTUFilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, DTUDatasetAnno]:
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


def load_data(args: tuple[Path, int]) -> DTUDatasetAnno:
    """Load data annotation per scene."""

    scene_dir, seq_len_min = args
    frame: DTUSequenceAnno = []
    for image_path in sorted((scene_dir / "images").iterdir()):
        file_id = int(image_path.name.split("_")[1]) - 1
        cam_path = scene_dir / "cameras" / f"{file_id:08d}_cam.txt"
        lines = cam_path.read_text().splitlines()
        extr = [float(v) for line in lines[1:5] for v in line.split()]
        intr = [float(v) for line in lines[7:10] for v in line.split()]
        frame.append({"file_id": file_id, "extr": extr, "intr": intr})

    if len(frame) < seq_len_min:
        return []

    return [(scene_dir.name, frame)]


def load_pool(dir: Path) -> list[Path]:
    """Load pool of scene directories from filesystem directory."""

    return list(p for p in dir.iterdir() if p.is_dir())
