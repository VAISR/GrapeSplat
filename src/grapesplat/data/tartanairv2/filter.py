"""Filter modules for TartanAirV2 pipeline data."""

from ... import util
from ..filter import FilterBuilder
from ..typing import TartanAirV2DatasetAnno, TartanAirV2SequenceAnno
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import gzip
import orjson
import psutil
import random


@dataclass
class TartanAirV2FilterBuilder(FilterBuilder):
    """Filter module for TartanAirV2/TartanGround dataset annotation."""

    def __call__(self, root_path: str) -> "TartanAirV2Filter":
        super().__call__(root_path)
        return TartanAirV2Filter(config=self)


class TartanAirV2Filter:
    """Filter module for TartanAirV2/TartanGround dataset annotation."""

    def __init__(self, config: TartanAirV2FilterBuilder):
        self.config = config

    def __call__(self) -> dict[str, TartanAirV2DatasetAnno]:
        if self.config.split_path.exists():
            with gzip.open(self.config.split_path, "rb") as f:
                try:
                    split = orjson.loads(f.read())
                except Exception:
                    split = {}
            if split.get("meta") == self.config.meta:
                return split["data"]

        pool = load_pool(self.config.root_path)
        args = [(env_dir, self.config.seq_len_min) for env_dir in pool]

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


def load_data(args: tuple[Path, int]) -> TartanAirV2DatasetAnno:
    """Load data annotation per environment."""

    env_dir, seq_len_min = args
    seg_label_map = orjson.loads((env_dir / "seg_label_map.json").read_bytes())
    seg_label_map = {
        name: label for name, label in seg_label_map["name_map"].items() if name in ("sky",)
    }

    traj_dir = [
        d
        for version_dir in env_dir.iterdir()
        if version_dir.is_dir()
        for d in version_dir.iterdir()
        if d.is_dir()
    ]

    data: TartanAirV2DatasetAnno = []
    for traj in traj_dir:
        frame = []
        with (traj / "pose_lcam_front.txt").open() as f:
            for index, line in enumerate(f):
                extr = [float(v) for v in line.split()]
                frame.append({"extr": extr, "file_id": f"{index:06d}"})

        if len(frame) < seq_len_min:
            continue

        seq_name = f"{env_dir.name}/{traj.parent.name}/{traj.name}"
        seq_anno: TartanAirV2SequenceAnno = {
            "seg_label_map": seg_label_map,
            "frame": frame,
        }
        data.append((seq_name, seq_anno))

    return data


def load_pool(dir: Path) -> list[Path]:
    """Load pool of environment directories from filesystem directory."""

    return list(p for p in dir.iterdir() if p.is_dir())
