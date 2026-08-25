"""
Run evaluation for GrapeSplat.

Usage:
    uv run run_eval
    uv run run_eval worker_count=4
    uv run run_eval 'method=[our_sem_encoded,ref_anysplat]'
"""

from hydra import compose, initialize
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from threading import Event
from tqdm.auto import tqdm
from typing import TypedDict
import shutil
import sys

PROC: list[Popen] = []
STOP = Event()


class EvalConfig(TypedDict):
    bench: dict[str, dict[str, list[str]]]
    ckpt_dir: str
    method: list[str]
    worker_count: int


def is_ours(method: str) -> bool:
    """Check if the method is one of ours (not a reference)."""
    return method.startswith("our_") or method == "default"


def launch(args: tuple[str, str, str, str, str]) -> None:
    """Launch one evaluation run and fail early on error."""

    name, data, view, method, ckpt_dir = args

    if STOP.is_set():
        return

    run_id = f"{data}-{view}-{method}"

    # Reset the run directory for a single-attempt history
    for dir in Path(".logs/wandb").glob(f"*-{run_id}"):
        tqdm.write(f"Removing previous run directory: {dir}")
        shutil.rmtree(dir)

    cmd = [
        "uv",
        "run",
        "run_once",
        "test",
        f"module/metric={name}",
        f"data={data}",
        f"+view@data={view}",
        f"module/pipeline={method}",
        f"trainer.logger.id={run_id}",
        f"trainer.logger.name={run_id}",
        "trainer.logger.offline=true",
    ]
    if is_ours(method):
        cmd.append(f"ckpt={Path(ckpt_dir, f'{method}.ckpt')}")

    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True)
    PROC.append(proc)
    out = proc.communicate()[0]

    if proc.returncode == 0 or STOP.is_set():
        return

    # Skip out-of-memory runs
    if any(v in out for v in ("OutOfMemoryError", "out of memory")):
        tqdm.write(f"Skipped OOM run {run_id}")
        return

    # Stop the bench on any other failure
    STOP.set()
    for p in PROC:
        p.terminate()
    raise RuntimeError(f"Failed evaluation run {run_id} ({proc.returncode}): {out}")


def main() -> None:
    with initialize(config_path="../../config/grapesplat", version_base="1.3"):
        config: EvalConfig = compose(config_name="eval", overrides=sys.argv[1:])

    # Verify checkpoints before launching
    for method in config["method"]:
        ckpt_path = Path(config["ckpt_dir"], f"{method}.ckpt")
        if is_ours(method) and not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: expected {ckpt_path}")

    # Expand bench table into run units
    unit = [
        (name, data, v, method, config["ckpt_dir"])
        for name, bench in config["bench"].items()
        for data, view in bench.items()
        for v in view
        for method in config["method"]
    ]
    for _, data, view, method, _ in unit:
        tqdm.write(f"Will run {data}-{view}-{method}")

    with ThreadPool(config["worker_count"]) as p:
        list(
            tqdm(
                p.imap_unordered(launch, unit),
                desc="Evaluating",
                dynamic_ncols=True,
                total=len(unit),
                unit="run",
            )
        )


if __name__ == "__main__":
    main()
