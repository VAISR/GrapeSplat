"""
Draw the GrapeSplat evaluation figure set from offline wandb logs.

One pass, config-driven like run_eval.py: the bench table in
config/grapesplat/eval.yaml names every expected run directory, so
dataset order, view ladders, and OOM classification (in-ladder
holes) all follow config. test/metric_* is already the only legal
aggregation; this script only moves values around.

Usage:
    uv run python src/script/run_eval_log.py

Output (under fig_dir from the same config):
    metric.csv         exact-value ledger
    {plot,qual}_*.png  figure set (+ .pdf twins on paper plates)
"""

from hydra import compose, initialize
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from subprocess import run
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
import json
import numpy as np
import re
import shutil
import tempfile

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
# Type 42 keeps embedded fonts TrueType; AAAI forbids Type 3
plt.rcParams["pdf.fonttype"] = 42
# STIX keeps mathtext serif; the default dejavusans breaks Times pages
plt.rcParams["mathtext.fontset"] = "stix"


def load_eval() -> dict:
    """The eval experiment config from config/grapesplat/eval.yaml."""
    with initialize(config_path="../../config/grapesplat", version_base="1.3"):
        return compose(config_name="eval")


def parse_rec(view: str) -> int:
    """cXnY -> X, the rec view count."""
    return int(view.split("n")[0].removeprefix("c"))


CONFIG = load_eval()
BENCH: dict[str, list[str]] = {}
for table in CONFIG["bench"].values():
    for data, view in table.items():
        BENCH[data] = BENCH.get(data, []) + list(view)
DATASET = list(BENCH)
LADDER = {data: [parse_rec(v) for v in view] for data, view in BENCH.items()}
VIEW = sorted({rec for ladder in LADDER.values() for rec in ladder})

GROUP = {
    "image": ["psnr", "ssim", "lpips"],
    "depth1": ["absrel", "delta1"],
    "depth2_ren": ["absrel", "delta1"],
    "point": ["cd", "nc"],
    "pose": ["auc3", "auc30"],
    "efficiency": ["gib", "sec", "numel"],
}
# Display order; families are figure editorial choices, not run toggles.
# Published methods run oldest to newest by arXiv submission (2505, 2511,
# 2603, 2605, 2606) and ours closes the list as the newest
METHOD = [
    "ref_anysplat",
    "ref_da3",
    "ref_twoxplat",
    "ref_splatweaver",
    "ref_structsplat",
    "default",
    "our_voxel_affine",
    "our_voxel_peach",
    "our_gauss_clustered",
    "our_sem_encoded",
    "our_black_bgcolored",
]

COLOR = plt.get_cmap("tab10").colors
# tab10 lands our slot on a yellow-green that reads at 2.4:1 against white,
# under the 4.5:1 contrast the kit asks for, so our curve takes SplatWeaver's
# palette slot and SplatWeaver takes the ink
INK = {
    "our_gauss_clustered": COLOR[METHOD.index("ref_splatweaver")],
    "ref_splatweaver": (0.12, 0.12, 0.12),
}
REF = [m for m in METHOD if m.startswith("ref_")]
# Depth Anything 3 renders blurred novel views while scoring ahead on
# image metrics, so the main tables leave it to the supplement and the
# cost chart keeps it, where its budget is the point
CANON = [m for m in REF if m != "ref_da3"] + ["our_gauss_clustered"]
CANON_SUPP = REF + ["our_gauss_clustered"]
ABLATION = ["our_voxel_affine", "our_voxel_peach", "our_gauss_clustered"]
SUPP_SEM = ["our_voxel_peach", "our_sem_encoded"]
SUPP_BG = ["our_gauss_clustered", "our_black_bgcolored"]
EFF = CANON

# The main body carries four benchmarks and stops one rung short, where
# the cells start to crowd; the supplement carries every benchmark and
# the whole ladder, and the cost chart keeps the top rung
MAIN_DATASET = ["scannetpp", "nrgbd", "sevenscenes", "dl3dv"]
MAIN_VIEW = [4, 8, 16, 32]

OUT = Path(CONFIG["fig_dir"])

# Paper display names; the dot marks the render background
# (open = white, filled = black)
DISPLAY_DATA = {
    "dtu": "DTU",
    "eth3d": "ETH3D",
    "nrgbd": "NRGBD",
    "scannetpp": "ScanNet++",
    "sevenscenes": "7-Scenes",
    "dl3dv": "DL3DV",
}
DISPLAY = {
    "GT": "GT",
    "default": "VGGT",
    "our_black_bgcolored": "Ours (black)",
    "our_gauss_clustered": "Ours",
    "our_sem_encoded": "+Sem",
    "our_voxel_affine": "Voxel",
    "our_voxel_peach": "+PEACH",
    "ref_anysplat": "AnySplat",
    "ref_da3": "DA3",
    "ref_splatweaver": "SplatWeaver",
    "ref_structsplat": "StructSplat",
    "ref_twoxplat": "2Xplat",
}
# Ablation ladder reads as increments on the voxel base; the supp
# pair isolates the semantic branch on the peach base
DISPLAY_ABL = DISPLAY | {
    "our_black_bgcolored": "+BlackBg",
    "our_gauss_clustered": "+Cluster",
    "our_voxel_affine": "Voxel",
    "our_voxel_peach": "+PEACH",
}
DISPLAY_TEX = {
    "default": "VGGT",
    "our_black_bgcolored": "Ours (black)",
    "our_gauss_clustered": "Ours",
    "our_sem_encoded": "+Sem",
    "our_voxel_affine": "Voxel",
    "our_voxel_peach": "+PEACH",
    "ref_anysplat": "AnySplat",
    "ref_da3": "DA3",
    "ref_splatweaver": "SplatWeaver",
    "ref_structsplat": "StructSplat",
    "ref_twoxplat": "2Xplat",
}
DISPLAY_TEX_ABL = DISPLAY_TEX | {
    "our_gauss_clustered": "+Cluster",
    "our_voxel_affine": "Voxel",
    "our_voxel_peach": "+PEACH",
}
DISPLAY_TEX_SUPP = DISPLAY_TEX | {
    "our_black_bgcolored": "Ours (black)",
    "our_gauss_clustered": "Ours (white)",
    "our_sem_encoded": "+Sem",
    "our_voxel_peach": "+PEACH",
}
# The full grid needs two tables; one would run 80 pt past the text
# width even at the smallest type the kit allows
SUPP_LEFT = ["dtu", "eth3d", "nrgbd"]
SUPP_RIGHT = ["scannetpp", "sevenscenes", "dl3dv"]
DISPLAY_SUPP = DISPLAY | {
    "our_black_bgcolored": "Ours (black)",
    "our_gauss_clustered": "Ours (white)",
    "our_sem_encoded": "+Sem",
    "our_voxel_peach": "+PEACH",
}

# Podium fills, tinted toward white
BEAN_BEST = (0.60, 0.85, 0.62, 1.0)
BEAN_SECOND = (0.82, 0.92, 0.66, 1.0)
BEAN_NONE = (1.0, 1.0, 1.0, 1.0)
# Ink for a configuration that did not complete: gray text, not a fill, so
# it reads as absent rather than as another podium tint
OOM_GRAY = 0.55


def method_color(method: str) -> tuple[float, float, float]:
    """Line color of a method, ink first and the palette otherwise."""
    return INK.get(method, COLOR[METHOD.index(method)])


def is_lower_better(metric: str) -> bool:
    """Judge by the modality-free suffix (depth2_delta1 -> delta1)."""
    return metric.split("_")[-1] in {"lpips", "absrel", "cd", "gib", "sec", "numel"}


# Metric display symbols; modality lives in the caption
LABEL = {
    "absrel": "AbsRel",
    "auc3": r"AUC@$3^{\circ}$",
    "auc30": r"AUC@$30^{\circ}$",
    "cd": "CD",
    "delta1": r"$\delta_1$",
    "gib": "Mem (GiB)",
    "lpips": "LPIPS",
    "nc": "NC",
    "numel": r"$|\mathcal{G}|$",
    "psnr": "PSNR",
    "sec": "Time (s)",
    "ssim": "SSIM",
}


# Table headers live in math mode, where the figure labels would set
# their words in italic. The split puts the name on one line and its
# qualifier with the arrow on the next, so column width follows the
# wider of the two halves instead of the whole string
LABEL_TEX = {
    "absrel": (r"\mathrm{AbsRel}", ""),
    "auc3": (r"\mathrm{AUC}", r"@3^{\circ}"),
    "auc30": (r"\mathrm{AUC}", r"@30^{\circ}"),
    "delta1": (r"\delta_1", ""),
    "gib": (r"\mathrm{Memory}", r"\ (\mathrm{GiB})"),
    "lpips": (r"\mathrm{LPIPS}", ""),
    "numel": (r"|\mathcal{G}|", r"\ (\mathrm{M})"),
    "psnr": (r"\mathrm{PSNR}", ""),
    "sec": (r"\mathrm{Time}", r"\ (\mathrm{s})"),
    "ssim": (r"\mathrm{SSIM}", ""),
}


def tex_metric(metric: str) -> str:
    arrow = r"\downarrow" if is_lower_better(metric) else r"\uparrow"
    name, qualifier = LABEL_TEX[metric.split("_")[-1]]
    return f"${name}{qualifier}{arrow}$"


def metric_label(metric: str) -> str:
    arrow = r"$\downarrow$" if is_lower_better(metric) else r"$\uparrow$"
    return LABEL[metric.split("_")[-1]] + arrow


def per_column_bean(data: np.ndarray, descending: bool = False) -> np.ndarray:
    """Per-column podium fill among methods: best and second take a bean
    each and the rest stay white, so a near tie no longer paints like a
    full-range gap. A two-entry column drops the second, since naming a
    runner-up among two is noise."""
    # (M, V) -> (M, V, 4)
    rgba = np.tile(BEAN_NONE, data.shape + (1,))
    for xi in range(data.shape[1]):
        live = np.flatnonzero(~np.isnan(data[:, xi]))
        if live.size < 2:
            continue
        rank = live[np.argsort(data[live, xi], kind="stable")]
        if not descending:
            rank = rank[::-1]
        rgba[rank[0], xi] = BEAN_BEST
        if live.size > 2:
            rgba[rank[1], xi] = BEAN_SECOND
    return rgba


def is_absent(group: str, dataset: str, method: str) -> bool:
    """Apply the metric presence rules."""
    if dataset == "dl3dv" and group in ("depth1", "depth2_ren", "point"):
        return True
    if group in ("depth1", "point") and method.startswith("ref_"):
        return True
    if method == "default" and group not in ("depth1", "point", "pose"):
        return True
    return False


def present_dataset(group: str) -> list[str]:
    """Datasets where the group exists at all (dl3dv has no depth)."""
    return [d for d in DATASET if not all(is_absent(group, d, m) for m in METHOD)]


def present_method(group: str, family: list[str]) -> list[str]:
    """Family members that attend the group (rules dataset-agnostic sans dl3dv)."""
    return [m for m in family if not is_absent(group, "dtu", m)]


# Grid geometry in inches: every cell and lane the same size in every
# figure, no layout solver involved
CELL_W = 0.26
CELL_H = 0.18
LANE = 0.08
MARGIN = {"left": 0.62, "right": 0.05, "top": 0.3, "bottom": 0.3}

# Ring geometry in points: the innermost radius wraps the view count,
# every further ring sits one step outside it, and a ring is added per
# RING_UNIT million Gaussians past the RING_ONSET million the first
# ring already covers
RING_R0 = 6.0
RING_DR = 1.5
RING_ONSET = 1.0
RING_UNIT = 3.0
RING_MAX = 4

# Dash patterns for the cost chart, one per method in family order, so
# the chart survives the grayscale printing the kit asks us to expect
DASH = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def plot_heatmap(
    name: str,
    page: list[tuple[str, str, np.ndarray]],
    family: list[str],
    label: dict[str, str] = DISPLAY,
) -> None:
    """Template b. Annotated heatmap: y = method, x = view, facet =
    dataset; color = per-column rank, in-cell text = exact value.
    Every facet spans exactly its own view ladder, and every row
    names its own metric group, so mixed-modality pages share one
    figure; a facet absent from a row's group stays blank.
    """
    dataset = [d for d in DATASET if any(d in present_dataset(g) for g, m, v in page)]
    method = [present_method(g, family) for g, m, v in page]
    col = {d: sorted(LADDER[d]) for d in dataset}
    axes_w = [CELL_W * len(col[d]) for d in dataset]
    axes_h = [CELL_H * len(m) for m in method]
    fig_w = MARGIN["left"] + sum(axes_w) + (len(dataset) - 1) * LANE + MARGIN["right"]
    fig_h = MARGIN["top"] + sum(axes_h) + (len(page) - 1) * LANE + MARGIN["bottom"]
    fig, axes = plt.subplots(
        len(page),
        len(dataset),
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={
            "width_ratios": [len(col[d]) for d in dataset],
            "height_ratios": [len(m) for m in method],
        },
    )
    fig.subplots_adjust(
        left=MARGIN["left"] / fig_w,
        right=1 - MARGIN["right"] / fig_w,
        top=1 - MARGIN["top"] / fig_h,
        bottom=MARGIN["bottom"] / fig_h,
        wspace=LANE / float(np.mean(axes_w)),
        hspace=LANE / float(np.mean(axes_h)),
    )
    # A dataset absent from the first row would lose its name with the
    # facet, so the title goes on its topmost living facet instead
    head = {
        d: min(ri for ri, (g, m, v) in enumerate(page) if d in present_dataset(g)) for d in dataset
    }
    for ri, (group, metric, value) in enumerate(page):
        for ci, d in enumerate(dataset):
            ax = axes[ri][ci]
            if d not in present_dataset(group):
                ax.axis("off")
                continue
            draw_cell(ax, value, metric, d, method[ri], col[d])
            if ri == head[d]:
                ax.set_title(DISPLAY_DATA[d], fontsize=9)
            if ri < len(page) - 1:
                ax.set_xticks(range(len(col[d])), labels=[])
            if ci == 0:
                yticks = [label[m] for m in method[ri]]
                ax.set_yticks(range(len(method[ri])), labels=yticks, fontsize=6)
                # Upright metric label costs a lane instead of a margin,
                # which keeps the canvas narrow and the printed type
                # large; a facet too short to hold it lays it down
                # instead of letting it run into the neighbouring row
                upright = len(method[ri]) >= 4
                ax.set_ylabel(
                    metric_label(metric),
                    fontsize=8,
                    rotation=90 if upright else 0,
                    ha="center" if upright else "right",
                    va="center",
                )
            else:
                ax.set_yticks([])
    # Tight crop keeps labels inside the canvas and sheds leftover
    # margin without rescaling the cells
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tex_rgb(fill: tuple[float, float, float, float]) -> str:
    """A matplotlib rgba as the rgb triple cellcolor takes."""
    return ",".join(f"{c:.2f}" for c in fill[:3])


def latex_cell(value: float, rank: int, live: int) -> str:
    """One table entry. The podium fill carries the ranking and bold
    carries it again for a grayscale reader; a short column drops the
    runner-up, since naming a second among two is noise."""
    text = f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
    if rank == 0:
        return rf"\cellcolor[rgb]{{{tex_rgb(BEAN_BEST)}}}\textbf{{{text}}}"
    if rank == 1 and live > 2:
        return rf"\cellcolor[rgb]{{{tex_rgb(BEAN_SECOND)}}}{text}"
    return text


def emit_table(
    name: str,
    page: list[tuple[str, str, np.ndarray]],
    family: list[str],
    label: dict[str, str] = DISPLAY_TEX,
    ladder: list[int] = None,
    keep: list[str] = None,
) -> None:
    """Template t. LaTeX tabular of the whole grid: rows are metric by
    method, columns are benchmark by view count. A benchmark that does
    not carry a metric leaves dashes on that metric's rows, and an
    in-ladder hole reads as an out-of-memory run."""
    ladder = VIEW if ladder is None else ladder
    dataset = [d for d in DATASET if keep is None or d in keep]
    col = {d: [v for v in ladder if v in LADDER[d]] for d in dataset}
    edge = np.cumsum([1] + [len(col[d]) for d in dataset])
    row = [
        r"\small",
        r"\setlength{\tabcolsep}{0.8mm}",
        rf"\begin{{tabular}}{{l*{{{edge[-1] - 1}}}{{c}}}}",
        r"\toprule",
        "& "
        + " & ".join(rf"\multicolumn{{{len(col[d])}}}{{c}}{{{DISPLAY_DATA[d]}}}" for d in dataset),
        r"\\",
        "".join(rf"\cmidrule(lr){{{edge[i] + 1}-{edge[i + 1]}}}" for i in range(len(dataset))),
        "Method & " + " & ".join(str(v) for d in dataset for v in col[d]),
        r"\\",
    ]
    for group, metric, value in page:
        method = present_method(group, family)
        # The metric axis rides a spanning row rather than a column of
        # its own, which is the column the full view ladder needs
        row += [r"\midrule", rf"\multicolumn{{{edge[-1]}}}{{l}}{{{tex_metric(metric)}}}", r"\\"]
        for m in method:
            mi = METHOD.index(m)
            cell = []
            for d in dataset:
                di = DATASET.index(d)
                for v in col[d]:
                    vi = VIEW.index(v)
                    if is_absent(group, d, m):
                        cell.append("--")
                        continue
                    if np.isnan(value[di, mi, vi]):
                        # Gray reads apart from the podium tints in print
                        cell.append(rf"\textcolor[gray]{{{OOM_GRAY}}}{{OOM}}")
                        continue
                    live = sorted(
                        (
                            value[di, METHOD.index(o), vi]
                            for o in method
                            if not np.isnan(value[di, METHOD.index(o), vi])
                        ),
                        reverse=not is_lower_better(metric),
                    )
                    rank = live.index(value[di, mi, vi])
                    cell.append(latex_cell(value[di, mi, vi], rank, len(live)))
            row += [f"{label[m]} & " + " & ".join(cell), r"\\"]
    row += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(row) + "\n")


def draw_cell(
    ax: plt.Axes, value: np.ndarray, metric: str, dataset: str, method: list[str], ladder: list[int]
) -> None:
    """One heatmap facet: color = per-column rank scale, text = value.

    Columns span the given ladder only, so a hole is an OOM by set
    difference (only OOM runs get skipped), unless its whole method
    row is empty, which marks a pending eval instead.
    """
    di = DATASET.index(dataset)
    grid = value[di][np.ix_([METHOD.index(m) for m in method], [VIEW.index(v) for v in ladder])]
    if np.isnan(grid).all():
        ax.axis("off")
        return
    fill = per_column_bean(grid, descending=is_lower_better(metric))
    ax.imshow(fill, aspect="auto")
    # The kit demands a figure stay decipherable in grayscale, where the
    # three bean fills collapse into one tone, so the winner also carries
    # bold weight
    best = np.all(fill == BEAN_BEST, axis=-1)
    for yi in range(grid.shape[0]):
        pending = np.isnan(grid[yi]).all()
        for xi in range(grid.shape[1]):
            cell = grid[yi, xi]
            if np.isnan(cell):
                label = "--" if pending else "OOM"
                ax.text(xi, yi, label, ha="center", va="center", fontsize=6.5, color="0.45")
            else:
                # Sub-one values keep three decimals so near ties stay
                # distinguishable (0.996 vs 0.995)
                ax.text(
                    xi,
                    yi,
                    f"{cell:.3f}" if abs(cell) < 1 else f"{cell:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black",
                    fontweight="bold" if best[yi, xi] else "normal",
                )
    ax.set_xticks(range(len(ladder)), labels=[str(v) for v in ladder], fontsize=7)


def numel_ring(count: float) -> list[float]:
    """Nested marker areas for a Gaussian budget in millions. The count
    reads off the number of rings, so a long tail a log area would
    flatten stays countable, and the innermost ring keeps a fixed size
    that wraps the view count without touching it."""
    rings = int(np.clip(1 + np.floor((count - RING_ONSET) / RING_UNIT), 1, RING_MAX))
    return [(2 * (RING_R0 + ring * RING_DR)) ** 2 for ring in range(rings)]


def plot_eff_qe(name: str, value: dict, cost: list[str], dataset: str) -> None:
    """Template f. Quality-efficiency coupling: x = cost (log),
    y = lpips, ring count = numel, the marker chain sweeps v; an
    OOM truncates the sweep and gets named at the break."""
    method = present_method("efficiency", EFF)
    di = DATASET.index(dataset)
    lpips = value["image"]["lpips"][di]
    # Single column: one cost axis, so the chart sits beside its text
    # Drawn a shade narrower than the column it prints in, so the 9 pt labels
    # land above the 9 pt floor the kit sets for text inside a figure, and
    # tall enough that the rings of neighbouring rungs stop touching
    fig, axes = plt.subplots(1, len(cost), figsize=(3.2, 3.0), sharey=True, squeeze=False)
    for ax, xmetric in zip(axes[0], cost):
        for si, m in enumerate(method):
            mi = METHOD.index(m)
            x = value["efficiency"][xmetric][di, mi]
            y = lpips[mi]
            area = value["efficiency"]["numel"][di, mi]
            keep = ~(np.isnan(x) | np.isnan(y))
            if not keep.any():
                continue
            # Faint thin line keeps the sweep readable without the
            # crossing chains piling up into twigs; its dash pattern
            # carries the method where grayscale drops the color
            ax.plot(
                x[keep],
                y[keep],
                color=method_color(m),
                linestyle=DASH[si],
                linewidth=0.7,
                alpha=0.55,
                zorder=1.5,
            )
            # Concentric rings carry the Gaussian budget: a long tail
            # that log area flattens stays countable here
            for xv, yv, nv in zip(x[keep], y[keep], np.nan_to_num(area[keep])):
                for size in numel_ring(nv):
                    ax.scatter(
                        xv,
                        yv,
                        s=size,
                        facecolor="none",
                        edgecolor=method_color(m),
                        linewidth=0.9,
                        zorder=3,
                    )
            for view, xv, yv in zip(np.array(VIEW)[keep], x[keep], y[keep]):
                ax.annotate(str(view), (xv, yv), fontsize=9, ha="center", va="center", zorder=4)
            last = int(np.flatnonzero(keep)[-1])
            if last + 1 < len(VIEW) and VIEW[last + 1] in LADDER[dataset]:
                ax.annotate(
                    f"OOM@{VIEW[last + 1]}",
                    (x[last], y[last]),
                    xytext=(5, -8),
                    textcoords="offset points",
                    fontsize=9,
                    color=method_color(m),
                    alpha=0.9,
                )
        ax.set_xscale("log")
        ax.set_xlabel(f"{metric_label(xmetric)} (log)", fontsize=9)
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2g"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        # The outermost ring needs room past the last marker centre
        ax.margins(x=0.18, y=0.12)
        ax.tick_params(labelsize=9, length=1.5)
        ax.tick_params(which="minor", length=1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(metric_label("lpips"), fontsize=9)
    # Ring key drawn from the same rule as the markers, parked in the
    # empty low-cost low-error corner
    key = axes[0][0]
    key.text(0.38, 0.96, f"{LABEL['numel']} (M)", transform=key.transAxes, fontsize=9)
    for slot, count in enumerate((RING_ONSET, RING_ONSET + RING_UNIT, RING_ONSET + 2 * RING_UNIT)):
        spot = 0.40 + 0.12 * slot
        for size in numel_ring(count):
            key.scatter(
                spot,
                0.87,
                s=size,
                facecolor="none",
                edgecolor="0.45",
                linewidth=0.6,
                transform=key.transAxes,
                zorder=5,
            )
        key.text(
            spot,
            0.87,
            f"{count:.0f}",
            transform=key.transAxes,
            fontsize=9,
            ha="center",
            va="center",
        )
    proxy = [
        Line2D([], [], color=method_color(m), linestyle=DASH[si], label=DISPLAY[m])
        for si, m in enumerate(method)
    ]
    fig.legend(handles=proxy, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 0.99))
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def read_history_media(run: Path) -> list[dict[str, list[str] | str]]:
    """Per-scene media manifests from the run datastore, step order;
    filename lists keep the visualizer frame order. The store is
    copied out of the read-only mount first (open_for_scan is r+b)."""
    source = next(run.glob("run-*.wandb"))
    row = []
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / source.name
        shutil.copyfile(source, copy)
        store = datastore.DataStore()
        store.open_for_scan(str(copy))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            media = {}
            for item in record.history.item:
                key = tuple(item.nested_key) or (item.key,)
                if len(key) == 2 and key[1] == "filenames":
                    media[key[0].removeprefix("test/")] = json.loads(item.value_json)
                if len(key) == 2 and key[0] == "test/vis_seq_name" and key[1] == "path":
                    media["seq_name"] = json.loads(item.value_json)
            if media:
                row.append(media)
    return row


def read_seq_name(run: Path, media: dict) -> str:
    """Scene name of the manifest, decoded from the seq_name html."""
    if "seq_name" not in media:
        return ""
    text = (run / "files" / media["seq_name"]).read_text()
    return re.sub("<[^>]+>", "", text).strip()


def ren_frame_index(rec_count: int, seq_size: int) -> list[int]:
    """Novel-target frame indices: complement of the rec comb.

    rec = {i * seq_size // rec_count} in the data transform, ren is
    the rest; cXnY has seq_size = c + n = 3c.
    """
    rec = {i * seq_size // rec_count for i in range(rec_count)}
    return [i for i in range(seq_size) if i not in rec]


def read_media_pair(run: Path, name: list[str], slot: list[int]) -> np.ndarray | None:
    """Two views of one media filename list, side by side. (H, 2W, C)"""
    if not name:
        return None
    image = [plt.imread(run / "files" / name[s]) for s in slot]
    height = min(i.shape[0] for i in image)
    return np.concatenate([i[:height] for i in image], axis=1)


# Qual column order follows the table: published methods by arXiv date,
# ours last. The ground truth comes from the AnySplat run, which carries
# the same frame manifest as every other run of a scene
QUAL_GT = "ref_anysplat"
QUAL_ORDER = [
    "ref_anysplat",
    "ref_twoxplat",
    "ref_splatweaver",
    "our_gauss_clustered",
]
# Pick sheets carry the ablation and supp ladders as well, so one sheet
# per dataset answers every plate's scene choice at once
PICK_ORDER = (
    QUAL_ORDER[:1]
    + ["ref_da3"]
    + QUAL_ORDER[1:]
    + ["our_voxel_peach", "our_voxel_affine", "our_sem_encoded"]
)
# Ground-truth and prediction manifests of one modality
IMAGE_MEDIA = ("vis_data_image", "vis_pred_image_ren")
DEPTH_MEDIA = ("vis_data_depth", "vis_pred_depth2_ren")
# Scene picks per plate; the depth plate drops the benchmark that carries
# no depth ground truth
PICK_CANON = {"nrgbd": [2], "scannetpp": [0, 4], "sevenscenes": [2], "dl3dv": [26, 31]}
PICK_DEPTH = {"nrgbd": [2], "scannetpp": [0, 4], "sevenscenes": [2]}


def collect_qual_row(
    dataset: str,
    view: str,
    index: list[int],
    order: list[str] = None,
    media: tuple[str, str] = IMAGE_MEDIA,
) -> list[tuple[str, dict]]:
    """Aligned GT + method cells for the given history scene indices;
    frames pair through the run manifest, scenes through vis_seq_name.
    The media pair names the ground-truth and prediction manifests, so
    one plate reads renders and another reads rendered depth. Missing
    media leave a None cell."""
    order = QUAL_ORDER if order is None else order
    target = ren_frame_index(parse_rec(view), 3 * parse_rec(view))
    # One mid-sequence novel target: a single view per scene buys the
    # cell twice the print size at the same column count
    base = len(target) // 2
    slot = [target[base]]
    k = [target.index(s) for s in slot]
    run = {
        p.name.rsplit("-", 1)[-1]: p
        for p in Path(".logs/wandb").glob(f"offline-run-*-{dataset}-{view}-*")
    }
    gt = run.get(QUAL_GT)
    if gt is None:
        return []
    gt_row = read_history_media(gt)
    table = {}
    for m in order:
        if m in run:
            table[m] = {read_seq_name(run[m], r): (run[m], r) for r in read_history_media(run[m])}
    entry = []
    for i in index:
        if i >= len(gt_row):
            continue
        row = gt_row[i]
        scene = read_seq_name(gt, row)
        cell = {"GT": read_media_pair(gt, row.get(media[0], []), slot)}
        for m in order:
            found = table.get(m, {}).get(scene)
            cell[m] = read_media_pair(found[0], found[1].get(media[1], []), k) if found else None
        entry.append((f"{DISPLAY_DATA[dataset]}[{i}]", cell))
    return entry


def box_center(direction: str, width: int, height: int) -> tuple[int, int]:
    """Off-center focus for one view, as a human eye would drift."""
    off = 0.22
    x = width // 2 + int(off * width * ("R" in direction)) - int(off * width * ("L" in direction))
    y = (
        height // 2
        + int(off * height * ("D" in direction))
        - int(off * height * ("U" in direction))
    )
    return x, y


# Magnification of the framed square: a crop of one quarter the view
# height fills the neighbouring cell exactly at this factor, which
# holds the pair at the aspect of two side-by-side views
ZOOM = 4

# Off-center frame drift per scene row, keyed by the plate row tag so
# plates that share a row share its direction; a new row needs its own
# entry rather than a fallback. Frames aim at high-texture regions,
# where sharpness differences stay legible in print
QUAL_BOX = {
    "DTU[0]": ["DL"],
    "DTU[4]": ["U"],
    "DTU[6]": ["U"],
    "DL3DV[2]": ["L"],
    "DL3DV[4]": ["D"],
    "DL3DV[10]": ["U"],
    "DL3DV[8]": ["L"],
    "DL3DV[9]": [""],
    "DL3DV[12]": ["D"],
    "DL3DV[16]": ["L"],
    "DL3DV[26]": ["U"],
    "DL3DV[31]": ["UL"],
    "DL3DV[36]": [""],
    "DL3DV[38]": [""],
    "DL3DV[41]": ["U"],
    "DL3DV[45]": ["L"],
    "NRGBD[2]": ["UR"],
    "ScanNet++[0]": ["UR"],
    "ScanNet++[4]": ["R"],
    "7-Scenes[2]": ["R"],
}


def zoom_pair(image: np.ndarray, direction: str) -> np.ndarray:
    """One view beside a nearest-neighbour magnification of the square
    its red frame marks. (H, W, C) -> (H, 2W, C)"""
    tall, wide = image.shape[:2]
    half = tall // (2 * ZOOM)
    cx, cy = box_center(direction, wide, tall)
    crop = image[cy - half : cy + half, cx - half : cx + half]
    return np.concatenate([image, np.repeat(np.repeat(crop, ZOOM, 0), ZOOM, 1)], axis=1)


def qual_box(entry: list[tuple[str, dict]]) -> list[list[str]]:
    """Frame directions of a plate, in its own row order."""
    return [QUAL_BOX[tag] for tag, cell in entry]


# Distiller parameters for the plate pass: a low QFactor keeps the
# photographic content close to lossless while the pdf shrinks by an
# order. -dJPEGQ is silently ignored by ghostscript 9.50, so the
# QFactor dictionaries are the only knob that bites
DISTILL = (
    "<</ColorImageDict <</QFactor 0.25 /Blend 1"
    " /HSamples [1 1 1 1] /VSamples [1 1 1 1]>>"
    " /GrayImageDict <</QFactor 0.25 /Blend 1"
    " /HSamples [1 1 1 1] /VSamples [1 1 1 1]>>>> setdistillerparams"
)


def compress_plate(path: Path) -> None:
    """Re-encode the photos of a plate as jpeg in place, text left
    vector. Two ghostscript defaults have to be pinned: AutoRotatePages
    reads upright row labels as a sideways page and turns the whole
    plate, and pdfwrite emits a version pdflatex refuses to embed."""
    packed = path.with_suffix(".gs.pdf")
    run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-dAutoRotatePages=/None",
            "-dCompatibilityLevel=1.5",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={packed}",
            "-c",
            DISTILL,
            "-f",
            str(path),
        ],
        check=True,
    )
    packed.replace(path)


def draw_qual_plate(
    name: str,
    entry: list[tuple[str, dict]],
    order: list[str] = None,
    boxes: list[list[str]] = None,
    label: dict[str, str] = DISPLAY,
    paper: bool = True,
) -> None:
    """Qual grid on equal lanes: col = GT + methods, cell = one view.
    boxes[ri] = one U D L R direction for row ri, which frames a square
    at the same off-center spot in every column and pairs each view
    with a magnification of that square."""
    cell_w, lane = 1.7, 0.06
    left, right, top, bottom = 0.45, 0.05, 0.35, 0.08
    # A framed plate carries the magnification beside every view, which
    # halves the cell height at the same column width
    shape = next(c[k].shape for tag, c in entry for k in c if c[k] is not None)
    cell_h = cell_w * shape[0] / shape[1] / (1 if boxes is None else 2)
    col = QUAL_ORDER + ["GT"] if order is None else order
    fig_w = left + len(col) * cell_w + (len(col) - 1) * lane + right
    fig_h = top + len(entry) * cell_h + (len(entry) - 1) * lane + bottom
    fig, axes = plt.subplots(len(entry), len(col), figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(
        left=left / fig_w,
        right=1 - right / fig_w,
        top=1 - top / fig_h,
        bottom=bottom / fig_h,
        wspace=lane / cell_w,
        hspace=lane / cell_h,
    )
    for ri, (tag, cell) in enumerate(entry):
        for ci, key in enumerate(col):
            ax = axes[ri][ci]
            ax.axis("off")
            if cell[key] is not None:
                view = cell[key] if boxes is None else zoom_pair(cell[key], boxes[ri][0])
                # interpolation none keeps the source pixels intact in
                # the pdf instead of resampling to figure dpi
                ax.imshow(view, aspect="auto", interpolation="none")
                if boxes is not None:
                    tall, wide = cell[key].shape[:2]
                    half = tall // (2 * ZOOM)
                    cx, cy = box_center(boxes[ri][0], wide, tall)
                    ax.add_patch(
                        plt.Rectangle(
                            (cx - half, cy - half),
                            2 * half,
                            2 * half,
                            fill=False,
                            edgecolor="red",
                            linewidth=0.4,
                        )
                    )
            if ri == 0:
                ax.set_title(label[key], fontsize=14)
            if ci == 0:
                # Paper plates show the plain dataset name; pick sheets
                # keep the absolute history index for unambiguous picks
                # Upright name costs a lane instead of a margin
                ax.text(
                    -0.09,
                    0.5,
                    tag.split("[")[0] if paper else tag,
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    rotation=90,
                    fontsize=14,
                )
    # Paper plates keep source pixels (about 530 dpi at cell size);
    # draw tools stay light and png-only
    crop = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(OUT / f"{name}.png", dpi=530 if paper else 220, **crop)
    if paper:
        fig.savefig(OUT / f"{name}.pdf", **crop)
        compress_plate(OUT / f"{name}.pdf")
    plt.close(fig)


# Datastore metric key -> (group, metric) in the value layout
KEY = {
    "test/metric_compute_gib": ("efficiency", "gib"),
    "test/metric_compute_sec": ("efficiency", "sec"),
    "test/metric_depth1_absrel": ("depth1", "absrel"),
    "test/metric_depth1_delta1": ("depth1", "delta1"),
    "test/metric_depth2_ren_absrel": ("depth2_ren", "absrel"),
    "test/metric_depth2_ren_delta1": ("depth2_ren", "delta1"),
    "test/metric_image_ren_lpips": ("image", "lpips"),
    "test/metric_image_ren_psnr": ("image", "psnr"),
    "test/metric_image_ren_ssim": ("image", "ssim"),
    "test/metric_point_cd": ("point", "cd"),
    "test/metric_point_nc": ("point", "nc"),
    "test/metric_pose_rel_auc3": ("pose", "auc3"),
    "test/metric_pose_rel_auc30": ("pose", "auc30"),
    "test/metric_scene_numel": ("efficiency", "numel"),
}


def read_summary_metric(run: Path) -> dict[str, float]:
    """Final test/metric_* values; later summary records overwrite
    earlier. The store is copied out of the read-only mount first."""
    source = next(run.glob("run-*.wandb"))
    metric = {}
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / source.name
        shutil.copyfile(source, copy)
        store = datastore.DataStore()
        store.open_for_scan(str(copy))
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "summary":
                continue
            for item in record.summary.update:
                key = ".".join(item.nested_key) or item.key
                if key in KEY:
                    metric[key] = float(item.value_json)
    return metric


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Scan every expected run; holes stay absent, so an in-ladder
    # hole reads as an OOM downstream
    value = {
        group: {m: np.full((len(DATASET), len(METHOD), len(VIEW)), np.nan) for m in ms}
        for group, ms in GROUP.items()
    }
    row = ["dataset,view,method,group,metric,value"]
    for data, view in BENCH.items():
        for v in view:
            for method in METHOD:
                run = sorted(Path(".logs/wandb").glob(f"offline-run-*-{data}-{v}-{method}"))
                if not run:
                    continue
                for key, cell in read_summary_metric(run[-1]).items():
                    group, metric = KEY[key]
                    row.append(f"{data},{v},{method},{group},{metric},{cell}")
                    # numel in millions keeps the in-cell figure readable
                    scale = 1e-6 if metric == "numel" else 1.0
                    di, mi, vi = DATASET.index(data), METHOD.index(method), VIEW.index(parse_rec(v))
                    value[group][metric][di, mi, vi] = cell * scale
    (OUT / "metric.csv").write_text("\n".join(row) + "\n")
    print(f"scan: {len(row) - 1} values from .logs/wandb")

    # Main-body tables: every benchmark over the whole ladder, typeset
    # as LaTeX so the type stays at the size the kit asks for
    main_page = [
        ("image", "lpips", value["image"]["lpips"]),
        ("depth2_ren", "depth2_delta1", value["depth2_ren"]["delta1"]),
        ("pose", "pose_auc30", value["pose"]["auc30"]),
    ]
    emit_table("table_canon_main", main_page, CANON, ladder=MAIN_VIEW, keep=MAIN_DATASET)
    emit_table(
        "table_abl_main", main_page, ABLATION, DISPLAY_TEX_ABL, ladder=MAIN_VIEW, keep=MAIN_DATASET
    )
    # The supplement carries every metric over the full grid, split in
    # two because one table would run past the text width
    supp_page = [
        ("image", "psnr", value["image"]["psnr"]),
        ("image", "ssim", value["image"]["ssim"]),
        ("image", "lpips", value["image"]["lpips"]),
        ("depth2_ren", "depth2_absrel", value["depth2_ren"]["absrel"]),
        ("depth2_ren", "depth2_delta1", value["depth2_ren"]["delta1"]),
        *[("pose", m, value["pose"][m]) for m in GROUP["pose"]],
    ]
    for half, keep in (("a", SUPP_LEFT), ("b", SUPP_RIGHT)):
        emit_table(f"table_supp_canon_{half}", supp_page, CANON_SUPP, keep=keep)
        emit_table(f"table_supp_abl_{half}", supp_page, ABLATION, DISPLAY_TEX_ABL, keep=keep)
        emit_table(f"table_supp_sem_{half}", supp_page, SUPP_SEM, DISPLAY_TEX_SUPP, keep=keep)
        emit_table(f"table_supp_bg_{half}", supp_page, SUPP_BG, DISPLAY_TEX_SUPP, keep=keep)

    # Efficiency chart and its exact-value companion page
    plot_eff_qe("plot_canon_eff", value, ["gib"], "nrgbd")
    eff_page = [
        ("image", "lpips", value["image"]["lpips"]),
        ("efficiency", "gib", value["efficiency"]["gib"]),
        ("efficiency", "sec", value["efficiency"]["sec"]),
        ("efficiency", "numel", value["efficiency"]["numel"]),
    ]
    plot_heatmap("plot_canon_eff_exact", eff_page, CANON_SUPP)
    # The main body promises latency and reports none, so the supplement
    # carries the exact cost of every run beside the quality it bought
    for half, keep in (("a", SUPP_LEFT), ("b", SUPP_RIGHT)):
        emit_table(f"table_supp_eff_{half}", eff_page, CANON_SUPP, keep=keep)

    # Qual plates: canon picks, ablation and supp ladders (GT first,
    # top rung next), then the per-dataset pick sheets
    plate = []
    for d in DATASET:
        if d in PICK_CANON:
            plate += collect_qual_row(d, "c16n32", PICK_CANON[d])
    draw_qual_plate("qual_canon", plate, boxes=qual_box(plate))
    # The depth plate lives in the supplement, whose tables carry Depth
    # Anything 3, so the column set includes it where the main plate
    # does not; columns keep the arXiv time order with ours before GT
    depth_order = QUAL_ORDER[:1] + ["ref_da3"] + QUAL_ORDER[1:]
    plate = []
    for d in DATASET:
        if d in PICK_DEPTH:
            plate += collect_qual_row(d, "c16n32", PICK_DEPTH[d], depth_order, media=DEPTH_MEDIA)
    draw_qual_plate("qual_canon_depth", plate, depth_order + ["GT"])
    # Plate columns read in the same direction as the table rows, the base
    # first and the added step after it, and the ground truth closes the row
    # as the most recent record of the scene
    plate = collect_qual_row("dl3dv", "c16n32", [4, 16, 36], ABLATION)
    draw_qual_plate("qual_abl", plate, ABLATION + ["GT"], qual_box(plate), DISPLAY_ABL)
    plate = collect_qual_row("dl3dv", "c16n32", [8, 9, 12, 38], SUPP_SEM)
    draw_qual_plate("qual_supp", plate, SUPP_SEM + ["GT"], qual_box(plate), DISPLAY_SUPP)
    # Supplement plates for the limitation and domain readings: the DTU
    # failure beside the background variant and Depth Anything 3, and
    # the same column set on DL3DV, its training domain. Columns run
    # oldest to newest, published methods first, then the ours ladder,
    # and the ground truth closes as the most recent record
    supp_dtu = ["ref_anysplat", "ref_da3", "our_gauss_clustered", "our_black_bgcolored"]
    plate = collect_qual_row("dtu", "c16n32", [0, 4, 6], supp_dtu)
    draw_qual_plate("qual_supp_dtu", plate, supp_dtu + ["GT"], qual_box(plate), DISPLAY_SUPP)
    # Scene picks come from the reviewed pick list and stay disjoint
    # from every other plate of the main and supplementary set
    supp_dl3dv = ["ref_anysplat", "ref_da3", "our_gauss_clustered"]
    plate = collect_qual_row("dl3dv", "c16n32", [2, 10, 45], supp_dl3dv)
    draw_qual_plate("qual_supp_dl3dv", plate, supp_dl3dv + ["GT"], label=DISPLAY)
    for d in DATASET:
        entry = collect_qual_row(d, "c16n32", list(range(60 if d == "dl3dv" else 10)), PICK_ORDER)
        if entry:
            draw_qual_plate(
                f"qual_pick_{d}", entry, ["GT"] + PICK_ORDER, label=DISPLAY_ABL, paper=False
            )
    print("draw: figure set in output/figure")


if __name__ == "__main__":
    main()
