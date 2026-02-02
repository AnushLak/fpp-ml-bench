#!/usr/bin/env python3
"""
Parse GAN training losses from a text log and save:
  1) a tidy CSV of all parsed rows
  2) one PNG plot per metric (individual images)

Expected line format (example):
(epoch: 1, iters: 50, time: 0.198) G_GAN: 0.670 G_GAN_Feat: 1.601 ... D_fake: 0.942
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


LINE_RE = re.compile(
    r"^\(epoch:\s*(\d+),\s*iters:\s*(\d+),\s*time:\s*([0-9]*\.?[0-9]+)\)\s*(.*)\s*$"
)
KV_RE = re.compile(
    r"([A-Za-z0-9_]+):\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass
class Row:
    epoch: int
    iters: int
    time: float
    metrics: Dict[str, float]


def parse_log(path: str) -> Tuple[List[Row], List[str]]:
    """
    Returns:
      rows: parsed rows
      all_metric_names: sorted list of all metric keys observed
    """
    rows: List[Row] = []
    metric_names = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("==="):  # header/separator
                continue

            m = LINE_RE.match(line)
            if not m:
                # Not a loss line; skip quietly
                continue

            epoch = int(m.group(1))
            iters = int(m.group(2))
            t = float(m.group(3))
            tail = m.group(4)

            kvs = {}
            for k, v in KV_RE.findall(tail):
                try:
                    kvs[k] = float(v)
                    metric_names.add(k)
                except ValueError:
                    # Skip any weird value tokens
                    pass

            rows.append(Row(epoch=epoch, iters=iters, time=t, metrics=kvs))

    all_metric_names = sorted(metric_names)
    # sort in case the file has interleaving
    rows.sort(key=lambda r: (r.epoch, r.iters))
    return rows, all_metric_names


def compute_x_axes(rows: List[Row]) -> Tuple[List[int], List[float], List[int]]:
    """
    Creates three x-axes:
      - idx: simple sequential index (always monotonic)
      - epoch_frac: epoch + iters/max_iters_in_that_epoch (approx progress)
      - global_iter: sum(max_iters of prior epochs) + iters (approx absolute iters)
    """
    idx = list(range(1, len(rows) + 1))

    # max iters observed per epoch (works even if the log is truncated)
    epoch_max: Dict[int, int] = {}
    for r in rows:
        epoch_max[r.epoch] = max(epoch_max.get(r.epoch, 0), r.iters)

    epochs_sorted = sorted(epoch_max.keys())
    prefix_sum: Dict[int, int] = {}
    run = 0
    for e in epochs_sorted:
        prefix_sum[e] = run
        run += epoch_max[e]

    epoch_frac: List[float] = []
    global_iter: List[int] = []
    for r in rows:
        denom = epoch_max.get(r.epoch, 0) or 1
        epoch_frac.append(r.epoch + (r.iters / denom))
        global_iter.append(prefix_sum.get(r.epoch, 0) + r.iters)

    return idx, epoch_frac, global_iter


def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return values[:]
    out: List[Optional[float]] = [None] * len(values)
    buf: List[float] = []
    s = 0.0
    for i, v in enumerate(values):
        if v is None:
            # reset on missing
            buf.clear()
            s = 0.0
            out[i] = None
            continue
        buf.append(v)
        s += v
        if len(buf) > window:
            s -= buf.pop(0)
        if len(buf) == window:
            out[i] = s / window
        else:
            out[i] = None
    return out


def write_csv(rows: List[Row], metric_names: List[str], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["epoch", "iters", "time"] + metric_names
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rec = {"epoch": r.epoch, "iters": r.iters, "time": r.time}
            for k in metric_names:
                rec[k] = r.metrics.get(k, "")
            w.writerow(rec)


def plot_metrics(
    rows: List[Row],
    metric_names: List[str],
    out_dir: str,
    x_mode: str = "global_iter",
    smooth: int = 1,
    dpi: int = 160,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to save images. Install with: pip install matplotlib"
        ) from e

    os.makedirs(out_dir, exist_ok=True)

    idx, epoch_frac, global_iter = compute_x_axes(rows)
    if x_mode == "idx":
        x = idx
        x_label = "Log index"
    elif x_mode == "epoch_frac":
        x = epoch_frac
        x_label = "Epoch (fractional)"
    else:
        x = global_iter
        x_label = "Global iteration (approx)"

    # build metric series
    series: Dict[str, List[Optional[float]]] = {k: [] for k in metric_names}
    for r in rows:
        for k in metric_names:
            series[k].append(r.metrics.get(k, None))

    for k in metric_names:
        y = series[k]
        ys = moving_average(y, smooth) if smooth > 1 else y

        # If everything is missing, skip
        if all(v is None for v in y):
            continue

        plt.figure()
        # plot raw
        plt.plot(x, [v if v is not None else float("nan") for v in y], linewidth=1)
        # plot smoothed if requested
        if smooth > 1:
            plt.plot(
                x,
                [v if v is not None else float("nan") for v in ys],
                linewidth=2,
            )
            plt.legend(["raw", f"MA({smooth})"], loc="best")

        plt.xlabel(x_label)
        plt.ylabel(k)
        plt.title(k)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", k)
        out_path = os.path.join(out_dir, f"{safe}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
        plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse GAN loss logs and save per-metric PNG plots + CSV."
    )
    ap.add_argument("--log", required=True, help="Path to the loss text file.")
    ap.add_argument(
        "--out",
        required=True,
        help="Output directory (CSV + plots/ will be created here).",
    )
    ap.add_argument(
        "--x",
        choices=["global_iter", "epoch_frac", "idx"],
        default="global_iter",
        help="X-axis for plots.",
    )
    ap.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="Moving average window (>=1). Use 1 for no smoothing.",
    )
    ap.add_argument("--dpi", type=int, default=160, help="PNG DPI.")
    args = ap.parse_args()

    rows, metric_names = parse_log(args.log)
    if not rows:
        print("No loss lines parsed. Check the log format/path.")
        return 2

    out_csv = os.path.join(args.out, "losses.csv")
    out_plots = os.path.join(args.out, "plots")

    write_csv(rows, metric_names, out_csv)
    plot_metrics(
        rows,
        metric_names,
        out_dir=out_plots,
        x_mode=args.x,
        smooth=max(1, args.smooth),
        dpi=args.dpi,
    )

    print(f"Parsed rows: {len(rows)}")
    print(f"Metrics: {', '.join(metric_names)}")
    print(f"Wrote: {out_csv}")
    print(f"Plots: {out_plots}/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
