"""
Plot the 6 data series in 2.1.csv as 3 plots:
- Plot A: series 1 (async) + series 4 (sync)
- Plot B: series 2 (async) + series 5 (sync)
- Plot C: series 3 (async) + series 6 (sync)

Usage:
    python plot_2_1.py [path/to/2.1.csv]

Default CSV path is the file in the current workspace.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSV = Path("/home/noah/Documents/GPU/GPU-Computing/exc02/2.1.csv")


def load_groups(csv_path):
    text = Path(csv_path).read_text()
    # split into groups separated by one or more blank lines
    groups = [g.strip() for g in text.split("\n\n") if g.strip()]
    parsed = []
    for g in groups:
        lines = [ln.strip() for ln in g.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # parse header labels and numeric values
        labels = [s for s in lines[0].split(",") if s != ""]
        vals = [s for s in lines[1].split(",") if s != ""]
        # convert values to float
        vals_f = [float(v) for v in vals]
        parsed.append((labels, vals_f))
    return parsed


def plot_pairs(groups, out_path=None):
    # Expect 6 groups: 0..2 async, 3..5 sync
    if len(groups) < 6:
        raise RuntimeError(f"expected 6 groups in CSV, got {len(groups)}")
    pairs = [(0, 3), (1, 4), (2, 5)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    axes = axes[0]
    for ax, (a_idx, s_idx) in zip(axes, pairs):
        labels_a, vals_a = groups[a_idx]
        labels_s, vals_s = groups[s_idx]
        # Ensure consistent x axis: prefer labels from async group
        labels = labels_a
        x = np.arange(len(labels))
        # Truncate or pad series to match label length
        def align(vals, target_len):
            vals = list(vals)
            if len(vals) > target_len:
                return vals[:target_len]
            if len(vals) < target_len:
                return vals + [np.nan] * (target_len - len(vals))
            return vals

        vals_a = align(vals_a, len(labels))
        vals_s = align(vals_s, len(labels))

        ax.plot(x, vals_a, marker="o", label="async", color="#1f77b4")
        ax.plot(x, vals_s, marker="s", label="sync", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("configuration")
        ax.set_ylabel("time (ms)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        ax.set_title(f"Series {a_idx+1} (async) vs {s_idx+1} (sync)")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"saved figure to {out_path}")
    plt.show()


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    groups = load_groups(csv_path)
    plot_pairs(groups, out_path=Path(csv_path).with_suffix(".png"))