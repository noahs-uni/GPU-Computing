"""
Load averaged GPU measurements (5_2_size.csv) and CPU measurements (5_2_cpu.csv),
average entries per matrix size, compute GPU total time (th2d+td2h+tkernel),
then plot speedup = cpu_time / gpu_time (both in ms) vs matrix size.

Usage (Linux):
  python3 plot_speedup_vs_cpu.py \
    --size-csv /home/noah/Documents/GPU/GPU-Computing/exc05/noah/plots/5_2/5_2_size.csv \
    --cpu-csv  /home/noah/Documents/GPU/GPU-Computing/exc05/noah/plots/5_2/5_2_cpu.csv \
    --outdir   ./plots
"""
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_SIZE_CSV = Path("/home/noah/Documents/GPU/GPU-Computing/exc05/noah/plots/5_2/5_2_size.csv")
DEFAULT_CPU_CSV = Path("/home/noah/Documents/GPU/GPU-Computing/exc05/noah/plots/5_2/5_2_cpu.csv")


def load_and_average_size(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = ['msize', 'th2d', 'td2h', 'tkernel']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in size CSV: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    grouped = df.groupby('msize', as_index=False)[['th2d', 'td2h', 'tkernel']].mean()
    grouped['total_ms'] = grouped['th2d'] + grouped['td2h'] + grouped['tkernel']
    return grouped


def load_and_average_cpu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = ['msize', 't']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in cpu CSV: {missing}")
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    # Treat CPU times as milliseconds (t in CSV is ms)
    grouped = df.groupby('msize', as_index=False)['t'].mean().rename(columns={'t': 'cpu_ms'})
    return grouped


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--size-csv', type=Path, default=DEFAULT_SIZE_CSV)
    p.add_argument('--cpu-csv', type=Path, default=DEFAULT_CPU_CSV)
    p.add_argument('--outdir', type=Path, default=Path('./plots'))
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    size_df = load_and_average_size(args.size_csv)
    cpu_df = load_and_average_cpu(args.cpu_csv)

    merged = pd.merge(size_df, cpu_df, on='msize', how='inner')
    if merged.empty:
        raise RuntimeError("No matching matrix sizes between size CSV and cpu CSV after grouping.")

    # Both CPU and GPU totals are in milliseconds -> compute speedup = cpu_ms / gpu_ms
    merged['gpu_total_ms'] = merged['total_ms']
    merged['speedup'] = merged['cpu_ms'] / merged['gpu_total_ms']

    merged['msize_num'] = pd.to_numeric(merged['msize'], errors='coerce')
    merged = merged.sort_values('msize_num')

    print("Averaged results per msize:")
    print(merged[['msize', 'cpu_ms', 'total_ms', 'gpu_total_ms', 'speedup']].to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(merged['msize_num'], merged['speedup'], marker='o', linestyle='-', linewidth=2)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Matrix size (msize)')
    ax.set_ylabel('Speedup (CPU ms / GPU ms)')
    ax.set_title('Speedup vs matrix size')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    out_png = args.outdir / 'speedup_vs_cpu.png'
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    out_csv = args.outdir / 'speedup_merged.csv'
    merged.to_csv(out_csv, index=False)

    print(f"Saved plot to {out_png}")
    print(f"Saved merged CSV to {out_csv}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])