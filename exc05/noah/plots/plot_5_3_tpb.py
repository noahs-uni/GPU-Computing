"""
Load 5_2_tpb.csv, average over 10 runs per (msize, bsize) pair,
then plot total execution time vs block size (bsize) with logarithmic time scale.

Usage (Linux):
  python3 plot_5_2_tpb.py --csv /home/noah/Documents/GPU/GPU-Computing/exc05/noah/5_2_tpb.csv \
                          --outdir ./plots
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--outdir', type=Path, default=Path('./plots'))
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    df.columns = [c.strip() for c in df.columns]

    # Expected columns
    required = ['msize', 'bsize', 'th2d', 'td2h', 'tkernel']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Convert to numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Group by (msize, bsize) and average
    grouped = df.groupby(['msize', 'bsize'], as_index=False)[['th2d', 'td2h', 'tkernel']].mean()

    # Compute total execution time
    grouped['total'] = grouped['th2d'] + grouped['td2h'] + grouped['tkernel']

    # Sort by bsize for plotting
    grouped = grouped.sort_values('bsize')

    print("Averaged data:")
    print(grouped)

    # Plot: total execution time vs bsize (logarithmic time scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(grouped['bsize'], grouped['total'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Block size (bsize)')
    ax.set_ylabel('Total execution time (ms, log scale)')
    ax.set_title(f'Total execution time vs block size (msize={grouped["msize"].iloc[0]})')
    ax.grid(True, linestyle=':', alpha=0.6, which='both')
    
    # add more y-axis labels
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    fig.tight_layout()

    out_path = args.outdir / 'total_vs_bsize_shared.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])