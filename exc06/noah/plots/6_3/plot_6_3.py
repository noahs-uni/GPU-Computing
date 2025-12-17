"""
Load 6_3.csv, compute median bandwidth for each (array_size, block_size) pair,
then plot bandwidth vs array_size (logarithmic y-axis) with one line per block_size.

Usage (Linux):
  python3 plot_6_3.py --csv /home/noah/Documents/GPU/GPU-Computing/exc06/noah/plots/6_3/6_3.csv \
                      --outdir ./plots
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation


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
    required = ['array_size', 'block_size', 'bandwidth']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Convert to numeric
    df['array_size'] = pd.to_numeric(df['array_size'], errors='coerce')
    df['block_size'] = pd.to_numeric(df['block_size'], errors='coerce')
    df['bandwidth'] = pd.to_numeric(df['bandwidth'], errors='coerce')

    # Group by (array_size, block_size) and compute median bandwidth
    grouped = df.groupby(['array_size', 'block_size'], as_index=False)['bandwidth'].median()

    print("Median data:")
    print(grouped)

    # Plot: bandwidth vs array_size (one line per block_size, logarithmic y-axis)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get unique block sizes and sort them
    block_sizes = sorted(grouped['block_size'].unique())

    for bs in block_sizes:
        data = grouped[grouped['block_size'] == bs].sort_values('array_size')
        ax.semilogy(data['array_size'], data['bandwidth'], marker='o', linewidth=2, 
                    markersize=6, label=f'block_size={int(bs)}')

    ax.set_xlabel('Array size')
    ax.set_ylabel('Bandwidth (bytes/sec, log scale)')
    ax.set_title('Bandwidth vs array size (median per configuration)')
    ax.grid(True, linestyle=':', alpha=0.6, which='both')
    ax.legend(loc='best', fontsize=10)

    # add more y-axis labels and formatter
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=100))

    fig.tight_layout()

    out_path = args.outdir / 'bandwidth_vs_array_size.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])