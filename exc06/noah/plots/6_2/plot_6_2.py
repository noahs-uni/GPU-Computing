"""
Load 6_2.csv, average bandwidth values for each size,
then plot bandwidth vs size (logarithmic scale).

Usage (Linux):
  python3 plot_6_2.py --csv /home/noah/Documents/GPU/GPU-Computing/exc06/noah/plots/6_2/6_2.csv \
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
    required = ['size', 'bandwidth']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Convert to numeric
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['bandwidth'] = pd.to_numeric(df['bandwidth'], errors='coerce')

    # Group by size and average bandwidth
    grouped = df.groupby('size', as_index=False)['bandwidth'].mean()
    grouped = grouped.sort_values('size')

    print("Averaged data:")
    print(grouped)

    # Plot: bandwidth vs size (logarithmic y-axis)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(grouped['size'], grouped['bandwidth'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Size')
    ax.set_ylabel('Bandwidth (bytes/sec, log scale)')
    ax.set_title('Bandwidth vs size')
    ax.grid(True, linestyle=':', alpha=0.6, which='both')
    
    # add more y-axis labels
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    fig.tight_layout()

    out_path = args.outdir / 'bandwidth_vs_size.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])