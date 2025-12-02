"""
Load 5_2_size.csv, average over 10 runs per msize,
then plot a single graph with 4 lines (logarithmic time scale) vs matrix size:
  - Host-to-Device (th2d)
  - Device-to-Host (td2h)
  - Kernel time (tkernel)
  - Total (sum)

Usage (Linux):
  python3 plot_5_2_size.py --csv /home/noah/Documents/GPU/GPU-Computing/exc05/noah/plots/5_2/5_2_size.csv \
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
    required = ['msize', 'th2d', 'td2h', 'tkernel']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Available: {list(df.columns)}")

    # Convert to numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Group by msize and average
    grouped = df.groupby('msize', as_index=False)[['th2d', 'td2h', 'tkernel']].mean()

    # Compute total execution time
    grouped['total'] = grouped['th2d'] + grouped['td2h'] + grouped['tkernel']

    # Sort by msize for plotting
    grouped = grouped.sort_values('msize')

    print("Averaged data:")
    print(grouped)

    # Create single plot with 4 lines
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.loglog(grouped['msize'], grouped['th2d'], marker='o', linewidth=2, markersize=6, label='Host-to-Device')
    ax.loglog(grouped['msize'], grouped['td2h'], marker='s', linewidth=2, markersize=6, label='Device-to-Host')
    ax.loglog(grouped['msize'], grouped['tkernel'], marker='^', linewidth=2, markersize=6, label='Kernel')
    ax.loglog(grouped['msize'], grouped['total'], marker='D', linewidth=2, markersize=6, label='Total')
    
    ax.set_xlabel('Matrix size (elements)')
    ax.set_ylabel('Time (ms, log scale)')
    ax.set_title('GPU execution time components vs matrix size')
    ax.grid(True, linestyle=':', alpha=0.6, which='both')
    ax.legend(loc='upper left', fontsize=10)
    
    # add more axis labels/ticks
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=100))
    ax.xaxis.set_minor_formatter(NullFormatter())
    
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs='all', numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    
    fig.tight_layout()
    out_png = args.outdir / 'gpu_times_vs_msize.png'
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    
    print(f"Saved combined plot to {out_png}")


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])