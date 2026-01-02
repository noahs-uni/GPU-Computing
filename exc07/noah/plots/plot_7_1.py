import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('7_1.csv')

# Group by size and blockSize, compute median IPS
median_df = df.groupby(['size', 'blockSize'])['ips'].median().reset_index()

# Get unique block sizes for different lines
block_sizes = sorted(median_df['blockSize'].unique())

# Create the plot
plt.figure(figsize=(10, 6))

for block_size in block_sizes:
    data = median_df[median_df['blockSize'] == block_size].sort_values('size')
    plt.plot(data['size'], data['ips'], marker='o', label=f'blockSize={block_size}')

plt.xlabel('Body Count')
plt.ylabel('Body-Body Operations / second')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('7_1.png', dpi=300)

# Group by size and blockSize, compute median IPS
median_df = df.groupby(['size', 'blockSize'])['time'].median().reset_index()

# Get unique block sizes for different lines
block_sizes = sorted(median_df['blockSize'].unique())

# Create the plot
plt.figure(figsize=(10, 6))

for block_size in block_sizes:
    data = median_df[median_df['blockSize'] == block_size].sort_values('size')
    plt.plot(data['size'], data['time'], marker='o', label=f'blockSize={block_size}')

plt.xlabel('Body Count')
plt.ylabel('Time (ms)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.tight_layout()
plt.savefig('7_1_time.png', dpi=300)
