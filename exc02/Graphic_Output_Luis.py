import csv
import matplotlib.pyplot as plt

# Read in the files

rows = []
with open(r"C:\Technische Informatik Master\Module\GPU Computing\Exercise\Exercise02\Python_Output\result_Luis_2_1.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        clean_row = [item.strip() for item in row if item.strip() != '']
        if clean_row:
            rows.append(clean_row)

# Row 0: Asynchro Grid numberx1
# row 1: Asynchro Grid data
# row 2: Asynchro Block 1xBlockNumber
# row 3: Asynchro Block data
# Row 4: synchro Grid numberx1
# row 5: synchro Grid data
# row 6: synchro Block 1xBlockNumber
# row 7: synchro Block data
labels1 = rows[0]
values1 = list(map(float, rows[1]))
labels2 = rows[4]
values2 = list(map(float, rows[5]))



# ------ Plot or Grid size -------
plt.figure(figsize=(8,5))
plt.plot(labels1, values1, 'o-', label="grid size Asynchronious")
plt.plot(labels2, values2, 's--', label="grid size Synchronious")

plt.xlabel("Grid configuration")
plt.ylabel("Time (µs per launch)")
plt.title("CUDA Kernel Launch Scaling")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()


labels1 = rows[2]
values1 = list(map(float, rows[3]))
labels2 = rows[6]
values2 = list(map(float, rows[7]))

# ------ Plot or Block size -------
plt.figure(figsize=(8,5))
plt.plot(labels1, values1, 'o-', label="Block size Aynchronious")
plt.plot(labels2, values2, 's--', label="Block size Synchronious")

plt.xlabel("Grid configuration")
plt.ylabel("Time (µs per launch)")
plt.title("CUDA Kernel Launch Scaling")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# === Showing the plots ===
plt.savefig("cuda_scaling_plot.png", dpi=200)
plt.show()
