import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

FILENAME = 'exc5_3_2.csv'

def parse_csv_file(filename):
    
    data_rows = []
    columns = ["matrixSize", "blockDim", "Time_H2D_ms", "Bandwidth_H2D", 
               "Time_D2H_ms", "Time_Kernel_ms", "Time_CPU_ms"]
    
    if not os.path.exists(filename):
        print(f"Fehler: Die Datei '{filename}' wurde nicht gefunden.")
        return None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.endswith(';'): line = line[:-1]

            if "matrixSize" in line or line.startswith("#"):
                continue
            
            # bei "-1" (Beginn Naive Section)
            if line.startswith("-1"):
                break
            
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    row = [float(p) for p in parts[:7]]
                    data_rows.append(row)
                except ValueError:
                    continue

    return pd.DataFrame(data_rows, columns=columns)

df = parse_csv_file(FILENAME)

if df is not None and not df.empty:

    df['Total_Time_ms'] = (df['Time_H2D_ms'] + 
                           df['Time_Kernel_ms'] + 
                           df['Time_D2H_ms'])

    plt.figure(figsize=(8, 4))

    grouped = df.groupby('matrixSize')
    
    # Farben für plots
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped)))
    
    for (size, group), color in zip(grouped, colors):
        # Sortieren nach blockDim
        group = group.sort_values('blockDim')
        
        # Legendenlabel
        dim = int(np.sqrt(size))
        label_text = f"Matrix {dim} x {dim}"
        
        # PLOT: X=blockDim, Y=Total_Time_ms
        plt.plot(group['blockDim'], group['Total_Time_ms'], 
                 marker='o', markersize=4, label=label_text, color=color)

        plt.title('Gesamtlaufzeit (inkl. Transfer) vs. Block Dimension', fontsize=15)
    plt.xlabel('Block Dimension (Threads per Side)', fontsize=12)
    plt.ylabel('Total Time (ms) [Log Scale]', fontsize=12)
    
    # Y-Achse Logarithmisch
    plt.yscale('log')
    
    # X-Achse Limits setzen (damit der Plot nicht bei 5 anfängt, sondern bei 0.5)
    plt.xlim(0.5, 32.5)
    
    # HIER IST DIE ÄNDERUNG:
    # Wir definieren explizit die Ticks: 5, 10, 15, 20, 25, 30
    custom_ticks = range(5, 35, 5) 
    plt.xticks(custom_ticks) 
    
    # Grid
    plt.grid(True, which="both", ls="--", alpha=0.6)
    
    # Legende Außerhalb (wie eben besprochen)
    plt.legend(
        title="Problemgröße", 
        fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.show()

    print("Diagramm wurde erstellt.")
else:
    print("Keine Daten gefunden.")