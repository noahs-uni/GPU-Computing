import matplotlib.pyplot as plt
import pandas as pd
import os

# Konfig start
FILE_NAIVE  = 'exc5_3_2_naiv.csv'   # Datei mit Naiv-Daten
FILE_SHARED = 'exc5_3_2.csv'         # Datei mit Shared-Daten (oder exc5_3_2_shared.csv)

TARGET_BLOCK_DIM = 32
#Konfig end

def load_and_clean_csv(filename):
    """Liest eine CSV ein, entfernt Kommentare und bereinigt Formate."""
    data = []
    columns = ["matrixSize", "blockDim", "Time_H2D_ms", "Bandwidth_H2D", 
               "Time_D2H_ms", "Time_Kernel_ms", "Time_CPU_ms"]
    
    if not os.path.exists(filename):
        print(f"FEHLER: Datei '{filename}' nicht gefunden.")
        return pd.DataFrame()

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line: continue
            
            if line.endswith(';'): line = line[:-1]
            
            if "matrixSize" in line or line.startswith("#"):
                continue
            
            if line.startswith("-1") or "nicht shared" in line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 6:
                try:
                    row = [float(p) for p in parts[:7]]
                    data.append(row)
                except ValueError:
                    continue
    
    return pd.DataFrame(data, columns=columns)

df_naive = load_and_clean_csv(FILE_NAIVE)
df_shared = load_and_clean_csv(FILE_SHARED)

if df_naive.empty or df_shared.empty:
    exit()

# 2. Daten verknüpfen (Merge)
merged = pd.merge(df_naive, df_shared, 
                  on=['matrixSize', 'blockDim'], 
                  suffixes=('_naive', '_shared'))

if merged.empty:
    exit()

# 3. Filtern auf feste Blockgröße

subset = merged[merged['blockDim'] == TARGET_BLOCK_DIM].copy()

if subset.empty:
    print(f"FEHLER: Keine Daten für blockDim {TARGET_BLOCK_DIM} gefunden.")
    print("Verfügbare Blockgrößen:", merged['blockDim'].unique())
    exit()

subset = subset.sort_values('matrixSize')

subset['Speedup_Kernel'] = subset['Time_Kernel_ms_naive'] / subset['Time_Kernel_ms_shared']

subset['Total_Naive'] = (subset['Time_H2D_ms_naive'] + 
                         subset['Time_Kernel_ms_naive'] + 
                         subset['Time_D2H_ms_naive'])

subset['Total_Shared'] = (subset['Time_H2D_ms_shared'] + 
                          subset['Time_Kernel_ms_shared'] + 
                          subset['Time_D2H_ms_shared'])

subset['Speedup_Total'] = subset['Total_Naive'] / subset['Total_Shared']

plt.figure(figsize=(10, 6))

plt.plot(subset['matrixSize'], subset['Speedup_Kernel'], 
         marker='o', label=f'Kernel Speedup (bei BlockDim {TARGET_BLOCK_DIM})', 
         color='blue', linewidth=2)

plt.plot(subset['matrixSize'], subset['Speedup_Total'], 
         marker='s', label=f'Application Speedup (bei BlockDim {TARGET_BLOCK_DIM})', 
         color='green', linestyle='--', linewidth=2)

# keine Speedup werte
plt.axhline(y=1.0, color='gray', linestyle=':', alpha=0.8)

# Formatierung
plt.title(f'Speedup Analyse: Shared Memory vs. Naive\n(Blockgröße: {TARGET_BLOCK_DIM}x{TARGET_BLOCK_DIM})', fontsize=14)
plt.xlabel('Matrix Größe (Anzahl Elemente)', fontsize=12)
plt.ylabel('Speedup Faktor', fontsize=12)
plt.xscale('log') 

plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()