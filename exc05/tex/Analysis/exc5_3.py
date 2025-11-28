import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

# --- HILFSFUNKTION ZUM BEREINIGEN ---
def clean_dataframe(df):
    # Leere Spalten entfernen
    df = df.dropna(axis=1, how='all')
    
    # Leerzeichen in Spaltennamen entfernen
    df.columns = [c.strip() for c in df.columns]
    
    # Semikolons in allen Spalten entfernen, die Objekte (Strings) sind
    # Das ist wichtig, da in Datei 2 das Semikolon am Ende der Zeile steht
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(';', '').str.strip()
            # Versuchen, zurück in Zahlen zu wandeln
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass # Falls es wirklich Text ist, lassen wir es so
                
    # Matrixbreite berechnen (Wurzel aus Gesamtgröße)
    if 'matrixSize' in df.columns:
        df['MatrixWidth'] = df['matrixSize'].apply(lambda x: int(math.sqrt(x)))
        
    return df

# --- 1. DATEIEN EINLESEN ---
try:
    df_shared = pd.read_csv('benchmark_results.csv', sep=',')
    df_shared = clean_dataframe(df_shared)
    print("Datei 1 (Shared) geladen.")
except FileNotFoundError:
    print("File 'benchmark_results.csv' could not be found")
    sys.exit()

try:
    df_noshared = pd.read_csv('benchmark_results_without_shared.csv', sep=',')
    df_noshared = clean_dataframe(df_noshared)
    print("Datei 2 (No Shared) geladen.")
except FileNotFoundError:
    print("File 'benchmark_results_without_shared.csv' could not be found")
    sys.exit()

# --- 2. ANALYSE (Beste Blockgröße basierend auf Shared Memory Performance) ---
# Wir optimieren auf die 'Shared Memory' Variante, da dies meist die Zielarchitektur ist.
max_width = df_shared['MatrixWidth'].max()
subset_large = df_shared[df_shared['MatrixWidth'] == max_width]

print(f"\n--- Analyse für Matrixbreite {max_width} (Basis: Shared Memory) ---")

best_block_size = 0
min_time = float('inf') 

for block_size in subset_large['blockDim'].unique():
    row = subset_large[subset_large['blockDim'] == block_size]
    if row.empty: continue
    
    # Spalte 'Time for Matrix Multiplication' aus Datei 1
    t_kernel = row['Time for Matrix Multiplication'].values[0]
    print(f"BlockSize {block_size}: {t_kernel:.4f} ms")
    
    if t_kernel < min_time:
        min_time = t_kernel
        best_block_size = block_size

print(f"\n>> GEWINNER: Optimale Blockgröße ist {best_block_size}")


# --- 3. DATEN VORBEREITEN ---

# A) Daten für Shared Memory (Datei 1) filtern
data_shared = df_shared[df_shared['blockDim'] == best_block_size].copy()

# CPU-Fix: Wir holen uns die maximalen CPU-Zeiten aus dem gesamten Datensatz (Datei 1),
# falls bei der besten Blockgröße zufällig 0 gemessen wurde.
real_cpu_times = df_shared.groupby('MatrixWidth')['Time for CPU'].max().reset_index()
if 'Time for CPU' in data_shared.columns:
    data_shared = data_shared.drop(columns=['Time for CPU'])
data_shared = pd.merge(data_shared, real_cpu_times, on='MatrixWidth', how='left')

# Gesamtlaufzeit GPU (Shared) berechnen
data_shared['TotalTimeGPU'] = (data_shared['Time to Copy to Device'] + 
                               data_shared['Time for Matrix Multiplication'] + 
                               data_shared['Time to Copy from Device'])
data_shared = data_shared.sort_values(by='MatrixWidth')


# B) Daten für No Shared / Global Memory (Datei 2) filtern
# Wir nutzen dieselbe Blockgröße, um einen fairen Vergleich im Diagramm zu haben
data_noshared = df_noshared[df_noshared['blockDim'] == best_block_size].copy()
data_noshared = data_noshared.sort_values(by='MatrixWidth')


# --- 4. PLOTTING ---
plt.figure(figsize=(12, 8))

# -- GPU Overhead (Kopieren) aus Datei 1 (stellvertretend für beide) --
plt.plot(data_shared['MatrixWidth'], data_shared['Time to Copy to Device'], 
         label='GPU: H2D Copy', marker='o', linestyle='--', alpha=0.5, color='gray', markersize=4)

plt.plot(data_shared['MatrixWidth'], data_shared['Time to Copy from Device'], 
         label='GPU: D2H Copy', marker='^', linestyle='--', alpha=0.5, color='silver', markersize=4)

# -- KERNEL VERGLEICH --

# 1. Kernel OHNE Shared Memory (Datei 2)
# Spaltenname aus Datei 2: 'Time for Matrix Multiplication without shared'
if 'Time for Matrix Multiplication without shared' in data_noshared.columns:
    plt.plot(data_noshared['MatrixWidth'], data_noshared['Time for Matrix Multiplication without shared'], 
             label='GPU: Kernel (Global Mem / No Shared)', color='magenta', marker='d', linestyle='-', linewidth=2)
else:
    print("Warnung: Spalte 'Time for Matrix Multiplication without shared' in Datei 2 nicht gefunden.")

# 2. Kernel MIT Shared Memory (Datei 1)
plt.plot(data_shared['MatrixWidth'], data_shared['Time for Matrix Multiplication'], 
         label='GPU: Kernel (Shared Mem)', color='blue', marker='x', linewidth=2)


# -- GPU Gesamtzeit (Shared Memory Variante) --
plt.plot(data_shared['MatrixWidth'], data_shared['TotalTimeGPU'], 
         label='GPU: Total Runtime (Shared)', color='black', linewidth=2.5)

# -- CPU Zeit --
cpu_data = data_shared[data_shared['Time for CPU'] > 0]
plt.plot(cpu_data['MatrixWidth'], cpu_data['Time for CPU'], 
         label='CPU Execution', color='red', marker='s', linestyle='-.', linewidth=2)


# Layout & Design
plt.xlabel('Matrix Width (N)')
plt.ylabel('Time (ms)')
plt.title(f'Performance: Shared vs. Global Memory (BlockSize {best_block_size})')
plt.legend() 
plt.grid(True, which="both", ls="-", alpha=0.4)

# Logarithmische Skala
plt.xscale('log')
plt.yscale('log')

# Speichern
output_filename = 'performance_comparison_shared_vs_global.png'
plt.savefig(output_filename)
print(f"Diagramm gespeichert als {output_filename}")

plt.show()