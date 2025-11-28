import pandas as pd
import matplotlib.pyplot as plt
import math

# --- DATEN EINLESEN ---
try:
    # sep=',' ist Standard, aber sicherheitshalber angegeben
    df = pd.read_csv('benchmark_results.csv', sep=',')
except FileNotFoundError:
    print("File 'benchmark_results.csv' could not be found")
    exit()
    
# Leere Spalten entfernen (durch das C++ Formatierungsproblem)
df = df.dropna(axis=1, how='all')

# Leerzeichen in Spaltennamen entfernen
df.columns = [c.strip() for c in df.columns]

# --- BEREINIGUNG ---
# Das Semikolon ist jetzt vermutlich in der 'Time for CPU' Spalte am Ende
target_cols = ['Time for CPU', 'Time for Matrix Multiplication']

for col in target_cols:
    if col in df.columns and df[col].dtype == object:
        # Entferne Semikolon und wandle in Zahl um
        df[col] = df[col].astype(str).str.replace(';', '').astype(float)

# Matrixbreite berechnen (Wurzel aus Gesamtgröße)
df['MatrixWidth'] = df['matrixSize'].apply(lambda x: int(math.sqrt(x)))

# --- ANALYSE ---
max_width = df['MatrixWidth'].max()
subset_large = df[df['MatrixWidth'] == max_width]

print(f"--- Analyse für Matrixbreite {max_width} ---")

best_block_size = 0
min_time = float('inf') 

# Wir gehen alle Blockgrößen durch um den GPU-Gewinner zu finden
for block_size in subset_large['blockDim'].unique():
    row = subset_large[subset_large['blockDim'] == block_size]
    if row.empty: continue
    
    t_kernel = row['Time for Matrix Multiplication'].values[0]
    print(f"BlockSize {block_size}: {t_kernel:.4f} ms")
    
    if t_kernel < min_time:
        min_time = t_kernel
        best_block_size = block_size

print(f"\n>> GEWINNER: Optimale Blockgröße ist {best_block_size}")

# --- DATEN VORBEREITEN ---

# 1. Wir nehmen die Zeilen der besten Blockgröße
best_df = df[df['blockDim'] == best_block_size].copy()

# 2. SPEZIAL-FIX FÜR CPU ZEITEN:
# Da dein Bash-Skript die CPU-Zeit vielleicht nur bei BlockSize 4 gemessen hat,
# könnte bei BlockSize 32 (best_df) eine 0 stehen.
# Wir holen uns pro MatrixWidth die maximale gemessene CPU-Zeit aus dem gesamten Datensatz.
real_cpu_times = df.groupby('MatrixWidth')['Time for CPU'].max().reset_index()

# Wir löschen die (eventuell 0 enthaltende) CPU-Spalte aus best_df und mergen die echte
if 'Time for CPU' in best_df.columns:
    best_df = best_df.drop(columns=['Time for CPU'])
best_df = pd.merge(best_df, real_cpu_times, on='MatrixWidth', how='left')


# 3. Gesamtlaufzeit GPU berechnen (H2D + Kernel + D2H)
best_df['TotalTimeGPU'] = (best_df['Time to Copy to Device'] + 
                           best_df['Time for Matrix Multiplication'] + 
                           best_df['Time to Copy from Device'])

# Sortieren für saubere Linien
best_df = best_df.sort_values(by='MatrixWidth')


# --- PLOTTING ---
plt.figure(figsize=(12, 7)) # Bild etwas größer machen

# Linie 1: Host to Device
plt.plot(best_df['MatrixWidth'], best_df['Time to Copy to Device'], 
         label='GPU: H2D Copy', marker='o', linestyle='--', alpha=0.7)

# Linie 2: Kernel
plt.plot(best_df['MatrixWidth'], best_df['Time for Matrix Multiplication'], 
         label='GPU: Kernel Only', marker='x', linewidth=2)

# Linie 3: Device to Host
plt.plot(best_df['MatrixWidth'], best_df['Time to Copy from Device'], 
         label='GPU: D2H Copy', marker='^', linestyle='--', alpha=0.7)

# Linie 4: GPU Gesamt
plt.plot(best_df['MatrixWidth'], best_df['TotalTimeGPU'], 
         label='GPU: Total Runtime', color='black', linewidth=2)

# Linie 5: CPU (NEU HINZUGEFÜGT)
# Wir plotten nur Punkte, wo die Zeit > 0 ist
cpu_data = best_df[best_df['Time for CPU'] > 0]
plt.plot(cpu_data['MatrixWidth'], cpu_data['Time for CPU'], 
         label='CPU Execution', color='red', marker='s', linestyle='-.', linewidth=2)

# Beschriftungen
plt.xlabel('Matrix Width (N)')
plt.ylabel('Time (ms)')
plt.title(f'Performance Comparison CPU vs GPU (BlockSize {best_block_size})')
plt.legend() 
plt.grid(True, which="both", ls="-", alpha=0.4)

# Logarithmische Skala ist hier essentiell
plt.xscale('log')
plt.yscale('log')

# Speichern
plt.savefig('performance_analysis.png')
print("Diagramm gespeichert als performance_analysis.png")

plt.show()

# --- ZUSATZ: SPEEDUP BERECHNEN ---
# Kleiner Bericht in der Konsole
print("\n--- Speedup (CPU / GPU Total) ---")
print(best_df[['MatrixWidth', 'TotalTimeGPU', 'Time for CPU']].to_string(index=False))