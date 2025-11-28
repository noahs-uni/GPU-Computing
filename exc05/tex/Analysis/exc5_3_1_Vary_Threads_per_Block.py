import matplotlib.pyplot as plt
import pandas as pd
import os

# Name deiner CSV Datei
FILENAME = 'exc5_3_1.csv'

def parse_benchmark_file(filename):
    """Liest die spezielle CSV-Struktur ein und trennt Shared von Naive."""
    shared_data = []
    naive_data = []
    
    # Header für den DataFrame
    columns = ["matrixSize", "blockDim", "Time_H2D_ms", "Bandwidth_H2D", 
               "Time_D2H_ms", "Time_Kernel_ms", "Time_CPU_ms"]
    
    is_naive_section = False
    
    #if not os.path.exists(filename):
    #    print(f"Fehler: Datei '{filename}' nicht gefunden.")
    #    return None, None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Leere Zeilen überspringen
            if not line: continue
            
            # Semikolon am Ende entfernen, falls vorhanden
            if line.endswith(';'):
                line = line[:-1]
            
            if "matrixSize" in line:
                continue
                
            if "nicht shared" in line or "Benchmark" in line:
                continue
                
            if line.startswith("-1"):
                is_naive_section = True
                continue
            
            
            elemente = line.split(',')
            if len(elemente) >= 6:
                row = [float(p) for p in elemente[:7]]
                
                if is_naive_section:
                    naive_data.append(row)
                else:
                    shared_data.append(row)
            
    # DataFrames erstellen
    df_shared = pd.DataFrame(shared_data, columns=columns)
    df_naive = pd.DataFrame(naive_data, columns=columns)
    
    return df_shared, df_naive


df_shared, df_naive = parse_benchmark_file(FILENAME)

if df_shared is not None and not df_shared.empty:
    
    # 2. Plot erstellen
    plt.figure(figsize=(12, 7))

    # Shared Memory Kurve
    plt.plot(df_shared['blockDim'], df_shared['Time_Kernel_ms'], 
             marker='o', label='Shared Memory (Optimized)', color='blue')

    plt.plot(df_shared['blockDim'], df_shared['Time_H2D_ms'], 
             marker='o', label='Time_H2D_ms', color='green')
    plt.plot(df_shared['blockDim'], df_shared['Bandwidth_H2D'], 
             marker='o', label='Bandwidth_H2D', color='green')
    plt.plot(df_shared['blockDim'], df_shared['Time_D2H_ms'], 
             marker='o', label='Time_D2H_ms', color='green')
    # Naive Kurve
    if not df_naive.empty:
        plt.plot(df_naive['blockDim'], df_naive['Time_Kernel_ms'], 
                 marker='x', linestyle='--', label='Naive (Global Memory)', color='red')

    # Beschriftung
    plt.title('GPU Kernel Laufzeit: Shared Memory vs. Naive\n(Daten aus ' + FILENAME + ')', fontsize=14)
    plt.xlabel('Block Dimension (Threads per Block-Side)', fontsize=12)
    plt.ylabel('Time Kernel (ms)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # X-Achse sauber beschriften (Alle Blockgrößen anzeigen)
    plt.xticks(range(1, 33)) # 1 bis 32
    
    # Optional: Logarithmische Skala, falls BlockSize 1 den Graphen verzerrt
    # plt.yscale('log') 

    plt.tight_layout()
    
    # Plot anzeigen
    plt.show()

    # 3. Analyse Ausgabe in der Konsole
    print(f"\n--- Analyse für {FILENAME} ---")
    
    # Bestes Ergebnis Shared
    min_idx_sh = df_shared['Time_Kernel_ms'].idxmin()
    best_sh = df_shared.iloc[min_idx_sh]
    print(f"Shared Memory Optimum: {best_sh['Time_Kernel_ms']} ms bei BlockDim {int(best_sh['blockDim'])}")
    
    # Bestes Ergebnis Naive
    if not df_naive.empty:
        min_idx_na = df_naive['Time_Kernel_ms'].idxmin()
        best_na = df_naive.iloc[min_idx_na]
        print(f"Naive Memory Optimum:  {best_na['Time_Kernel_ms']} ms bei BlockDim {int(best_na['blockDim'])}")
        
        # Speedup berechnen
        speedup = best_na['Time_Kernel_ms'] / best_sh['Time_Kernel_ms']
        print(f"Speedup (Kernel only): {speedup:.2f}x")

else:
    print("Keine Daten konnten gelesen werden.")