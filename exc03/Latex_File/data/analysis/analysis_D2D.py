import pandas as pd
import matplotlib.pyplot as plt
import io
import os

# single marking important for later
file_configs = [
    # H2D
    ("ex3_out_H2D.txt", "H2D", "dual"),
    # D2H
    ("ex3_out_D2H.txt", "D2H", "dual"),
    # we have only one data for D2D
    ("ex3_out_D2D.txt", "D2D", "single") 
]


plt.figure(figsize=(8, 5))

for config in file_configs:
    input_filename = config[0]
    copy_type = config[1]
    mode = config[2]
    
    input_path = os.path.join("..", input_filename) 
    
    print(f"--- working with file: {input_filename} ({copy_type}, Mode: {mode}) ---")

    
    try:
        with open(input_path, 'r') as f:
            data_string = f.read()
    except FileNotFoundError:
        print(f"ERROR: File '{input_path}' not found. Skip file.")
        continue

    
    if mode == "dual":
        # when its dual, do both, else print only one line
        
        # Der Trenner ist der spezifische Typ (z.B. d2h) gefolgt von 'using '
        data_blocks = data_string.strip().split(f'Measuring data transfer {copy_type.lower()} using ')

        if len(data_blocks) < 3:
            print(f"ERROR: Not Pinnable and pagable in dual block found: {input_filename}")
            continue
        
        # Pageable Memory Data
        pageable_data = data_blocks[1].split('memory...')[1].strip()
        df_pageable = pd.read_csv(io.StringIO(pageable_data), sep='\s+', header=None, 
                                  names=['Bytes', 'Time_s', 'Bandwidth_B_s'])

        # Pinned Memory Data
        pinned_data = data_blocks[2].split('memory...')[1].strip()
        df_pinned = pd.read_csv(io.StringIO(pinned_data), sep='\s+', header=None,
                                names=['Bytes', 'Time_s', 'Bandwidth_B_s'])

        # Bandwidth transfer
        df_pageable['Bandwidth_GB_s'] = df_pageable['Bandwidth_B_s'] / 1e9
        df_pinned['Bandwidth_GB_s'] = df_pinned['Bandwidth_B_s'] / 1e9

        # plot dual data
        plt.plot(df_pageable['Bytes'], df_pageable['Bandwidth_GB_s'], 
                 marker='.', linestyle='-', 
                 label=f'{copy_type}: Pageable')
        
        plt.plot(df_pinned['Bytes'], df_pinned['Bandwidth_GB_s'], 
                 marker='x', linestyle='--', 
                 label=f'{copy_type}: Pinned')

    elif mode == "single":
        # cutting the String off
        data_block = data_string.strip().split('Measuring data transfer d2d using ')[-1].split('\n', 1)[-1]
        
        # load the actual data
        df_single = pd.read_csv(io.StringIO(data_block), sep='\s+', header=None, 
                                names=['Bytes', 'Time_s', 'Bandwidth_B_s'])
        
        # well, we have seen that already in dual
        df_single['Bandwidth_GB_s'] = df_single['Bandwidth_B_s'] / 1e9

        # plot data
        plt.plot(df_single['Bytes'], df_single['Bandwidth_GB_s'], 
                 marker='s', linestyle='-', color='purple',
                 label=f'{copy_type}: Device-to-Device')
        

#formating the axis and diagramm
plt.xscale('log')
plt.title('CUDA Memory Copy Bandwidth: H2D, D2H, and D2D Comparison')
plt.xlabel('Dataamount (Bytes, Log-Skala)')
plt.ylabel('Bandbreite (GB/s)')
plt.legend(title="Copitype")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

plt.show()