import numpy as np
import pandas as pd
import json
import os  

# Defina corretamente as topologias e FPS alvo
topologies = [
    {'id': 1, 'quant': [2, 4, 8]},
    {'id': 2, 'quant': [2, 4, 8]}
]
target_fps_list = [500, 5000, 50000]

headers = ['Hardware config', 'Total LUT', 'Total LUTRAM', 'Total FFs', 'BRAM', 'DSP Blocks']
PYNQ_Z1_max = ['', 53200, 17400, 106400, 140, 220]
 
results_vivado = [headers]
pre_data = [headers]

for tp in topologies:
    for quant in tp['quant']:
        for target_fps in target_fps_list:
            hardware_config = f"t{tp['id']}w{quant}_{target_fps}fps"
            
            filename = f"../notebooks/sat6_cnn/builds_pynq/{hardware_config}_u/zynq_proj/finn_zynq_link.runs/impl_1/top_wrapper_utilization_placed.rpt"
            
            if not os.path.exists(filename):
                continue

            with open(filename) as my_file:
                content = my_file.readlines() 
                data = [hardware_config]
                utilization = {}
                for line in content:
                    if '|' not in line:
                        continue
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    
                    if len(parts) < 6:
                        continue
                    
                    site_type = parts[0]
                    used = parts[1].replace(',', '')
                    util_percent = parts[-1].replace('%', '')
                    
                    if site_type in ["Slice LUTs", "LUT as Memory", "Slice Registers", "Block RAM Tile", "DSPs"]:
                        utilization[site_type] = (used, util_percent)

                if utilization:
                    data.extend([
                        int(utilization.get("Slice LUTs", (0, 0))[0]),
                        int(utilization.get("LUT as Memory", (0, 0))[0]),
                        int(utilization.get("Slice Registers", (0, 0))[0]),
                        float(utilization.get("Block RAM Tile", (0, 0))[0]),
                        int(utilization.get("DSPs", (0, 0))[0]),
                    ])
                
                results_vivado.append(data)

index = 1
for data in results_vivado[1:]:
    data[1] = f"{data[1]} ({int(data[1])*100/PYNQ_Z1_max[1]:.2f}%)"
    data[2] = f"{data[2]} ({int(data[2])*100/PYNQ_Z1_max[2]:.2f}%)"
    data[3] = f"{data[3]} ({int(data[3])*100/PYNQ_Z1_max[3]:.2f}%)"
    data[4] = f"{data[4]} ({float(data[4])*100/PYNQ_Z1_max[4]:.2f}%)"
    data[5] = f"{data[5]} ({int(data[5])*100/PYNQ_Z1_max[5]:.2f}%)"
    results_vivado[index] = data
    index += 1

df_vivado = pd.DataFrame(results_vivado)
print("(Vivado) Hardware area results")
print(df_vivado.to_string(header=None, index=False))

df_vivado.to_csv('./results/vivado_area_results_modified.csv')
