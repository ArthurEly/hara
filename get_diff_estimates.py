import os
import json
import matplotlib.pyplot as plt
import numpy as np
from hw_utils import utils
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

base_path = "/home/arthurely/Desktop/finn/notebooks/sat6_cnn/builds_pynq"

# Lista apenas os diretórios que começam com 't' e NÃO contêm '_sources'
all_dirs = [
    os.path.join(base_path, d) for d in os.listdir(base_path)
    if d.startswith("t") and "_sources" not in d and os.path.isdir(os.path.join(base_path, d))
]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def sum_estimate_layer_resources_hls(data):
    result = {"DSP Blocks": 0, "Total LUTs": 0, "FFs": 0, "BRAM (36k)": 0}
    for layer, res in data.items():
        result["DSP Blocks"] += int(res.get("DSP", 0))
        result["Total LUTs"] += int(res.get("LUT", 0))
        result["FFs"] += int(res.get("FF", 0))
        result["BRAM (36k)"] += int(res.get("BRAM_18K", 0)) // 2
    return result

def sum_estimate_layer_resources(data):
    result = {"DSP Blocks": 0, "Total LUTs": 0, "FFs": 0, "BRAM (36k)": 0}
    for layer, res in data.items():
        if layer == "total":
            result["DSP Blocks"] += int(res.get("DSP", 0))
            result["Total LUTs"] += int(res.get("LUT", 0))
            result["BRAM (36k)"] += int(res.get("BRAM_18K", 0)) // 2
    return result

# Dados acumulados
build_labels = []
total_real_area = []
diffs_est = {"DSP Blocks": [], "Total LUTs": [], "FFs": [], "BRAM (36k)": []}
diffs_hls = {"DSP Blocks": [], "Total LUTs": [], "FFs": [], "BRAM (36k)": []}

# Coleta dados de todas as builds
for build_dir in sorted(all_dirs):
    build_name = os.path.basename(build_dir)
    report_path = os.path.join(build_dir, "report")
    try:
        # Área real
        area_data_full = utils.extract_area_from_rpt(build_dir)
        area_real = {
            k: area_data_full[k]
            for k in ["DSP Blocks", "Total LUTs", "FFs", "BRAM (36k)"]
            if k in area_data_full
        }

        # Soma para ordenação
        total_real = sum(area_real.values())
        total_real_area.append(total_real)
        build_labels.append(build_name)

        # Estimativas
        est_hls = sum_estimate_layer_resources_hls(
            load_json(os.path.join(report_path, "estimate_layer_resources_hls.json"))
        )
        est_total = sum_estimate_layer_resources(
            load_json(os.path.join(report_path, "estimate_layer_resources.json"))
        )

        # Diferenças reais (positivas ou negativas)
        for k in diffs_est.keys():
            diffs_est[k].append(area_real.get(k, 0) - est_total.get(k, 0))
            diffs_hls[k].append(area_real.get(k, 0) - est_hls.get(k, 0))

    except Exception as e:
        print(f"Erro ao processar {build_dir}: {e}")

# Ordena os labels usando ordenação natural
sorted_pairs = sorted(zip(build_labels, total_real_area), key=lambda x: natural_sort_key(x[0]))
sorted_labels, sorted_total_area = zip(*sorted_pairs)
sorted_indices = [build_labels.index(label) for label in sorted_labels]


# Reorganiza as listas conforme ordenação
for k in diffs_est:
    diffs_est[k] = [diffs_est[k][i] for i in sorted_indices]
    diffs_hls[k] = [diffs_hls[k][i] for i in sorted_indices]

resource_types = ["DSP Blocks", "Total LUTs", "FFs", "BRAM (36k)"]
colors = {
    "DSP Blocks": "#FF5733",
    "Total LUTs": "#33C3FF",
    "FFs": "#77DD77",
    "BRAM (36k)": "#FFB347",
}

x = np.arange(len(sorted_labels))
width = 0.35

# Diretório de saída (opcional)
os.makedirs("plots", exist_ok=True)

for res in resource_types:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - width/2, diffs_est[res], width, label="Estimativa FINN", color=colors[res])
    ax.bar(x + width/2, diffs_hls[res], width, label="Estimativa HLS", color="gray")

    ax.set_ylabel("Diferença Absoluta")
    ax.set_title(f"Erro de Estimativa - {res}")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plots/diff_{res.replace(' ', '_')}.pdf")
    plt.show()

