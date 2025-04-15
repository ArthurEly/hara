import os
import csv
import onnx
import re
from onnx import helper
from collections import defaultdict
import xml.etree.ElementTree as ET
from datetime import datetime

# =============================
# Configurações
# =============================

ignore_attrs = [
    "backend", "code_gen_dir_ipgen", "ipgen_path", "ip_path", "gen_top_module", "partition_id", "ip_vlnv",
    "cycles_estimate", "slr", "depth_monitor", "depthwise", "is1D",
]

base_dirs = [
    "/home/arthurely/Desktop/finn/notebooks/sat6_cnn/builds_pynq/",
    "/home/arthurely/Desktop/finn/notebooks/CIFAR10/builds/",
]

topologies = [
    {'id': 1, 'quant': [2, 4, 8]}, {'id': 2, 'quant': [2, 4, 8]}
]
target_fps_list = [500, 5000, 50000]

repos = [
    f"t{t['id']}w{quant}_{fps}fps_u"
    for t in topologies
    for quant in t['quant']
    for fps in target_fps_list
]

area_components = {
    "LUTs": ["Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs"],
    "FFs": ["FFs"],
    "RAMs": ["RAMB36", "RAMB18"],
    "DSPs": ["DSP Blocks"],
}

# =============================
# Diretórios de saída
# =============================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
onnx_results_dir = f"../data/results_onnx/run_{timestamp}/"
last_run_dir = "../data/results_onnx/last_run/"

os.makedirs(onnx_results_dir, exist_ok=True)
os.makedirs(last_run_dir, exist_ok=True)

area_csv_paths = [
    os.path.join(onnx_results_dir, "area_summary.csv"),
    os.path.join(last_run_dir, "area_summary.csv"),
]

# =============================
# Extração dos dados
# =============================

optype_rows = defaultdict(list)
optype_all_keys = defaultdict(set)
area_summary_rows = []

for base_dir in base_dirs:
    print(f"\n📂 Processando base_dir: {base_dir}")
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for repo in sorted(subdirs):
        if repo.endswith("sources_u"):
            continue

        onnx_dir = os.path.join(base_dir, repo, "intermediate_models/")
        if not os.path.isdir(onnx_dir):
            print(f"🔍 Diretório não encontrado: {onnx_dir}")
            continue

        filename = "step_generate_estimate_reports.onnx"
        # --- ONNX ---
        onnx_path = os.path.join(onnx_dir, filename)
        try:
            model = onnx.load(onnx_path)
        except Exception as e:
            print(f"❌ Erro ao carregar {onnx_path}: {e}")
            continue

        for node in model.graph.node:
            op_type = node.op_type
            if op_type == "IODMA_hls":
                continue

            row = {
                "Repo": repo,
                "NodeName": node.name or "(sem nome)",
                "OpType": op_type,
            }

            for attr in node.attribute:
                attr_name = attr.name
                if attr_name in ignore_attrs:
                    continue
                attr_val = helper.get_attribute_value(attr)
                row[attr_name] = str(attr_val)
                optype_all_keys[op_type].add(attr_name)

            optype_rows[op_type].append(row)

        # --- Área (XML) ---
        xml_path = os.path.join(base_dir, repo, "zynq_proj/synth_report.xml")
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for row in root.findall(".//tablerow"):
                values = [col.get("contents") for col in row.findall(".//tablecell") if col.get("contents")]
                if values and "Streaming" in values[0] and "IODMA" not in values[0]:
                    instance = values[0].replace("StreamingDataflowPartition_1_","").strip()
                    area_data = {
                        "Repo": repo,
                        "Instance": instance,
                        "Total LUTs": values[2],
                        "Logic LUTs": values[3],
                        "LUTRAMs": values[4],
                        "SRLs": values[5],
                        "FFs": values[6],
                        "RAMB36": values[7],
                        "RAMB18": values[8],
                        "DSP Blocks": values[9]
                    }
                    area_summary_rows.append(area_data)

# =============================
# Escrita dos CSVs ONNX (sem merge)
# =============================
for op_type, rows in optype_rows.items():
    attr_names = sorted(optype_all_keys[op_type])
    print(attr_names)
    input()
    filenames = [f"{op_type}.csv"]
    csv_paths = [os.path.join(p, f"{op_type}.csv") for p in [onnx_results_dir, last_run_dir]]

    for csv_path in csv_paths:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["Repo", "NodeName", "OpType"] + attr_names
            writer.writerow(header)
            for row in rows:
                writer.writerow([row.get(k, "") for k in header])
        print(f"✅ CSV gerado para OpType '{op_type}': {csv_path}")

# =============================
# Escrita dos CSVs de área
# =============================
area_headers = ["Repo", "Instance"] + sum(area_components.values(), [])
for path in area_csv_paths:
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=area_headers)
        writer.writeheader()
        for row in area_summary_rows:
            writer.writerow(row)
    print(f"✅ CSV de área gerado: {path}")

input()

# =============================
# Merge ONNX + Área
# =============================
area_lookup = {
    (row["Repo"], row["Instance"]): [row.get(h, "") for h in area_headers[2:]]
    for row in area_summary_rows
}
area_fields = sum(area_components.values(), [])

for op_type, rows in optype_rows.items():
    attr_names = sorted(optype_all_keys[op_type])
    filenames = [f"{op_type}_merged.csv"]
    csv_paths = [os.path.join(p, f) for p in [onnx_results_dir, last_run_dir] for f in filenames]

    for csv_path in csv_paths:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["Repo", "NodeName", "OpType"] + attr_names + area_fields
            writer.writerow(header)

            for row in rows:
                key = (row["Repo"], row["NodeName"])
                if key in area_lookup:
                    line = [
                        row["Repo"], row["NodeName"], op_type,
                    ] + [row.get(k, "") for k in attr_names] + area_lookup[key]
                    writer.writerow(line)
        print(f"📁 CSV gerado com merge: {csv_path}")

# =============================
# Limpeza de CSVs temporários
# =============================
def limpar_csvs(pasta):
    for fname in os.listdir(pasta):
        if fname.endswith(".csv") and not fname.endswith("_merged.csv") and fname != "area_summary.csv":
            path = os.path.join(pasta, fname)
            os.remove(path)
            print(f"🧹 Removido: {path}")

limpar_csvs(onnx_results_dir)
limpar_csvs(last_run_dir)
