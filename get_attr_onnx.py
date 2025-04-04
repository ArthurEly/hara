import os
import csv
import onnx
from onnx import helper
from collections import defaultdict
import xml.etree.ElementTree as ET

# =============================
# Configurações
# =============================

ignore_attrs = [
    "backend", "code_gen_dir_ipgen", "ipgen_path", "ip_path", "gen_top_module", "partition_id", "ip_vlnv",
    "cycles_estimate", "slr", "depth_monitor", "depthwise", "is1D",
]

base_dir = "/home/arthurely/Desktop/finn/notebooks/sat6_cnn/builds_pynq/"
onnx_results_dir = "./results_onnx/"
area_csv_path = os.path.join(onnx_results_dir, "area_summary.csv")
os.makedirs(onnx_results_dir, exist_ok=True)

topologies = [{'id': 1, 'quant': [2, 4, 8]},{'id': 2, 'quant': [2, 4, 8]}]
target_fps_list = [500, 5000, 50000]

repos = [
    f"t{t['id']}w{quant}_{fps}fps_u"
    for t in topologies
    for quant in t['quant']
    for fps in target_fps_list
]

# =============================
# Inicialização
# =============================

optype_rows = defaultdict(list)
optype_all_keys = defaultdict(set)
area_summary_rows = []

area_components = {
    "LUTs": ["Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs"],
    "FFs": ["FFs"],
    "RAMs": ["RAMB36", "RAMB18"],
    "DSPs": ["DSP Blocks"],
}

# =============================
# Processamento dos repositórios
# =============================

for repo in repos:
    onnx_dir = os.path.join(base_dir, repo, "intermediate_models/kernel_partitions/")
    if not os.path.isdir(onnx_dir):
        print(f"🔍 Diretório não encontrado: {onnx_dir}")
        continue

    # --- ONNX ---
    for filename in sorted(os.listdir(onnx_dir)):
        if filename.endswith(".onnx"):
            onnx_path = os.path.join(onnx_dir, filename)
            try:
                model = onnx.load(onnx_path)
            except Exception as e:
                print(f"❌ Erro ao carregar {onnx_path}: {e}")
                continue

            partition_name = filename.replace(".onnx", "")
            for node in model.graph.node:
                op_type = node.op_type
                if op_type == "IODMA_hls":
                    continue

                row = {
                    "Repo": repo,
                    "NodeName": node.name or "(sem nome)",
                    "OpType": op_type,
                    "Partition": partition_name
                }

                for attr in node.attribute:
                    attr_name = attr.name
                    if attr_name in ignore_attrs:
                        continue
                    attr_val = helper.get_attribute_value(attr)
                    row[attr_name] = str(attr_val)
                    optype_all_keys[op_type].add(attr_name)

                optype_rows[op_type].append(row)

    # --- XML de Área ---
    full_path = os.path.join(base_dir, repo, "zynq_proj/synth_report.xml")
    if os.path.exists(full_path):
        tree = ET.parse(full_path)
        root = tree.getroot()

        for row in root.findall(".//tablerow"):
            values = [col.get("contents") for col in row.findall(".//tablecell") if col.get("contents")]

            if values and "Streaming" in values[0] and "IODMA" not in values[0]:
                instance = values[0].strip()
                #instance = values[0].replace("StreamingDataflowPartition_1_", "").strip()

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
# Escrita dos arquivos CSV
# =============================

# --- CSVs ONNX ---
for op_type, rows in optype_rows.items():
    attr_names = sorted(optype_all_keys[op_type])
    csv_path = os.path.join(onnx_results_dir, f"{op_type}.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["Repo", "NodeName", "OpType", "Partition"] + attr_names
        writer.writerow(header)

        for row in rows:
            line = [row.get(k, "") for k in header]
            writer.writerow(line)

    print(f"✅ CSV gerado para OpType '{op_type}': {csv_path}")

# --- CSV de Área ---
area_headers = ["Repo", "Instance"] + sum(area_components.values(), [])
with open(area_csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=area_headers)
    writer.writeheader()
    for row in area_summary_rows:
        writer.writerow(row)

print(f"✅ CSV de área gerado: {area_csv_path}")
print("🏁 Processamento finalizado com sucesso.")

# =============================
# Merge por OpType com dados de Área
# =============================

# Prepara os dados de área para lookup por (Repo, Instance)
area_lookup = {
    (row["Repo"], row["Instance"]): [
        row.get(h, "") for h in area_headers[2:]
    ]
    for row in area_summary_rows
}


# Cabeçalhos por componente de área (na mesma ordem usada antes)
area_headers = sum(area_components.values(), [])

# Para cada OpType, salva o CSV com atributos + dados de área
for op_type, rows in optype_rows.items():
    attr_names = sorted(optype_all_keys[op_type])
    csv_path = os.path.join(onnx_results_dir, f"{op_type}_merged.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["Repo", "NodeName", "OpType", "Partition"] + attr_names + area_headers
        writer.writerow(header)

        for row in rows:
            repo = row["Repo"]
            node_name = row["NodeName"]
            key = (repo, node_name)

            area_values = area_lookup.get(key, [""] * len(area_headers))
            line = [repo, node_name, op_type, row["Partition"]] + [row.get(k, "") for k in attr_names] + area_values
            writer.writerow(line)

    print(f"📁 CSV gerado com merge: {csv_path}")
print("🏁 Merge finalizado com sucesso.")

# =============================
# Limpeza: Remove CSVs não merged
# =============================

for fname in os.listdir(onnx_results_dir):
    if fname.endswith(".csv") and not fname.endswith("_merged.csv") and fname != "area_summary.csv":
        path = os.path.join(onnx_results_dir, fname)
        os.remove(path)
        print(f"🧹 Removido: {path}")

print("🧼 Diretório limpo! Só os arquivos *_merged.csv foram mantidos.")