import xml.etree.ElementTree as ET
import csv
import os
import json

# Diretório base
base_dir = "/home/arthurely/Desktop/finn/notebooks/sat6_cnn/builds_pynq/"

# Topologias e FPS alvo
topologies = [
    {'id': 1, 'quant': [2, 4, 8]},
]
target_fps_list = [500, 5000, 50000]

# Geração dinâmica dos arquivos XML e JSON
xml_files = [
    (top, f"{top}/zynq_proj/synth_report.xml")
    for t in topologies
    for quant in t['quant']
    for fps in target_fps_list
    for top in [f"t{t['id']}w{quant}_{fps}fps_u"]
]

# Diretórios de saída
results_dir = "./results/"
instances_dir = "./instance_reports/"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(instances_dir, exist_ok=True)

# Definição dos arquivos CSV por área
csv_files = {
    "LUTs": os.path.join(results_dir, "synth_report_LUTs.csv"),
    "FFs": os.path.join(results_dir, "synth_report_FFs.csv"),
    "RAMs": os.path.join(results_dir, "synth_report_RAMs.csv"),
    "DSPs": os.path.join(results_dir, "synth_report_DSPs.csv"),
}

# Definição das colunas comuns
json_fields = ["PE", "SIMD", "ram_style", "resType", "mem_mode", "runtime_writeable_weights", 
               "inFIFODepths", "outFIFODepths", "impl_style", "depth", "parallel_window"]

# Componentes individuais
area_components = {
    "LUTs": ["Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs"],
    "FFs": ["FFs"],
    "RAMs": ["RAMB36", "RAMB18"],
    "DSPs": ["DSP Blocks"],
}

# Criando arquivos CSV gerais
csv_writers = {}
csv_files_handles = {}

for key, file_path in csv_files.items():
    csv_files_handles[key] = open(file_path, mode="w", newline="")
    csv_writers[key] = csv.writer(csv_files_handles[key])
    csv_writers[key].writerow(["Repository", "Instance"] + json_fields + area_components[key])  # Escreve cabeçalhos

# Dicionário para armazenar dados de instâncias
instance_data = {}

# Processamento dos arquivos XML
for repo_name, xml_file in xml_files:
    full_path = os.path.join(base_dir, xml_file)

    if os.path.exists(full_path):
        tree = ET.parse(full_path)
        root = tree.getroot()

        for row in root.findall(".//tablerow"):
            values = [col.get("contents") for col in row.findall(".//tablecell") if col.get("contents")]

            if values and "Streaming" in values[0] and "IODMA" not in values[0]:
                instance = values[0].replace("StreamingDataflowPartition_1_", "").strip()
                json_path = os.path.join(base_dir, repo_name, "final_hw_config.json")

                # Inicializar campos do JSON
                json_data_values = {key: "" for key in json_fields}  # Garante que todos os campos existam

                # Ler JSON se existir
                if os.path.exists(json_path):
                    with open(json_path, "r") as json_file:
                        json_data = json.load(json_file)
                        if instance in json_data:
                            instance_data_json = json_data[instance]
                            for key in json_fields:
                                if key in instance_data_json:
                                    json_data_values[key] = instance_data_json[key]
                            json_data_values["inFIFODepths"] = ",".join(map(str, instance_data_json.get("inFIFODepths", [])))
                            json_data_values["outFIFODepths"] = ",".join(map(str, instance_data_json.get("outFIFODepths", [])))

                # Criar dicionário de componentes de área
                area_values = {
                    "LUTs": values[2:6],  # Total LUTs, Logic LUTs, LUTRAMs, SRLs
                    "FFs": [values[6]],   # FFs
                    "RAMs": values[7:9],  # RAMB36, RAMB18
                    "DSPs": [values[9]],  # DSP Blocks
                }

                # Adicionar ao dicionário de instâncias
                if instance not in instance_data:
                    instance_data[instance] = []
                instance_data[instance].append([repo_name] + [json_data_values[field] for field in json_fields] + sum(area_values.values(), []))

                # Escrever nos CSVs de área
                for key in area_components.keys():
                    row_data = [repo_name, instance] + [json_data_values[field] for field in json_fields] + area_values[key]
                    csv_writers[key].writerow(row_data)

# Criar arquivos CSV separados por instância
for instance, data_rows in instance_data.items():
    instance_file_path = os.path.join(instances_dir, f"{instance}.csv")
    with open(instance_file_path, mode="w", newline="") as instance_file:
        writer = csv.writer(instance_file)
        writer.writerow(["Repository"] + json_fields + sum(area_components.values(), []))  # Cabeçalhos
        writer.writerows(data_rows)

# Fechar arquivos CSV gerais
for file in csv_files_handles.values():
    file.close()

print("Arquivos CSV gerados com sucesso:")
for key, file_path in csv_files.items():
    print(f"- {file_path}")

print("Arquivos CSV por instância gerados em:", instances_dir)
