from hw_utils import utils  # Importa a função de extração de área
import pandas as pd
import os
import json

# --------------------------------------------
# Parte 4: Coleta de áreas e throughput dos builds
# --------------------------------------------

# Lista dos diretórios de build que você quer analisar
build_dirs = [
    "/home/arthurely/Desktop/finn/notebooks/sat6_cnn/builds_HARA_proof_not_two_pass/t2w4_44000fps_u", 
    # adicione mais caminhos conforme necessário
]

# Lista para armazenar os dados extraídos
area_data = []

# Loop para extrair informações de cada build_dir
for build_dir in build_dirs:
    if os.path.exists(build_dir):
        try:
            # Extrai área
            area_info = utils.extract_area_from_rpt(build_dir)
            area_info['build_dir'] = build_dir  # Adiciona o caminho do build

            # Inicializa o campo de throughput com None
            area_info['estimated_throughput_fps'] = None

            # Tenta carregar o throughput do arquivo rtlsim_performance.json
            rtlsim_path = os.path.join(build_dir, "report/rtlsim_performance.json")
            if os.path.exists(rtlsim_path):
                with open(rtlsim_path, "r") as f:
                    rtlsim_data = json.load(f)
                throughput_fps = rtlsim_data.get("throughput[images/s]", None)
                area_info['estimated_throughput_fps'] = throughput_fps

            area_data.append(area_info)
        except Exception as e:
            print(f"Failed to process {build_dir}: {e}")
    else:
        print(f"Build directory not found: {build_dir}")

# Transformar a lista de áreas em um DataFrame
if area_data:
    area_df = pd.DataFrame(area_data)
    print("Área e throughput dos builds carregados:")
    print(area_df)
else:
    print("Nenhuma área foi extraída.")

# (Opcional) Salvar em CSV para inspeção posterior
output_area_csv = "/home/arthurely/Desktop/finn/hara/data/area_from_tf_builds.csv"
area_df.to_csv(output_area_csv, index=False)
print(f"Área e throughput dos builds salva em: {output_area_csv}")
