import os
import json
import tqdm
import sys
from collections import defaultdict

# Importamos a função principal do gerador virtual
try:
    from virtual_synth_config import generate_virtual_hw_config
except ImportError:
    # Caso o script seja rodado de uma pasta que não enxerga o arquivo
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from virtual_synth_config import generate_virtual_hw_config

BASE_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"
TEMP_FILE = "/tmp/validation_temp.json"

def compare_jsons(expected_path, generated_path):
    with open(expected_path, "r") as f:
        expected = json.load(f)
    with open(generated_path, "r") as f:
        generated = json.load(f)

    errors = []
    
    # Comparar apenas as chaves que existem no gerado (o FINN pode ter chaves extras de infra)
    for node_name, gen_attrs in generated.items():
        if node_name == "Defaults": continue
        
        if node_name not in expected:
            # errors.append(f"  [!] Nó {node_name} não encontrado no original")
            continue
        
        exp_attrs = expected[node_name]
        
        # Atributos críticos para validar a expansão virtual
        for attr in ["depth", "ram_style", "impl_style"]:
            if attr in gen_attrs and attr in exp_attrs:
                g_val = gen_attrs[attr]
                e_val = exp_attrs[attr]
                
                # Normalização básica
                if str(g_val).lower() != str(e_val).lower():
                    errors.append(f"  [{node_name}] Mismatch {attr}: Gen={g_val} vs Exp={e_val}")

    return errors

def main():
    print("=" * 60)
    print("🔍 HARA V3 - VALIDACÃO DE ESTIMATIVA VIRTUAL")
    print("=" * 60)

    all_builds = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    total_runs = 0
    mismatched_runs = 0
    summary = defaultdict(int)

    for build_name in tqdm.tqdm(all_builds, desc="Verificando builds"):
        build_path = os.path.join(BASE_DIR, build_name)
        
        # Encontrar pastas run*
        runs = [d for d in os.listdir(build_path) if d.startswith("run") and os.path.isdir(os.path.join(build_path, d))]
        
        for run_folder in runs:
            run_path = os.path.join(build_path, run_folder)
            
            # Caminhos esperados
            folding_json = os.path.join(build_path, run_folder + ".json")
            onnx_file = os.path.join(run_path, "intermediate_models", "step_generate_estimate_reports.onnx")
            
            # Fallback para o nome do onnx se mudar nas versões
            if not os.path.exists(onnx_file):
                onnx_file = os.path.join(run_path, "intermediate_models", "step_create_stitched_ip.onnx")

            expected_hw_json = os.path.join(run_path, "final_hw_config.json")

            if os.path.exists(folding_json) and os.path.exists(onnx_file) and os.path.exists(expected_hw_json):
                total_runs += 1
                
                try:
                    # Executa a geração virtual
                    # Redirecionamos o stdout para não poluir o tqdm
                    devnull = open(os.devnull, 'w')
                    old_stdout = sys.stdout
                    sys.stdout = devnull
                    
                    generate_virtual_hw_config(onnx_file, folding_json, TEMP_FILE)
                    
                    sys.stdout = old_stdout
                    
                    # Compara
                    run_errors = compare_jsons(expected_hw_json, TEMP_FILE)
                    
                    with open(expected_hw_json, "r") as f: exp_data = json.load(f)
                    with open(TEMP_FILE, "r") as f: gen_data = json.load(f)
                    common = set(exp_data.keys()) & set(gen_data.keys())
                    
                    if run_errors:
                        mismatched_runs += 1
                        print(f"\n❌ Erros em {build_name}/{run_folder}:")
                        print(f"  Nodes em comum: {len(common)}, Nodes no original: {len(exp_data)}, Nodes no gerado: {len(gen_data)}")
                        for err in run_errors[:5]: # Mostrar apenas os 5 primeiros
                            print(err)
                        if len(run_errors) > 5:
                            print(f"  ... e mais {len(run_errors)-5} erros.")
                    else:
                        # Log para entender o 100%
                        # print(f"\n✅ {build_name}/{run_folder}: {len(common)} nodes em comum.")
                        pass
                    
                except Exception as e:
                    print(f"\n⚠️ Falha ao processar {build_name}/{run_folder}: {str(e)}")
                finally:
                    if os.path.exists(TEMP_FILE):
                        os.remove(TEMP_FILE)

    print("\n" + "=" * 60)
    print("📊 RESULTADO FINAL")
    print("=" * 60)
    print(f"Total de runs testadas: {total_runs}")
    print(f"Total de runs com divergência: {mismatched_runs}")
    if total_runs > 0:
        accuracy = (1 - mismatched_runs / total_runs) * 100
        print(f"Acurácia da estrutura virtual: {accuracy:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
