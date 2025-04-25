import os
import sys

# obtém o diretório onde o script está
current_dir = os.path.dirname(os.path.abspath(__file__))
# sobe um nível (pai)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# insere no início do sys.path
sys.path.insert(0, parent_dir)

# agora sim você pode importar
from hw_utils import utils
import json

def test_modify_folding():
    starting_build_dir = f"/home/arthurely/Desktop/finn/hara/builds/run_2025-04-23_18-21-42/t2w8_run1"
    starting_json = f"{starting_build_dir}/final_hw_config.json"
    onnx_path = f"{starting_build_dir}/intermediate_models/step_generate_estimate_reports.onnx"
    with open(starting_json, 'r') as f:
        folding_input = json.load(f)
    #print(f"Mudando aqui 1: {last_build_dir}/report/estimate_layer_cycles.json")
    estimate_layer_cycles_path = f"{starting_build_dir}/report/estimate_layer_cycles.json"
    with open(estimate_layer_cycles_path, 'r') as f:
            estimate_layer_cycles = json.load(f)    

    print("=== Antes ===")
    print(json.dumps(folding_input, indent=4))
    print("\nExecutando modify_folding...\n")
    new_fold = utils.modify_folding(folding_input, onnx_path, estimate_layer_cycles)
    print("\n=== Depois ===")
    print(json.dumps(new_fold, indent=4))
    print(utils.dict_diff(fol))

if __name__ == '__main__':
    test_modify_folding()
