import onnx

# --- INÍCIO DO CÓDIGO DE DIAGNÓSTICO ---

def print_all_nodes(onnx_path):
    """
    Carrega um modelo ONNX e imprime o nome e o tipo de operação de cada nó.
    """
    try:
        model = onnx.load(onnx_path)
        print("\n--- Lista de Todos os Nós no Modelo ONNX ---")
        print(f"Modelo: {onnx_path}\n")
        
        if not model.graph.node:
            print("Nenhum nó encontrado no grafo do modelo.")
            return

        for i, node in enumerate(model.graph.node):
            print(f"Nó #{i+1}:")
            print(f"  - Nome do Nó: {node.name}")
            print(f"  - Tipo de Operação (op_type): \033[93m{node.op_type}\033[0m") # Em amarelo para destaque
        
        print("\n-------------------------------------------")

    except FileNotFoundError:
        print(f"ERRO: O arquivo de modelo não foi encontrado em '{onnx_path}'")
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo ONNX: {e}")

# Substitua pelo caminho do seu modelo
ONNX_MODEL_PATH = "/home/arthurely/Desktop/finn_chi2p/hara/hw/builds/2025-09-13_01-48-43_SAT6_T2_cb4027d5/SAT6_T2w4_run0_final_balanced/intermediate_models/step_generate_estimate_reports.onnx"

# Chama a função para imprimir os nós
print_all_nodes(ONNX_MODEL_PATH)

# --- FIM DO CÓDIGO DE DIAGNÓSTICO ---