# main.py
import argparse
import os
import subprocess
from datetime import datetime

def create_main_build_dir(base_dir="hw/builds"): # <-- DEIXE ESTE COMO 'hw/builds'
    """Cria um diretório de build único para a execução atual."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    build_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(build_dir, exist_ok=True)
    print(f"Diretório principal da execução: {build_dir}")
    return build_dir

def find_latest_build_dir(base_dir="sw/builds"): # <-- MUDANÇA AQUI
    """Encontra o diretório de build mais recente para usar na Fase 2."""
    try:
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not subdirs:
            return None
        latest_subdir = max(subdirs, key=os.path.getmtime)
        return latest_subdir
    except FileNotFoundError:
        return None

def find_model_files(build_dir):
    """
    Encontra os ficheiros .pth do modelo otimizado dentro de um diretório de build.
    Retorna o caminho do modelo, a quantização e o ID da topologia.
    """
    pytorch_models_dir = os.path.join(build_dir, "pytorch_models")
    if not os.path.isdir(pytorch_models_dir):
        return None, None, None

    for f in os.listdir(pytorch_models_dir):
        if f.endswith("_final.pth"):
            # Ex: t2w4a4_final.pth
            base_name = f.replace(".pth", "")
            try:
                parts = base_name.split('w')
                topology_id = parts[0].replace('t', '')
                quant_str = parts[1].split('a')[0]
                quant = int(quant_str)
                return os.path.join(pytorch_models_dir, f), quant, topology_id
            except (IndexError, ValueError) as e:
                print(f"[AVISO] Não foi possível extrair metadados do nome do ficheiro: {f}. Erro: {e}")
                continue
    return None, None, None

def main():
    parser = argparse.ArgumentParser(
        description="HARA Co-Design: Ferramenta para otimização de Modelo e Hardware para FPGAs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'model-only', 'hardware-only'],
        help="""Define o modo de operação:
  - full:          (Padrão) Executa a Fase 1 (otimização de modelo) e, se bem-sucedida,
                   inicia automaticamente a Fase 2 (exploração de hardware).
  - model-only:    Executa o script 'run_model_optimization.py' para encontrar
                   e salvar os melhores modelos de software (.pth e .yaml).
  - hardware-only: Executa o script 'run_hardware_exploration.py' a partir
                   de um modelo já treinado. Requer --input-file."""
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default=None,
        help="Caminho para o ficheiro de modelo (.pth ou .onnx) para o modo 'hardware-only'."
    )
    
    # Argumentos para o modo hardware-only
    parser.add_argument('--topology-id', type=str, help="ID da topologia (ex: 'SAT6_T2'). Necessário para 'hardware-only'.")
    parser.add_argument('--quant', type=int, help="Largura de bits (ex: 4). Necessário para 'hardware-only'.")
    parser.add_argument('--build-dir', type=str, help="Diretório de build para 'hardware-only'. Se não especificado, um novo será criado.")

    args = parser.parse_args()
    
    hw_args = None

    if args.mode == 'full' or args.mode == 'model-only':
        print("\n=======================================================")
        print("                 INICIANDO FASE 1                 ")
        print("         Otimização de Modelo de Software         ")
        print("=======================================================")
        
        try:
            # Chama o script que agora salva em 'sw/builds'
            subprocess.run(["python3", "run_model_optimization.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERRO] A Fase 1 (otimização de modelo) falhou com o código de saída {e.returncode}.")
            return # Termina a execução se a Fase 1 falhar

        if args.mode == 'full':
            print("\n[INFO] Fase 1 concluída. Preparando para iniciar a Fase 2...")
            # Esta função agora procura em 'sw/builds'
            latest_build_dir = find_latest_build_dir() 
            if not latest_build_dir:
                print("[ERRO] Não foi possível encontrar o diretório de build da Fase 1 em 'sw/builds'.")
                return

            model_pth_path, quant, topology_id = find_model_files(latest_build_dir)
            if not model_pth_path:
                print("[ERRO] Nenhum modelo .pth final foi encontrado no diretório de build. A Fase 2 não pode continuar.")
                return
                
            hw_args = [
                "--build-dir", latest_build_dir, # Passa o diretório 'sw/builds/run_...'
                "--input-file", model_pth_path,
                "--topology-id", str(topology_id),
                "--quant", str(quant)
            ]
        else:
            print("\n[INFO] Fase 1 concluída. Modelos salvos.")
            return

    elif args.mode == 'hardware-only':
        if not args.input_file or args.topology_id is None or args.quant is None:
            parser.error("--input-file, --topology-id e --quant são obrigatórios para o modo 'hardware-only'.")
        
        # O modo 'hardware-only' usará 'hw/builds' (ou o que for passado)
        build_dir = args.build_dir or create_main_build_dir() 
        
        hw_args = [
            "--build-dir", build_dir,
            "--input-file", args.input_file,
            "--topology-id", str(args.topology_id),
            "--quant", str(args.quant)
        ]

    if hw_args:
        print("\n=======================================================")
        print("                 INICIANDO FASE 2                 ")
        print("            Exploração de Hardware            ")
        print("=======================================================")
        
        command = ["python3", "run_hardware_exploration.py"] + hw_args
        print(f"\n[INFO] Executando comando da Fase 2:\n{' '.join(command)}\n")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERRO] A Fase 2 (exploração de hardware) falhou com o código de saída {e.returncode}.")

if __name__ == "__main__":
    main()