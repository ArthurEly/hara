# main.py
import argparse
import os
import subprocess
import re
import glob

def run_command(command):
    """Executa um comando no shell, mostra o comando e a saída em tempo real."""
    print(f"\n[CMD] Executando: {' '.join(command)}")
    try:
        # Usando Popen para streaming de output em tempo real
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            output_lines.append(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
            
        return "".join(output_lines)
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] O comando falhou com o código de saída {e.returncode}.")
        exit(1) # Encerra o script principal em caso de erro

def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador do Fluxo de Co-Design HARA.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Argumentos para controlar o fluxo, como no original
    parser.add_argument(
        '--mode', type=str, default='model-only',
        choices=['full', 'model-only', 'hardware-only'],
        help="""Define o modo de operação:
  - full:          Executa a Fase 1 e depois a Fase 2.
  - model-only:    Executa apenas a Fase 1.
  - hardware-only: Executa apenas a Fase 2 (requer flags adicionais)."""
    )
    # Argumentos específicos para o modo 'hardware-only'
    parser.add_argument(
        '--build-dir', type=str, help="Diretório de build (obrigatório para --mode=hardware-only)."
    )
    parser.add_argument(
        '--input-file', type=str, help="Arquivo de entrada .pth (obrigatório para --mode=hardware-only)."
    )
    parser.add_argument(
        '--topology-id', type=str, help="ID da topologia (obrigatório para --mode=hardware-only)."
    )
    parser.add_argument(
        '--quant', type=int, help="Largura de bits (obrigatório para --mode=hardware-only)."
    )
    
    args = parser.parse_args()
    python_executable = "python3" # Ou "python", dependendo do seu ambiente

    if args.mode == 'model-only':
        print("--- MODO: OTIMIZAÇÃO DE MODELO (FASE 1) ---")
        cmd = [python_executable, "run_model_optimization.py"]
        run_command(cmd)

    elif args.mode == 'hardware-only':
        print("--- MODO: EXPLORAÇÃO DE HARDWARE (FASE 2) ---")
        if not all([args.build_dir, args.input_file, args.topology_id, args.quant]):
            parser.error("--build-dir, --input-file, --topology-id, e --quant são obrigatórios para o modo hardware-only.")
        
        cmd = [
            python_executable, "run_hardware_exploration.py",
            "--build-dir", args.build_dir,
            "--input-file", args.input_file,
            "--topology-id", args.topology_id,
            "--quant", str(args.quant)
        ]
        run_command(cmd)

    elif args.mode == 'full':
        print("--- MODO: FLUXO COMPLETO (FASE 1 + FASE 2) ---")
        
        # Fase 1
        print("\n--- [FASE 1] Executando otimização de modelo... ---")
        cmd_phase1 = [python_executable, "run_model_optimization.py"]
        phase1_output = run_command(cmd_phase1)
        
        # Extrai o diretório de build da saída do script da Fase 1
        build_dir_match = re.search(r"Diretório principal da execução: (.*)", phase1_output)
        if not build_dir_match:
            print("[ERRO] Não foi possível determinar o diretório de build da saída da Fase 1.")
            exit(1)
        build_dir = build_dir_match.group(1).strip()
        print(f"\n[INFO] Diretório de build detectado: {build_dir}")

        # Fase 2
        print("\n--- [FASE 2] Buscando modelos .pth para exploração de hardware... ---")
        pytorch_models_dir = os.path.join(build_dir, "pytorch_models")
        model_paths = glob.glob(os.path.join(pytorch_models_dir, "*.pth"))

        if not model_paths:
            print("[AVISO] Nenhum modelo .pth encontrado para a Fase 2.")
        else:
            for model_path in model_paths:
                filename = os.path.basename(model_path)
                # Extrai topologia e quantização do nome do arquivo
                match = re.search(r"t(\w+)w(\d+)_final\.pth", filename)
                if match:
                    topology_id, quant = match.groups()
                    print(f"\n--- Processando {filename} ---")
                    cmd_phase2 = [
                        python_executable, "run_hardware_exploration.py",
                        "--build-dir", build_dir,
                        "--input-file", model_path,
                        "--topology-id", topology_id,
                        "--quant", quant
                    ]
                    run_command(cmd_phase2)
                else:
                    print(f"[AVISO] Nome de arquivo não padronizado, pulando: {filename}")

    print("\n[SUCESSO] Fluxo principal concluído.")

if __name__ == "__main__":
    main()