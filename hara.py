import os
import torch
import torch
import os
import shutil
import subprocess
from cnns_classes import t1_quantizedCNN, t2_quantizedCNN
import torch
import json
from hw_utils import utils
from datetime import datetime
import time
import threading

# Crie um diretório base único para esta execução
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
build_dir = f"/home/arthurely/Desktop/finn/hara/builds/run_{timestamp}"
os.makedirs(build_dir, exist_ok=True)

# Se quiser, salve esse caminho num lugar (ex: symlink "last_run")
if os.path.islink("last_run"):
    os.unlink("last_run")
os.symlink(build_dir, "last_run")

target_fps_list = [501]
device = torch.device('cpu')

folding_json = None
finn_build_dir = os.environ["FINN_BUILD_DIR"] + '/'

target_fps = 1
topologies = [
    {
        'id':2, 
        'tp_class':t2_quantizedCNN,
        'quant': [4]
    }
]

first_steps = [
    'step_qonnx_to_finn', 
    'step_tidy_up', 
    'step_streamline', 
    'step_convert_to_hw', 
    'step_create_dataflow_partition', 
    'step_specialize_layers', 
    'step_target_fps_parallelization', 
    'step_apply_folding_config', 
    'step_minimize_bit_width', 
    'step_generate_estimate_reports', 
]

default_steps =  [
    'step_qonnx_to_finn', 
    'step_tidy_up', 
    'step_streamline', 
    'step_convert_to_hw', 
    'step_create_dataflow_partition', 
    'step_specialize_layers', 
    'step_target_fps_parallelization', 
    'step_apply_folding_config', 
    'step_minimize_bit_width', 
    'step_generate_estimate_reports', 
    'step_hw_codegen', 
    'step_hw_ipgen', 
    'step_set_fifo_depths', 
    'step_create_stitched_ip', 
    'step_synthesize_bitfile', 
    'step_make_pynq_driver', 
    'step_deployment_package'
]

config = {
    'first_run': {
        'steps': default_steps,
        'target_fps': 1
    },
    'hara': {
        'steps':default_steps,
        'target_fps': None
    }
}
bypass_first_phase = True
summary_file = f"{build_dir}/run_summary.csv"

def tail_log(log_path):
    """Imprime linhas novas do log em tempo real."""
    with open(log_path, "r") as f:
        f.seek(0, os.SEEK_END)  # Vai para o final do arquivo
        while True:
            line = f.readline()
            if line:
                print("[LOG]", line.strip())
            else:
                time.sleep(0.2)  # Espera um pouco por novas linhas

import threading

def run_and_capture(args, timeout_sec=300):
    from io import StringIO
    import sys

    output_log = StringIO()

    print(f"🛠️  [BUILD] Rodando subprocesso para {args[-1]}...")

    env = os.environ.copy()
    env["PYTHONBREAKPOINT"] = "0"

    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,  # <<<< aqui
        text=True,
        bufsize=1,
        env=env  # <<<<< isso aqui evita entrar no PDB
    ) as proc:
        def monitor_output():
            for line in proc.stdout:
                print("----------------------")
                print(line, end='')
                output_log.write(line)

        t = threading.Thread(target=monitor_output)
        t.start()
        t.join(timeout=timeout_sec)

        if t.is_alive():
            proc.kill()
            raise RuntimeError("Build travado ou demorando demais")

    full_output = output_log.getvalue()

    if "Traceback" in full_output or "ValueError" in full_output:
        raise RuntimeError("Erro detectado no build")

    return full_output


for tp in topologies:
    for quant in tp['quant']:
        # --- First run | Ainda precisa de revisão ---
        fps_first = config['first_run']['target_fps']
        first_hw_name = utils.get_hardware_config_name(quant=quant, topology=tp['id'], target_fps=fps_first)
        run = 0
        if not bypass_first_phase:
            has_error = False
            while not has_error:
                print(f"\n🚀 [FIRST RUN] {first_hw_name} (run={run})")
                utils.build_hardware(
                    topology=tp['id'],
                    target_fps=fps_first,
                    topology_class=tp['tp_class'],
                    quant=quant,
                    steps=config['first_run']['steps'],
                    folding_file=None,
                    run=run,
                    hw_name=first_hw_name
                )

                log_text = utils.read_build_log(f"{build_dir}/{first_hw_name}")
                has_error = utils.detect_synthesis_error(log_text)

                if has_error:
                    print(f"[X] First run failed for {first_hw_name}")
                    utils.save_crash_report(f"{build_dir}/{first_hw_name}")
                    utils.append_run_summary(summary_file, first_hw_name, "fail", folding)
                    run -= 1
                else:
                    print(f"[✓] First run OK for {first_hw_name}")
                    folding = utils.read_folding_config(f"{build_dir}/{first_hw_name}")
                    print(json.dumps(folding, indent=2))
                    utils.append_run_summary(summary_file, first_hw_name, "ok", folding)
                    break

        # --- Prepare folding config for hara (iterativo até falhar) ---
        print(f"\n📐 [HARA] Iniciando exploração de paralelismo para t{tp['id']}w{quant}")
        folding_first = utils.read_folding_config(f"{build_dir}/{first_hw_name}")
        
        prev_folding = folding_first
        last_valid_folding = prev_folding
        last_valid_hw_name = None
        run = 1
        consecutive_errors = 0
        max_errors = 3
        starting_json = "/home/arthurely/Desktop/finn/hara/builds/t2w4_run3_folding.json"
        pre_built_accelerator = "/home/arthurely/Desktop/finn/hara/builds/t2w4_1fps"

        while consecutive_errors < max_errors:
            if starting_json == None:
                print(f"\n🔁 [HARA RUN {run}] Modificando folding...")
            else:
                print(f"\n🔁 [HARA RUN {run}] Modificando folding com JSON previo: {starting_json}...")

            hw_name_hara = utils.get_hardware_config_name(
                quant=quant,
                topology=tp['id'],
                target_fps=config['hara']['target_fps'],
                extra=f"_run{run}"
            )

            if starting_json in [None]:
                folding_input = prev_folding
            else:
                folding_input_path = starting_json
                try:
                    with open(folding_input_path, 'r') as f:
                        folding_input = json.load(f)
                except Exception as e:
                    print(f"[X] Erro ao carregar folding: {e}")
                    break

            if pre_built_accelerator == None:
                onnx_path = f"{build_dir}/{first_hw_name}/intermediate_models/step_target_fps_parallelization.onnx"
            else: 
                onnx_path = f"{pre_built_accelerator}/intermediate_models/step_target_fps_parallelization.onnx"

            folding_hara = utils.modify_folding(folding_input, onnx_path)

            if folding_hara == prev_folding:
                print(f"[✓] Folding estável alcançado após {run - 1} iteração(ões).")
                break

            folding_path_hara = f"{build_dir}/{hw_name_hara}_folding.json"
            os.makedirs(os.path.dirname(folding_path_hara), exist_ok=True)

            # argumentos para subprocesso
            args = [
                "python3", "run_build.py",
                "--build_dir", str(build_dir),
                "--topology", str(tp['id']),
                "--target_fps", str(config['hara']['target_fps']),
                "--quant", str(quant),
                "--steps", json.dumps(config['hara']['steps']),
                "--folding_file", folding_path_hara,
                "--run", str(run),
                "--hw_name", hw_name_hara,
            ]

            with open(folding_path_hara, 'w') as f:
                json.dump(folding_hara, f, indent=2)

            try:
                try:
                    run_and_capture(args)
                except RuntimeError as e:
                    print(f"[✗] Subprocesso falhou: {e}")

                # sucesso: reset contador de erros
                consecutive_errors = 0
                utils.get_zynq_proj(src=finn_build_dir, dst=f"{build_dir}/{hw_name_hara}/zynq_proj/")
                utils.append_run_summary(summary_file, hw_name_hara, "ok", folding_hara)
                last_valid_folding = folding_hara
                last_valid_hw_name = hw_name_hara
                prev_folding = folding_hara

            except subprocess.CalledProcessError as e:
                print(f"[✗] Subprocesso falhou para {hw_name_hara}")
                print("[stdout]:", e.stdout)
                print("[stderr]:", e.stderr)
                utils.save_crash_report(f"{build_dir}/{hw_name_hara}")
                utils.append_run_summary(summary_file, hw_name_hara, "fail", folding_hara)
                consecutive_errors += 1
                print(f"[!] Falhas consecutivas: {consecutive_errors}/{max_errors}")

            run += 1

        print(f"[🏁] Encerrando após {run-1} iterações.")

        final_folding_path = f"{build_dir}/{last_valid_hw_name}_final_folding.json"
        with open(final_folding_path, 'w') as f:
            json.dump(last_valid_folding, f, indent=2)
        print(f"[✓] Último folding válido salvo em {final_folding_path}")