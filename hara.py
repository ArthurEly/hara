import os
import torch
import shutil
import subprocess
import json
import time
import threading
from cnns_classes import t1_quantizedCNN, t2_quantizedCNN
from hw_utils import utils
from datetime import datetime

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
        'quant': [8]
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
    'step_out_of_context_synthesis'
    #'step_synthesize_bitfile', 
    #'step_make_pynq_driver', 
    #'step_deployment_package'
]

config = {
    'first_run': {
        'steps': default_steps,
        'target_fps': 1
    },
    "check": {
        'steps':first_steps,
        'target_fps': None
    },
    'hara': {
        'steps':default_steps,
        'target_fps': None
    }
}

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

def run_and_capture(args, timeout_sec=7200, log_path="build.log"):
    from io import StringIO
    import subprocess, threading, os

    output_log = StringIO()

    print(f"🛠️  [BUILD] Rodando subprocesso para {args[-1]}...")

    env = os.environ.copy()
    env["PYTHONBREAKPOINT"] = "0"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as log_file:
        with subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env
        ) as proc:
            # Define a função de monitoramento
            def monitor_output():
                for line in proc.stdout:
                    output_log.write(line)
                    log_file.write(line)
                    log_file.flush()

            # Inicia a thread
            t = threading.Thread(target=monitor_output)
            t.start()

            # Espera o tempo limite
            t.join(timeout=timeout_sec)

            # Se a thread ainda estiver viva, o processo travou
            if t.is_alive():
                proc.kill()
                t.join()  # Espera a thread terminar de fato
                log_file.write("\n⏱️ Build travado ou demorando demais\n")
                log_file.flush()
                raise RuntimeError("Build travado ou demorando demais")

    # Aguarda a thread acabar, caso ainda não tenha finalizado
    t.join()

    # Agora é seguro lidar com output e prints
    full_output = output_log.getvalue()

    if "Traceback" in full_output or "ValueError" in full_output:
        with open(log_path, "a") as log_file:
            log_file.write("\n❌ Erro detectado no build\n")
        print("❌ Erro detectado no build")  # só aqui o print é feito, após a thread
        raise RuntimeError("Erro detectado no build")

    return full_output


def try_alternate_foldings(base_folding, onnx_path, estimate_layer_cycles, base_args, build_dir, run, summary_file, resource_limits):
    suffixes = ["PE", "SIMD"]
    strategies = [dict(only_pe=True), dict(only_simd=True)]
    valid_foldings = []

    for i, (suffix, strategy) in enumerate(zip(suffixes, strategies)):
        folding = utils.modify_folding(base_folding, onnx_path, estimate_layer_cycles, **strategy)

        alt_hw_name = base_args[-1] + f"_{suffix}"
        folding_path = f"{build_dir}/{alt_hw_name}_folding.json"
        with open(folding_path, 'w') as f:
            json.dump(folding, f, indent=2)

        args = base_args.copy()
        args[args.index("--folding_file") + 1] = folding_path
        args[args.index("--hw_name") + 1] = alt_hw_name

        log_path = f"{build_dir}/build_{alt_hw_name}.log"

        try:
            start_time = time.time()
            run_and_capture(args, log_path=log_path)
            duration_sec = round(time.time() - start_time, 2)

            area_data = utils.extract_area_from_rpt(f"{build_dir}/{alt_hw_name}")
            resource_diffs = utils.check_resource_usage(area_data, resource_limits)
            exceeded = utils.raise_if_exceeds_limits(resource_diffs)
            if not exceeded:
                utils.append_run_summary(
                    summary_file, alt_hw_name, "success", folding,
                    duration=duration_sec, build_dir=f"{build_dir}/{alt_hw_name}"
                )
                valid_foldings.append((folding, alt_hw_name))
            else:
                utils.append_run_summary(
                    summary_file, alt_hw_name, "resources exceeded", folding,
                    duration=duration_sec, build_dir=f"{build_dir}/{alt_hw_name}"
                )
        except RuntimeError:
            utils.save_crash_report(f"{build_dir}/{alt_hw_name}")
            utils.append_run_summary(
                summary_file, alt_hw_name, "crash", folding,
                duration=0, build_dir=f"{build_dir}/{alt_hw_name}"
            )

    if not valid_foldings:
        print("[✗] Nenhum dos foldings alternativos foi válido. Encerrando.")
        return None, None
    elif len(valid_foldings) == 1:
        print("[✓] Apenas um folding alternativo funcionou. Continuando com ele.")
        return valid_foldings[0][0], valid_foldings[0][1]
    else:
        print("[✓] Ambos os foldings funcionaram. Selecionando o de maior throughput.")
        fps = []
        for _, name in valid_foldings:
            path = f"{build_dir}/{name}/report/estimate_network_performance.json"
            with open(path) as f:
                data = json.load(f)
                fps.append((data.get("estimated_throughput_fps", 0), name))
        best = max(fps, key=lambda x: x[0])[1]
        best_folding = [f for f, n in valid_foldings if n == best][0]
        return best_folding, best

bypass_first_phase = False
summary_file = f"{build_dir}/run_summary.csv"
#starting_build_dir = f"/home/arthurely/Desktop/finn/hara/builds/run_2025-04-21_04-51-11/t2w2_run4"
#starting_json = f"{starting_build_dir}/final_hw_config.json"
use_only_starting_json = False
starting_build_dir = None
starting_json = None

RESOURCE_LIMITS = {
    "Total LUTs": 20000,
    "FFs": 20000,
    "BRAM (36k)": 15,
    "DSP Blocks": 50
}

for tp in topologies:
    for quant in tp['quant']:
        # --- First run | Ainda precisa de revisão ---
        fps_first = config['first_run']['target_fps']
        first_hw_name = utils.get_hardware_config_name(quant=quant, topology=tp['id'], target_fps=fps_first)
        run = 0
        if not bypass_first_phase:
            print(f"\n🚀 [FIRST RUN] {first_hw_name} (run={run})")
            try:
                start_time = time.time()
                args = [
                    "python3", "run_build.py",
                    "--build_dir", str(build_dir),
                    "--topology", str(tp['id']),
                    "--target_fps", str(config['first_run']['target_fps']),
                    "--quant", str(quant),
                    "--steps", json.dumps(config['first_run']['steps']),
                    "--folding_file", "",
                    "--run", str(run),
                    "--hw_name", first_hw_name,
                ]
                print(f"{build_dir}/build_{first_hw_name}.log")
                run_and_capture(args, log_path=f"{build_dir}/build_{first_hw_name}.log")
                end_time = time.time()
                duration_sec = round(end_time - start_time, 2)
                
                folding = utils.read_folding_config(f"{build_dir}/{first_hw_name}")
                area_data = utils.extract_area_from_rpt(f"{build_dir}/{first_hw_name}")
                utils.raise_if_exceeds_limits(utils.check_resource_usage(area_data, RESOURCE_LIMITS))
                utils.append_run_summary(
                    file_path=summary_file,
                    hw_name=first_hw_name,
                    status="success",
                    folding_config=folding,
                    build_dir=f"{build_dir}/{first_hw_name}",
                    duration=duration_sec 
                )
                utils.plot_area_usage_from_csv(summary_file, f"{build_dir}/")
                print(f"[✓] First run OK for {first_hw_name}")
                last_valid_folding = folding
                last_valid_hw_name = first_hw_name
                prev_folding = folding
            except RuntimeError as e:
                end_time = time.time()
                duration_sec = round(end_time - start_time, 2)
                
                print(f"[X] First run crashed for {first_hw_name}")
                utils.save_crash_report(f"{build_dir}/{first_hw_name}")
                #utils.append_run_summary(summary_file, first_hw_name, "crash", folding)
                utils.append_run_summary(
                    file_path=summary_file,
                    hw_name=first_hw_name,
                    status="crash",
                    folding_config=folding,
                    build_dir=f"{build_dir}/{first_hw_name}",
                    duration=duration_sec 
                )
                flags = utils.get_exceeded_resources_flags(resource_diffs)
                print(f"[!] Recursos excedidos: {flags}")
                print(f"O hardware {first_hw_name} falhou na primeira execução. ")
                break

        # --- Prepare folding config for hara (iterativo até falhar) ---
        print(f"\n📐 [HARA] Iniciando exploração de paralelismo para t{tp['id']}w{quant}")
        folding_first = utils.read_folding_config(f"{build_dir}/{first_hw_name}")
        
        prev_folding = folding_first
        last_valid_folding = prev_folding
        last_valid_hw_name = first_hw_name
        run = 1
        consecutive_errors = 0
        max_errors = 10
        max_runs = -1
        
        flags = {
            "bram_exceed": False,
            "lut_exceed": False,
            "ff_exceed": False,
            "dsp_exceed": False
        }

        while consecutive_errors < max_errors and (run <= max_runs or max_runs == -1):
            if starting_json == None:
                print(f"\n🔁 [HARA RUN {run}] Modificando folding...")
            elif run <= 1:
                print(f"\n🔁 [HARA RUN {run}] Modificando folding com JSON previo: {starting_json}...")
            else:
                print(f"\n🔁 [HARA RUN {run}] Modificando folding com JSON da última rodada")

            hw_name_hara = utils.get_hardware_config_name(
                quant=quant,
                topology=tp['id'],
                target_fps=config['hara']['target_fps'],
                extra=f"_run{run}"
            )

            # Decide qual folding usar
            first_json = None
            if starting_json and use_only_starting_json:
                try:
                    with open(starting_json, 'r') as f:
                        folding_hara = json.load(f)
                        first_json = json.load(f)
                except Exception as e:
                    print(f"[X] Erro ao carregar folding: {e}")
                    break
            else:
                last_build_dir = f"{build_dir}/{last_valid_hw_name}"
                
                #if consecutive_errors > 0:
                #    last_build_dir = f"{build_dir}/{hw_name_hara}"
                    
                if (starting_json and (run <= 1)):
                    print(1)
                    with open(starting_json, 'r') as f:
                        folding_input = json.load(f)
                    last_build_dir = starting_build_dir
                else:
                    print(2)
                    folding_input = prev_folding

                if run <= 1 and consecutive_errors > 0:
                    last_build_dir = f"{build_dir}/{last_valid_hw_name}"

                print(f"[✓] Usando folding: {folding_input}")
                
                if starting_build_dir:
                    onnx_path = f"{starting_build_dir}/intermediate_models/step_generate_estimate_reports.onnx"
                else:
                    onnx_path = f"{build_dir}/{first_hw_name}/intermediate_models/step_generate_estimate_reports.onnx"
                
                print(f"Mudando aqui 1: {last_build_dir}/report/estimate_layer_cycles.json")
                estimate_layer_cycles_path = f"{last_build_dir}/report/estimate_layer_cycles.json"
                with open(estimate_layer_cycles_path, 'r') as f:
                        estimate_layer_cycles = json.load(f)
            
            if consecutive_errors > 0:
                if flags["bram_exceed"] and (not flags["lut_exceed"]) and (not flags["ff_exceed"]) and (not flags["dsp_exceed"]):
                    folding_hara = utils.modify_folding(folding_input, onnx_path, estimate_layer_cycles)
                    #args = [
                    #    "python3", "run_build.py",
                    #    "--build_dir", str(build_dir),
                    #    "--topology", str(tp['id']),
                    #    "--target_fps", str(config['check']['target_fps']),
                    #    "--quant", str(quant),
                    #    "--steps", json.dumps(config['check']['steps']),
                    #    "--folding_file", folding_path_hara,
                    #    "--run", str(run),
                    #    "--hw_name", hw_name_hara,
                    #]
                    #
                    #for _ in range(consecutive_errors):                        
                    #    run_and_capture(args, log_path=f"{build_dir}/build_{hw_name_hara}.log")
                    #    estimate_layer_cycles_path = f"{build_dir}/{hw_name_hara}/report/estimate_layer_cycles.json"
                    #    with open(estimate_layer_cycles_path, 'r') as f:
                    #            estimate_layer_cycles = json.load(f)                        
                    #    folding_hara = utils.modify_folding(folding_hara, onnx_path, estimate_layer_cycles)
                else:                  
                    print(f"[!] Recursos excedidos no try_alternate_foldings: {flags}")
                    folding_hara, hw_name_hara = try_alternate_foldings(
                        folding_input, onnx_path, estimate_layer_cycles,
                        base_args=args, build_dir=build_dir, run=run,
                        summary_file=summary_file, resource_limits=RESOURCE_LIMITS
                    )
                    if folding_hara is None and hw_name_hara is None:
                        print("[✗] Nenhum folding alternativo válido encontrado. Encerrando.")
                        break
                    last_valid_folding = folding_hara
                    last_valid_hw_name = hw_name_hara
                    print("LHN error: " + last_valid_hw_name)
                    prev_folding = folding_hara
                    consecutive_errors = 0
                    run += 1
                    print(f"Partindo pra próxima rodada...")
                    continue
            else:
                folding_hara = utils.modify_folding(folding_input, onnx_path, estimate_layer_cycles)
        
            if folding_hara == prev_folding or folding_hara == first_json:
                print(f"[✓] Folding estável alcançado após {run - 1} iteração(ões).")
                break

            folding_path_hara = f"{build_dir}/{hw_name_hara}_folding.json"
            os.makedirs(os.path.dirname(folding_path_hara), exist_ok=True)
            
            print(folding_path_hara)
            
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
                start_time = time.time()
                run_and_capture(args, log_path=f"{build_dir}/build_{hw_name_hara}.log")
                end_time = time.time()
                duration_sec = round(end_time - start_time, 2)
                
                area_data = utils.extract_area_from_rpt(f"{build_dir}/{hw_name_hara}")
                resource_diffs = utils.check_resource_usage(area_data, RESOURCE_LIMITS)
                exceed = utils.raise_if_exceeds_limits(resource_diffs)
                print(area_data)
                print(resource_diffs)
                if exceed:
                    consecutive_errors += 1
                    flags = utils.get_exceeded_resources_flags(resource_diffs)
                    print(f"[!] Recursos excedidos: {exceed}")
                    utils.append_run_summary(
                        file_path=summary_file,
                        hw_name=hw_name_hara,
                        status="resources exceeded",
                        folding_config=folding_hara,
                        build_dir=f"{build_dir}/{hw_name_hara}",
                        duration=duration_sec  # novo campo
                    )
                    if flags["bram_exceed"] and (not flags["lut_exceed"]) and (not flags["ff_exceed"]) and (not flags["dsp_exceed"]):
                        last_valid_folding = folding_hara
                        last_valid_hw_name = hw_name_hara
                        print("LHN bram: " + last_valid_hw_name)
                        prev_folding = folding_hara
                    continue
                else:
                    flags = {
                        "bram_exceed": False,
                        "lut_exceed": False,
                        "ff_exceed": False,
                        "dsp_exceed": False
                    }
                    utils.append_run_summary(
                        file_path=summary_file,
                        hw_name=hw_name_hara,
                        status="success",
                        folding_config=folding_hara,
                        build_dir=f"{build_dir}/{hw_name_hara}",
                        duration=duration_sec  # novo campo
                    )
                    consecutive_errors = 0
                    utils.plot_area_usage_from_csv(summary_file, f"{build_dir}/")
                    last_valid_folding = folding_hara
                    last_valid_hw_name = hw_name_hara
                    print("LHN ok: " + last_valid_hw_name)
                    prev_folding = folding_hara
                    print(prev_folding)
            except RuntimeError as e:
                end_time = time.time()
                duration_sec = round(end_time - start_time, 2)

                print(f"[✗] Subprocesso falhou para {hw_name_hara}")
                utils.save_crash_report(f"{build_dir}/{hw_name_hara}")
                utils.append_run_summary(
                    file_path=summary_file,
                    hw_name=hw_name_hara,
                    status="crash",
                    folding_config=folding_hara,
                    build_dir=f"{build_dir}/{hw_name_hara}",
                    duration=duration_sec  # novo campo também no erro
                )
                #utils.plot_area_usage_from_csv(summary_file, f"{build_dir}/")
                consecutive_errors += 1
                run += 1
                print(f"[!] Falhas consecutivas: {consecutive_errors}/{max_errors}")

            run += 1

        print(f"[🏁] Encerrando após {run-1} iterações.")

        final_folding_path = f"{build_dir}/{last_valid_hw_name}_final_folding.json"
        with open(final_folding_path, 'w') as f:
            json.dump(last_valid_folding, f, indent=2)
        print(f"[✓] Último folding válido salvo em {final_folding_path}")