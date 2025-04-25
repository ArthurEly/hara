import pickle
import pandas as pd
import os
import shutil
import json 
from datetime import datetime
import csv
import onnx
import onnx
from onnx import helper
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.fold_constants import FoldConstants
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.core.datatype import DataType
import torch
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hls
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_data_layouts import InferDataLayouts
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
import csv
import json
from datetime import datetime 
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

class utils():
    def __init__(self):
        super(utils, self).__init__()
        
    def save_object(filename, object):
        with open(filename, 'wb') as outp:
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
    
    def read_object(filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    def get_model_output_filename (topology,quant):
        return f"./pytorch_models/sat6-cnn-t{topology}w{quant}.pt"

    def get_hardware_config_name(topology, quant, target_fps, extra=''):
        fps_part = f"_{target_fps}fps" if target_fps is not None else ""
        return f"t{topology}w{quant}{fps_part}{extra}"


    def save_csv_table(results,csv_pathname):
        df = pd.DataFrame(results)
        print(df.to_string(header=None, index=False))
        df.to_csv(csv_pathname) 
        print(f"succesfully saved at {csv_pathname}")

    def move_intermediate_outputs_dir(src, dst):
        source_folder = src
        destination_folder = dst
        
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            
        # fetch all files
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name
            destination = destination_folder + file_name
            # move only files
            if "pyverilator_ipstitched" in file_name:
                shutil.rmtree(source)
            else:
                try:
                    shutil.move(source, destination)
                    print("Arquivo movido com sucesso!")
                except FileNotFoundError:
                    print("Erro: Arquivo de origem não encontrado.")
                except PermissionError:
                    print("Erro: Permissão negada para mover o arquivo.")
                except Exception as e:
                    print(f"Erro inesperado: {e}")

        print('All files moved successfully')

    def get_zynq_proj(src,dst):      
        for folder in os.listdir(src):
            if "vivado_zynq_proj" in folder:
                try:
                    shutil.copytree(src + folder, dst)
                    print("Arquivo movido com sucesso!")
                except FileNotFoundError:
                    print("Erro: Arquivo de origem não encontrado.")
                except PermissionError:
                    print("Erro: Permissão negada para mover o arquivo.")
                except Exception as e:
                    print(f"Erro inesperado: {e}")
                break
        print('ZYNQ project successfully copied')

    def read_build_log(build_dir):
        log_path = os.path.join(build_dir, "build_dataflow.log")
        if not os.path.exists(log_path):
            return ""
        with open(log_path, "r") as f:
            return f.read()

    def detect_synthesis_error(log_text, timeout_keywords=None):
        if timeout_keywords is None:
            timeout_keywords = ["ERROR", "failed", "timeout", "synthesis failed", "crash"]

        for line in log_text.splitlines():
            if any(keyword.lower() in line.lower() for keyword in timeout_keywords):
                return True
        return False

    def read_folding_config(build_dir):
        fold_path = os.path.join(build_dir, "auto_folding_config.json")
        if not os.path.exists(fold_path):
            return {}
        with open(fold_path, "r") as f:
            return json.load(f)

    def save_crash_report(build_dir, destination_base="./crash_reports"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dest_dir = os.path.join(destination_base, f"crash_{timestamp}")
        os.makedirs(dest_dir, exist_ok=True)

        files_to_copy = ["build_dataflow.log", "auto_folding_config.json"]
        for fname in files_to_copy:
            src_path = os.path.join(build_dir, fname)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(dest_dir, fname))

        print(f"[!] Crash report saved at: {dest_dir}")
        return dest_dir
    
    def extract_area_from_rpt(build_dir):
        rpt_file = None
        for root, _, files in os.walk(build_dir):
            for file in files:
                if file.endswith('finn_design_partition_util.rpt'):
                    rpt_file = os.path.join(root, file)
                    break
            if rpt_file:
                break

        if not rpt_file:
            return None

        with open(rpt_file) as f:
            for line in f:
                if "finn_design_wrapper" in line and "(top)" in line:
                    tokens = line.split()
                    try:
                        total_luts = int(tokens[5])
                        logic_luts = int(tokens[7])
                        lutrams = int(tokens[9])
                        srls = int(tokens[11])
                        ffs = int(tokens[13])
                        ramb36 = int(tokens[15])
                        ramb18 = int(tokens[17])
                        dsp = int(tokens[19])
                        bram_36k = ramb36 + ramb18 / 2.0

                        return {
                            "Total LUTs": total_luts,
                            "Logic LUTs": logic_luts,
                            "LUTRAMs": lutrams,
                            "SRLs": srls,
                            "FFs": ffs,
                            "BRAM (36k)": round(bram_36k, 1),
                            "DSP Blocks": dsp
                        }
                    except (IndexError, ValueError):
                        return None
        return None

    
    def dict_diff(prev, curr):
        changes = {}
        for layer in curr:
            if layer not in prev or prev[layer] != curr[layer]:
                changes[layer] = {"from": prev.get(layer), "to": curr[layer]}
        return changes
    
    def append_run_summary(file_path, hw_name, status, folding_config, duration, build_dir):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prev_folding = {}
        folding_diff = {}

        # Tenta calcular o diff com a entrada anterior
        if os.path.isfile(file_path):
            try:
                df = pd.read_csv(file_path)
                df = df[df['status'] == 'success']
                if not df.empty:
                    last_row = df.iloc[-1]
                    prev_folding = json.loads(last_row['folding_summary'])
                    folding_diff = utils.dict_diff(prev_folding, folding_config)
            except Exception as e:
                print(f"[!] Erro ao processar folding anterior: {e}")

        # Informações base
        summary = {
            "date": now,
            "hw_name": hw_name,
            "status": status,
            "duration_in_seconds": duration,
            "folding_summary": json.dumps(folding_config),
            "folding_diff": json.dumps(folding_diff),
            "build_dir": build_dir,
        }

        # Dados de área
        area_data = utils.extract_area_from_rpt(build_dir)
        if area_data:
            summary.update(area_data)

        # Dados de throughput
        perf_path = os.path.join(build_dir, "report", "estimate_network_performance.json")
        if os.path.isfile(perf_path):
            try:
                with open(perf_path, 'r') as f:
                    perf_data = json.load(f)
                    summary["estimated_throughput_fps"] = perf_data.get("estimated_throughput_fps", None)
                    summary["max_cycles_node_name"] = perf_data.get("max_cycles_node_name", None)
            except Exception as e:
                print(f"[!] Erro ao ler estimated throughput: {e}")

        # Ordem dos campos no CSV
        field_order = [
            "date", "hw_name", "status", "duration_in_seconds",
            "folding_summary", "folding_diff", "build_dir",
            "Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs", "FFs", "BRAM (36k)", "DSP Blocks",
            "estimated_throughput_fps", "max_cycles_node_name"
        ]

        # Escrita no CSV
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_order)
            if not file_exists:
                writer.writeheader()
            writer.writerow(summary)

            
    def plot_area_usage_from_csv(csv_path, output_dir=None):
        if not os.path.isfile(csv_path):
            print(f"[!] Arquivo CSV não encontrado: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        df = df[df['status'] == 'success']

        if df.empty:
            print("[!] Nenhuma build com status 'success' encontrada.")
            return

        # Função para extrair número de hw_name para ordenação
        def extract_number(hw_name):
            match = re.search(r'(\d+)', hw_name)
            return int(match.group(1)) if match else float('inf')

        df = df.sort_values(by='hw_name', key=lambda x: x.apply(extract_number))

        resource_types = ['Total LUTs', 'FFs', 'BRAM (36k)', 'DSP Blocks']
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

        for i, res in enumerate(resource_types):
            if res not in df.columns:
                continue

            x = df['hw_name']
            y = df[res]

            plt.figure(figsize=(12, 5))
            plt.bar(x, y, color=colors[i], alpha=0.7, label=res)
            plt.plot(x, y, color=colors[i], linestyle='--', marker='o', linewidth=1.5, alpha=0.8)

            plt.title(f"Evolução do uso de {res}")
            plt.xlabel("Configuração de hardware")
            plt.ylabel(res)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.legend()

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"{res.replace(' ', '_')}.png")
                plt.savefig(filename)
            else:
                plt.show()

            plt.close()

    def modify_folding(folding, onnx_path, estimate_layer_cycles,
                    only_pe=False, only_simd=False, mvau_wwidth_max=36, max_pe=128):
        import onnx
        from onnx import helper

        def get_layer_features(path):
            model = onnx.load(path)
            feats = {}
            for node in model.graph.node:
                name = node.name
                op = node.op_type
                attr = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                f = {"op_type": op}
                if op in ["MVAU_hls", "MVAU_rtl"]:
                    f["MW"] = int(attr.get("MW", 0))
                    f["MH"] = int(attr.get("MH", 0))
                    f["WBits"] = int(attr.get("WBits", 1))
                if op.startswith("ConvolutionInputGenerator"):
                    f["IFMChannels"] = int(attr.get("IFMChannels", 0))
                    f["depthwise"]    = int(attr.get("depthwise", 0))
                if op in ["FMPadding_rtl", "FMPadding_hls"]:
                    f["NumChannels"] = int(attr.get("NumChannels", 0))
                feats[name] = f
            return feats

        def next_divisor(n, current):
            """
            Retorna o menor divisor de n que é > current. 
            Se não houver, retorna None.
            """
            for d in range(current+1, n+1):
                if n % d == 0:
                    return d
            return None

        feature_dims = get_layer_features(onnx_path)
        new_folding = {}
        layers_modified = False

        # Camadas críticas (maior latência estimada)
        max_lat = max(estimate_layer_cycles.values())
        critical = {k for k,v in estimate_layer_cycles.items() if v == max_lat}

        for layer, cfg in folding.items():
            if layer == "Defaults":
                new_folding[layer] = {k: cfg[k] for k in ("PE","SIMD") if k in cfg}
                continue

            new_cfg = {k: cfg[k] for k in ("PE","SIMD","parallel_window") if k in cfg}
            f = feature_dims.get(layer, {})
            modified = False

            if layer in critical:
                op = f.get("op_type","")

                # Caso especial de ConvInpGen depthwise=0
                if op.startswith("ConvolutionInputGenerator") and f.get("depthwise",1)==0:
                    if new_cfg.get("SIMD",1) == f.get("IFMChannels",0):
                        new_cfg["parallel_window"] = 1
                        modified = True

                # 1) Avançar SIMD para o próximo divisor da dimensão relevante
                if not only_pe and "SIMD" in new_cfg:
                    # escolhe a dimensão: MW, IFMChannels ou NumChannels
                    dim = f.get("MW") or f.get("IFMChannels") or f.get("NumChannels")
                    simd0 = new_cfg["SIMD"]
                    nxt = next_divisor(dim, simd0)
                    if nxt is not None:
                        # verifica condição de largura de stream p/ MVAU
                        if not op.startswith("MVAU") or (nxt * f.get("WBits",1)) <= new_cfg.get("PE",1) * mvau_wwidth_max:
                            new_cfg["SIMD"] = nxt
                            modified = True

                # 2) Só avança PE se SIMD não mudou
                if not modified and not only_simd and "PE" in new_cfg:
                    mh = f.get("MH")
                    pe0 = new_cfg["PE"]
                    nxt_pe = next_divisor(mh, pe0)
                    if nxt_pe is not None and nxt_pe <= max_pe:
                        new_cfg["PE"] = nxt_pe
                        modified = True

            if modified:
                layers_modified = True
                print(f"[↑] {layer}: PE {cfg.get('PE')}→{new_cfg.get('PE')}, SIMD {cfg.get('SIMD')}→{new_cfg.get('SIMD')}")
            #else:
                #print(f"[=] {layer}: sem modificação (PE {cfg.get('PE')}, SIMD {cfg.get('SIMD')})")

            new_folding[layer] = new_cfg

        return folding if not layers_modified else new_folding

    def get_exceeded_resources_flags(resource_diffs):
        return {
            "lut_exceed": resource_diffs.get("Total LUTs", 0) < 0,
            "ff_exceed": resource_diffs.get("FFs", 0) < 0,
            "bram_exceed": resource_diffs.get("BRAM (36k)", 0) < 0,
            "dsp_exceed": resource_diffs.get("DSP Blocks", 0) < 0,
        }


    def check_resource_usage(area_data, limits):
        if area_data is None:
            print("[!] Nenhum dado de área encontrado.")
            return {}
        diffs = {}
        for res, max_val in limits.items():
            used = area_data.get(res, 0)
            diffs[res] = max_val - used
        return diffs

    def get_exceeded_resources_flags(resource_diffs):
        return {
            "lut_exceed": resource_diffs.get("Total LUTs", 0) < 0,
            "ff_exceed": resource_diffs.get("FFs", 0) < 0,
            "bram_exceed": resource_diffs.get("BRAM (36k)", 0) < 0,
            "dsp_exceed": resource_diffs.get("DSP Blocks", 0) < 0,
        }

    
    def raise_if_exceeds_limits(resource_diffs):
        exceeded = {res: -diff for res, diff in resource_diffs.items() if diff < 0}
        if exceeded:
            msg_lines = ["[🚫] Recursos excedidos:"]
            for res, amount in exceeded.items():
                msg_lines.append(f"  - {res}: excedido por {amount}")
            return exceeded
        return None


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

    def build_hardware(build_dir,topology, target_fps, topology_class, quant, steps, folding_file, run, hw_name):
        def make_onnx(build_dir, cnv,quant,topology):
            cnv.load_state_dict(torch.load(f"../notebooks/sat6_cnn/pytorch_models/sat6-cnn-t{topology}w{quant}.pt"))
            onnx_output_filename = f"../notebooks/sat6_cnn/hardware_onnxs/sat6-cnn-t{topology}w{quant}.onnx"
            
            export_onnx_path = build_dir + f"/end2end_cnv_t{topology}w{quant}_export.onnx"
            #tidy up
            export_qonnx(cnv, torch.randn(1, 4, 32, 32), export_onnx_path)    
            qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
            model = ModelWrapper(export_onnx_path)
            model = model.transform(ConvertQONNXtoFINN())
            model = model.transform(InferShapes())
            model = model.transform(FoldConstants())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(RemoveStaticGraphInputs())

            #preprocessing
            global_inp_name = model.graph.input[0].name
            model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

            # postprocessing: insert Top-1 node at the end
            model = model.transform(InsertTopK(k=1))
            # tidy-up again
            model = model.transform(InferShapes())
            model = model.transform(FoldConstants())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferDataTypes())
            model = model.transform(RemoveStaticGraphInputs())

            model = model.transform(MoveScalarLinearPastInvariants())
            model = model.transform(Streamline())
            model = model.transform(LowerConvsToMatMul())
            model = model.transform(MakeMaxPoolNHWC())
            model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
            model = model.transform(ConvertBipolarMatMulToXnorPopcount())
            model = model.transform(Streamline())
            # absorb final add-mul nodes into TopK
            model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
            model = model.transform(InferDataLayouts())
            model = model.transform(RemoveUnusedTensors())

            # choose the memory mode for the MVTU units, decoupled or const
            model = model.transform(to_hls.InferBinaryMatrixVectorActivation())
            model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
            # TopK to LabelSelect
            model = model.transform(to_hls.InferLabelSelectLayer())
            # input quantization (if any) to standalone thresholding
            model = model.transform(to_hls.InferThresholdingLayer())
            model = model.transform(to_hls.InferConvInpGen())
            model = model.transform(to_hls.InferStreamingMaxPool())
            # get rid of Reshape(-1, 1) operation between hlslib nodes
            model = model.transform(RemoveCNVtoFCFlatten())
            # get rid of Tranpose -> Tranpose identity seq
            model = model.transform(absorb.AbsorbConsecutiveTransposes())
            # infer tensor data layouts
            model = model.transform(InferDataLayouts())
            parent_model = model.transform(CreateDataflowPartition())
            parent_model.save(build_dir + f"/end2end_cnv_t{topology}w{quant}_dataflow_parent.onnx")
            sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            # save the dataflow partition with a different name for easier access
            dataflow_model = ModelWrapper(dataflow_model_filename)
            dataflow_model.save(onnx_output_filename)
            return onnx_output_filename

        def generate_hardware(build_dir, topology, quant, target_fps, steps, folding_file, run, hw_name):
            model_file = f"../notebooks/sat6_cnn/hardware_onnxs/sat6-cnn-t{topology}w{quant}.onnx"
            estimates_output_dir = f"{build_dir}/{hw_name}"

            # Delete previous run results if they exist
            if os.path.exists(estimates_output_dir):
                shutil.rmtree(estimates_output_dir)
                print("Previous run results deleted!")

            cfg_estimates = build.DataflowBuildConfig(
                output_dir=estimates_output_dir,
                folding_config_file=folding_file,
                target_fps=target_fps,
                synth_clk_period_ns=10,
                stitched_ip_gen_dcp = True,
                #verbose = True,
                board="Pynq-Z1",
                shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
                generate_outputs=[
                    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                    build_cfg.DataflowOutputType.STITCHED_IP,
                    build_cfg.DataflowOutputType.OOC_SYNTH,
                ],
                steps=steps,
            )

            build.build_dataflow_cfg(model_file, cfg_estimates)

        cnv = topology_class(bit_quantization=quant)
        make_onnx(build_dir, cnv, quant, topology=topology)
        generate_hardware(
            build_dir=build_dir,
            target_fps=target_fps,
            quant=quant,
            topology=topology,
            steps=steps,
            folding_file=folding_file,
            run=run,
            hw_name=hw_name
    )


