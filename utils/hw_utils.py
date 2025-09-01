# --- Bibliotecas Padrão do Python ---
import csv
import json
import os
import pickle
import re
import shutil
from datetime import datetime

# --- Bibliotecas de Terceiros ---
import matplotlib.pyplot as plt
import onnx
import pandas as pd
import torch
from onnx import helper

# --- Módulos Específicos do Projeto (Brevitas, QONNX, FINN) ---

# Brevitas
from brevitas.export import export_qonnx

# QONNX (Utilitários e Core)
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.cleanup import cleanup as qonnx_cleanup

# QONNX (Transformações Gerais)
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

# FINN (Builder)
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

# FINN (Transformações)
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)

class utils():
    def __init__(self):
        super(utils, self).__init__()
        
    @staticmethod    
    def save_object(filename, object):
        with open(filename, 'wb') as outp:
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def read_object(filename):
        with open(filename, 'rb') as inp:
            return pickle.load(inp)

    @staticmethod
    def get_model_output_filename (topology,quant):
        return f"./pytorch_models/sat6-cnn-t{topology}w{quant}.pt"

    @staticmethod
    def get_hardware_config_name(topology, quant, target_fps, extra=''):
        fps_part = f"_{target_fps}fps" if target_fps is not None else ""
        return f"t{topology}w{quant}{fps_part}{extra}"

    @staticmethod
    def save_csv_table(results,csv_pathname):
        df = pd.DataFrame(results)
        print(df.to_string(header=None, index=False))
        df.to_csv(csv_pathname) 
        print(f"succesfully saved at {csv_pathname}")

    @staticmethod
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

    @staticmethod   
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

    @staticmethod
    def read_build_log(build_dir):
        log_path = os.path.join(build_dir, "build_dataflow.log")
        if not os.path.exists(log_path):
            return ""
        with open(log_path, "r") as f:
            return f.read()

    @staticmethod
    def detect_synthesis_error(log_text, timeout_keywords=None):
        if timeout_keywords is None:
            timeout_keywords = ["ERROR", "failed", "timeout", "synthesis failed", "crash"]

        for line in log_text.splitlines():
            if any(keyword.lower() in line.lower() for keyword in timeout_keywords):
                return True
        return False

    @staticmethod
    def read_folding_config(build_dir):
        fold_path = os.path.join(build_dir, "auto_folding_config.json")
        if not os.path.exists(fold_path):
            fold_path = os.path.join(build_dir, "final_hw_config.json")
            if not os.path.exists(fold_path):
                return {}
            
        with open(fold_path, "r") as f:
            return json.load(f)

    @staticmethod
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
    
    @staticmethod
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

    @staticmethod
    def dict_diff(prev, curr):
        changes = {}
        for layer in curr:
            if layer not in prev or prev[layer] != curr[layer]:
                changes[layer] = {"from": prev.get(layer), "to": curr[layer]}
        return changes
    
    @staticmethod
    def append_run_summary(file_path, hw_name, status, folding_config, duration, build_dir, resource_limits):
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
            "resource_limits": json.dumps(resource_limits)
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
            "folding_summary", "folding_diff", "build_dir", "resource_limits",
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

    @staticmethod
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

    @staticmethod
    def modify_folding_naive(folding, onnx_path, estimate_layer_cycles,
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
                    f["depthwise"] = int(attr.get("depthwise", 0))
                if op in ["FMPadding_rtl", "FMPadding_hls"]:
                    f["NumChannels"] = int(attr.get("NumChannels", 0))
                feats[name] = f
            return feats

        def next_divisor(n, current):
            if n is None or current is None:
                return None
            for d in range(current + 1, n + 1):
                if n % d == 0:
                    return d
            return None

        feature_dims = get_layer_features(onnx_path)
        new_folding = {}
        layers_modified = False

        for layer, cfg in folding.items():
            if layer == "Defaults":
                new_folding[layer] = {k: cfg[k] for k in ("PE", "SIMD") if k in cfg}
                continue

            new_cfg = {k: cfg[k] for k in ("PE", "SIMD", "parallel_window") if k in cfg}
            f = feature_dims.get(layer, {})
            modified = False

            op = f.get("op_type", "")
            
            # Primeiro avança SIMD
            if "SIMD" in new_cfg:
                dim = f.get("MW") or f.get("IFMChannels") or f.get("NumChannels")
                simd0 = new_cfg["SIMD"]
                nxt = next_divisor(dim, simd0)
                if nxt is not None and (not op.startswith("MVAU") or (nxt * f.get("WBits",1)) <= new_cfg.get("PE",1) * mvau_wwidth_max):
                    new_cfg["SIMD"] = nxt
                    modified = True

            # Depois avança PE
            if "PE" in new_cfg:
                mh = f.get("MH")
                pe0 = new_cfg["PE"]
                nxt_pe = next_divisor(mh, pe0)
                if nxt_pe is not None and nxt_pe <= max_pe:
                    new_cfg["PE"] = nxt_pe
                    modified = True

            if modified:
                layers_modified = True
                print(f"[↑] {layer}: PE {cfg.get('PE')}→{new_cfg.get('PE')}, SIMD {cfg.get('SIMD')}→{new_cfg.get('SIMD')}")

            new_folding[layer] = new_cfg

        return folding if not layers_modified else new_folding

    @staticmethod
    def modify_folding_greedy(folding, onnx_path, estimate_layer_cycles, only_pe=False, only_simd=False):
        def get_layer_features(onnx_path):
            model = onnx.load(onnx_path)
            features = {}

            for node in model.graph.node:
                layer_name = node.name
                op_type = node.op_type
                attr_dict = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}

                feat = {}
                if op_type in ["MVAU_hls", "MVAU_rtl"]:
                    feat["MW"] = int(attr_dict.get("MW", 0))
                    feat["MH"] = int(attr_dict.get("MH", 0))
                elif op_type in ["FMPadding_rtl", "FMPadding_hls"]:
                    feat["NumChannels"] = int(attr_dict.get("NumChannels", 0))
                elif op_type in ["ConvolutionInputGenerator_rtl", "ConvolutionInputGenerator_hls"]:
                    feat["IFMChannels"] = int(attr_dict.get("IFMChannels", 0))
                    feat["depthwise"] = int(attr_dict.get("depthwise", 0))
                    feat["op_type"] = op_type
                if feat:
                    features[layer_name] = feat

            return features

        feature_dims = get_layer_features(onnx_path)
        new_folding = {}
        layers_modified = False

        max_latency = max(estimate_layer_cycles.values())
        critical_layers = [k for k, v in estimate_layer_cycles.items() if v == max_latency]
            
        for layer in folding:
            if layer == "Defaults":
                new_folding[layer] = {k: v for k, v in folding[layer].items() if k in ["PE", "SIMD"]}
                continue

            cfg = folding[layer]
            new_cfg = {k: v for k, v in cfg.items() if k in ["PE", "SIMD", "parallel_window"]}
            layer_features = feature_dims.get(layer, {})
            modified = False

            if layer in critical_layers:
                op_type = layer_features.get("op_type", "")
                if op_type.startswith("ConvolutionInputGenerator") and layer_features.get("depthwise", 0) == 0:
                    if op_type == "ConvolutionInputGenerator_rtl":
                        ifm_channels = layer_features.get("IFMChannels", 0)
                        current_simd = new_cfg.get("SIMD", 1)
                        if current_simd == ifm_channels:
                            new_cfg["parallel_window"] = 1
                            print(f"[→] Ativado parallel_window em {layer}")
                            modified = True

                # Tentar modificar SIMD
                if not only_pe and "SIMD" in new_cfg:
                    mw = layer_features.get("MW") or layer_features.get("IFMChannels") or layer_features.get("NumChannels")
                    if mw:
                        new_simd = new_cfg["SIMD"] * 2
                        while new_simd > 1 and mw % new_simd != 0:
                            new_simd -= 1
                        if new_simd > new_cfg["SIMD"] and mw % new_simd == 0:
                            new_cfg["SIMD"] = new_simd
                            modified = True

                # Tentar modificar PE
                if not only_simd and "PE" in new_cfg and "MH" in layer_features:
                    mh = layer_features["MH"]
                    new_pe = new_cfg["PE"] * 2
                    while new_pe > 1 and mh % new_pe != 0:
                        new_pe -= 1
                    if new_pe > new_cfg["PE"] and mh % new_pe == 0:
                        new_cfg["PE"] = new_pe
                        modified = True

                if modified:
                    layers_modified = True
                    print(f"[↑] Modificado: {layer} - PE: {cfg.get('PE')}→{new_cfg.get('PE')}, SIMD: {cfg.get('SIMD')}→{new_cfg.get('SIMD')}")
                else:
                    print(f"[=] Não modificado: {layer} - PE: {cfg.get('PE')}, SIMD: {cfg.get('SIMD')}")
                    
            new_folding[layer] = new_cfg

        return folding if not layers_modified else new_folding

    @staticmethod
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

    @staticmethod
    def reset_folding(folding, onnx_path):
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
                feats[name] = f
            return feats

        def min_valid_simd(mw):
            simd = max(1, -(-mw // 1024))  # ceil(mw / 1024)
            for d in range(simd, mw + 1):
                if mw % d == 0:
                    return d
            return mw

        feature_dims = get_layer_features(onnx_path)
        new_folding = {}

        for layer, cfg in folding.items():
            if layer == "Defaults":
                new_folding[layer] = dict(cfg)
                continue

            new_cfg = {}
            f = feature_dims.get(layer, {})
            op = f.get("op_type", "")
            mw = f.get("MW", 0)

            for key in cfg:
                if key == "PE":
                    new_cfg["PE"] = 1
                elif key == "SIMD":
                    if op.startswith("MVAU") and mw > 0:
                        new_cfg["SIMD"] = min_valid_simd(mw)
                    else:
                        new_cfg["SIMD"] = 1
                else:
                    new_cfg[key] = cfg[key]

            new_folding[layer] = new_cfg

        return new_folding

    @staticmethod
    def get_exceeded_resources_flags(resource_diffs):
        return {
            "lut_exceed": resource_diffs.get("Total LUTs", 0) < 0,
            "ff_exceed": resource_diffs.get("FFs", 0) < 0,
            "bram_exceed": resource_diffs.get("BRAM (36k)", 0) < 0,
            "dsp_exceed": resource_diffs.get("DSP Blocks", 0) < 0,
        }

    @staticmethod
    def check_resource_usage(area_data, limits):
        if area_data is None:
            print("[!] Nenhum dado de área encontrado.")
            return {}
        diffs = {}
        for res, max_val in limits.items():
            used = area_data.get(res, 0)
            diffs[res] = max_val - used
        return diffs

    @staticmethod
    def get_exceeded_resources_flags(resource_diffs):
        return {
            "lut_exceed": resource_diffs.get("Total LUTs", 0) < 0,
            "ff_exceed": resource_diffs.get("FFs", 0) < 0,
            "bram_exceed": resource_diffs.get("BRAM (36k)", 0) < 0,
            "dsp_exceed": resource_diffs.get("DSP Blocks", 0) < 0,
        }

    @staticmethod
    def raise_if_exceeds_limits(resource_diffs):
        exceeded = {res: -diff for res, diff in resource_diffs.items() if diff < 0}
        if exceeded:
            msg_lines = ["[🚫] Recursos excedidos:"]
            for res, amount in exceeded.items():
                msg_lines.append(f"  - {res}: excedido por {amount}")
            return exceeded
        return None

    @staticmethod
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
    
    @staticmethod
    def build_hardware(model_path, build_dir, hw_name, steps, folding_file=None, target_fps=None, **kwargs):
        """
        Executa o fluxo de build de hardware do FINN para um modelo ONNX já processado.

        Esta função recebe um caminho para um modelo .onnx e executa as etapas de
        síntese de hardware definidas, usando a configuração especificada.

        Args:
            model_path (str): Caminho para o arquivo .onnx pronto para o FINN.
            build_dir (str): Diretório base onde os resultados do build serão salvos.
            hw_name (str): Nome específico para esta execução de build, usado para criar um subdiretório.
            steps (list): Lista de etapas do build do FINN a serem executadas.
            folding_file (str, optional): Caminho para o arquivo de configuração de folding (.json).
            target_fps (int, optional): FPS alvo para o otimizador do FINN.
            **kwargs: Aceita outros argumentos (como topology, quant, run) para compatibilidade
                    de chamada, mas eles não são usados diretamente aqui.
        """
        
        # A função interna `generate_hardware` foi mantida para encapsular a lógica do FINN.
        def generate_hardware(model_file_local, build_dir_local, target_fps_local, steps_local, folding_file_local, hw_name_local):
            """Configura e executa o build do FINN."""
            
            # Define o diretório de saída específico para este build
            output_dir_for_this_run = os.path.join(build_dir_local, hw_name_local)

            # Remove resultados de uma execução anterior com o mesmo nome para evitar conflitos
            if os.path.exists(output_dir_for_this_run):
                shutil.rmtree(output_dir_for_this_run)
                print(f"Diretório de build anterior removido: {output_dir_for_this_run}")

            # Configuração completa do DataflowBuildConfig, como na sua versão original.
            # Estes parâmetros definem como o FINN deve se comportar.
            cfg_estimates = build.DataflowBuildConfig(
                output_dir                  = output_dir_for_this_run,
                folding_config_file         = folding_file_local,
                target_fps                  = target_fps_local,
                synth_clk_period_ns         = 10.0,
                board                       = "Pynq-Z1",
                shell_flow_type             = build_cfg.ShellFlowType.VIVADO_ZYNQ,
                generate_outputs            = [
                    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                    build_cfg.DataflowOutputType.STITCHED_IP,
                    build_cfg.DataflowOutputType.OOC_SYNTH,
                    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                ],
                steps                       = steps_local,
                # Parâmetros adicionais da sua implementação original
                stitched_ip_gen_dcp         = True,
                rtlsim_batch_size           = 1000,
                # verbose                   = True, # Descomente para logs mais detalhados
            )

            # Ponto de entrada para a API do FINN que inicia o processo de build
            build.build_dataflow_cfg(model_file_local, cfg_estimates)

        # A chamada para make_onnx foi removida.
        # A função agora simplesmente chama a sub-função generate_hardware
        # com os parâmetros recebidos.
        generate_hardware(
            model_file_local=model_path,
            build_dir_local=build_dir,
            target_fps_local=target_fps,
            steps_local=steps,
            folding_file_local=folding_file,
            hw_name_local=hw_name
        )
        
def export_to_onnx(model, build_dir, topology_id, quant):
    """ 
    Exporta um modelo PyTorch/Brevitas treinado para o formato QONNX e aplica
    a sequência completa de transformações do FINN para prepará-lo para a 
    síntese de hardware.
    """
    print("-> Exportando modelo para ONNX com pipeline completo de transformações...")
    
    model_build_dir = os.path.join(build_dir, f"t{topology_id}w{quant}_model_files")
    os.makedirs(model_build_dir, exist_ok=True)
    
    temp_export_path = os.path.join(model_build_dir, f"t{topology_id}w{quant}_temp_exported.onnx")
    parent_model_path = os.path.join(model_build_dir, f"t{topology_id}w{quant}_dataflow_parent.onnx")
    final_dataflow_model_path = os.path.join(model_build_dir, f"t{topology_id}w{quant}_finn_ready.onnx")

    # --- Etapa 1: Exportação e Limpeza Inicial ---
    export_qonnx(model, torch.randn(1, 4, 32, 32), export_path=temp_export_path)
    qonnx_cleanup(temp_export_path, out_file=temp_export_path)
    model = ModelWrapper(temp_export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    
    # --- Etapa 2: Pré-processamento e Pós-processamento ---
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))
    # Tidy-up após adicionar nós
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    # --- Etapa 3: Streamlining e Otimizações de Alto Nível ---
    # Esta etapa prepara o grafo para um fluxo de dados contínuo (streaming)
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    # --- Etapa 4: Conversão para Camadas de Hardware (HLS) ---
    # Esta é a etapa crucial onde nós de alto nível (como MatMul) são
    # convertidos em nós customizados do FINN que representam blocos de hardware.
    model = model.transform(to_hls.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hls.InferLabelSelectLayer())
    model = model.transform(to_hls.InferThresholdingLayer())
    model = model.transform(to_hls.InferConvInpGen())
    model = model.transform(to_hls.InferStreamingMaxPool())
    
    # --- Etapa 5: Limpeza Final e Particionamento do Dataflow ---
    # Limpezas finais no grafo já com nós de HLS
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    
    # Separa o grafo em uma parte que será implementada em hardware (dataflow)
    # e uma parte que permanece em software (se houver).
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(parent_model_path)
    
    # Extrai e salva apenas a parte de dataflow, que é o que o HardwareExplorer usará.
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename_in_build_dir = sdp_node.get_nodeattr("model")
    
    # Copia o modelo de dataflow para o nosso caminho final e mais legível
    shutil.copy(dataflow_model_filename_in_build_dir, final_dataflow_model_path)

    # Remove o arquivo temporário inicial
    os.remove(temp_export_path)

    print(f"[✓] Modelo ONNX pronto para o FINN salvo em: {final_dataflow_model_path}")
    return final_dataflow_model_path        