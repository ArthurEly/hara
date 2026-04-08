# --- Bibliotecas Padrão do Python ---
import copy
import csv
import json
import os
import pickle
import re
import shutil
from datetime import datetime
import math

# --- Bibliotecas de Terceiros ---
import matplotlib.pyplot as plt
import onnx
import pandas as pd
import torch
from onnx import helper

# --- Módulos Específicos do Projeto (Brevitas, QONNX, FINN) ---
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.cleanup import cleanup as qonnx_cleanup
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
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.test import get_test_model_trained
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
import yaml
from utils.ml_utils import load_pruned_model
from brevitas_examples.bnn_pynq.models import model_with_cfg
from brevitas_examples.bnn_pynq.extractor import ModelExtractor
import importlib.util
from brevitas_examples.bnn_pynq.models.cnv_common import FlexibleCNV
from brevitas_examples.bnn_pynq.models.FC import FlexibleFC

from qonnx.transformation.merge_onnx_models import MergeONNXModels
from finn.util.pytorch import ToTensor

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
        return f"{topology}w{quant}{fps_part}{extra}"

    @staticmethod
    def save_csv_table(results,csv_pathname):
        df = pd.DataFrame(results)
        print(df.to_string(header=None, index=False))
        df.to_csv(csv_pathname) 
        print(f"succesfully saved at {csv_pathname}")

    @staticmethod
    def clean_build_artifacts(build_dir):
        """
        Remove diretórios pesados de uma build do FINN para economizar espaço,
        mantendo apenas logs, relatórios JSON e configurações.
        """
        # Lista de pastas/arquivos que costumam ser grandes e descartáveis após a extração de dados
        trash_list = [
            "intermediate_models",
            "output_ip", 
            "stitched_ip",
            "vivado_ip_cache",
            "vivado_zynq_proj",
            "bitfile", 
            "driver",
            "pyverilator_ipstitched",
        ]

        print(f"   -> 🧹 Limpando artefatos pesados em: {build_dir}")
        for item in trash_list:
            path = os.path.join(build_dir, item)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
            except Exception as e:
                # É normal falhar se o arquivo não existir
                pass

    @staticmethod
    def move_intermediate_outputs_dir(src, dst):
        source_folder = src
        destination_folder = dst
        
        if os.path.exists(destination_folder):
            shutil.rmtree(destination_folder)
            
        for file_name in os.listdir(source_folder):
            source = source_folder + file_name
            destination = destination_folder + file_name
            if "pyverilator_ipstitched" in file_name:
                shutil.rmtree(source)
            else:
                try:
                    shutil.move(source, destination)
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
    def map_resources_to_components(folding_config_path, utilization_report_path):
        if not os.path.exists(folding_config_path) or not os.path.exists(utilization_report_path):
            print("Erro: Arquivo de configuração ou de relatório não encontrado.")
            return None

        with open(folding_config_path, 'r') as f:
            folding_config = json.load(f)
        
        component_names = [key for key in folding_config.keys() if key != "Defaults"]

        with open(utilization_report_path, 'r') as f:
            report_lines = f.readlines()

        resource_map = {}
        for line in report_lines:
            if not line.strip().startswith('|'):
                continue

            match = re.search(r"\|\s*(\S+)\s*\|\s*\S+.*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", line)
            
            if not match:
                continue
                
            instance_name = match.group(1)
            
            for component in component_names:
                if instance_name.startswith(component):
                    try:
                        total_luts = int(match.group(2))
                        logic_luts = int(match.group(3))
                        lutrams = int(match.group(4))
                        srls = int(match.group(5))
                        ffs = int(match.group(6))
                        ramb36 = int(match.group(7))
                        ramb18 = int(match.group(8))
                        dsp = int(match.group(9))
                        bram_36k = ramb36 + ramb18 / 2.0

                        if component not in resource_map:
                            resource_map[component] = {
                                "Total LUTs": total_luts,
                                "Logic LUTs": logic_luts,
                                "LUTRAMs": lutrams,
                                "SRLs": srls,
                                "FFs": ffs,
                                "BRAM (36k)": round(bram_36k, 1),
                                "RAMB18": ramb18,
                                "RAMB36": ramb36,
                                "DSP Blocks": dsp
                            }
                    except (IndexError, ValueError):
                        continue

        return resource_map
    
    @staticmethod
    def attempt_resource_tradeoff(current_folding, failed_build_dir, resource_limits):
        report_path = None
        for root, _, files in os.walk(failed_build_dir):
            for file in files:
                if file.endswith('finn_design_partition_util.rpt'):
                    report_path = os.path.join(root, file)
                    break
            if report_path:
                break
        
        if not report_path:
            print("  -> [!] Relatório de utilização do build que falhou não encontrado.")
            return current_folding, False

        total_usage = utils.extract_area_from_rpt(failed_build_dir)
        if not total_usage:
             print("  -> [!] Não foi possível extrair a utilização total de recursos do relatório.")
             return current_folding, False
             
        resource_diffs = utils.check_resource_usage(total_usage, resource_limits)
        exceeded_resources = [res for res, diff in resource_diffs.items() if diff < 0]

        if "BRAM (36k)" in exceeded_resources:
            print(f"  -> [!] Recurso BRAM excedido. Analisando componentes para trade-off.")

            final_hw_config_path = os.path.join(failed_build_dir, "final_hw_config.json")
            if not os.path.exists(final_hw_config_path):
                 print(f"  -> [!] final_hw_config.json não encontrado em {failed_build_dir}")
                 return current_folding, False

            component_resources = utils.map_resources_to_components(
                final_hw_config_path,
                report_path
            )

            if not component_resources:
                print("  -> [!] Não foi possível mapear recursos por componente.")
                return current_folding, False

            sorted_by_bram = sorted(
                component_resources.items(), 
                key=lambda item: item[1].get('BRAM (36k)', 0), 
                reverse=True
            )

            modified_folding = copy.deepcopy(current_folding)
            was_modified = False

            for component_name, resources in sorted_by_bram:
                if resources.get('BRAM (36k)', 0) > 0 and component_name in modified_folding:
                    component_config = modified_folding[component_name]

                    if component_config.get('ram_style') != 'distributed':
                        print(f"  -> Modificando '{component_name}' (consumidor de BRAM) para usar LUTRAM.")
                        component_config['ram_style'] = 'distributed'
                        was_modified = True
                        break
                    else:
                        print(f"  -> Ignorando '{component_name}', pois já está configurado para 'distributed'.")
            
            if not was_modified:
                print("  -> Nenhum outro componente pôde ser modificado para o trade-off BRAM -> LUTRAM.")

            return modified_folding, was_modified

        else:
            print(f"  -> [!] Falha por outros recursos ({exceeded_resources}). Nenhuma ação de trade-off implementada.")
            return current_folding, False
    
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

        area_data = utils.extract_area_from_rpt(build_dir)
        if area_data:
            summary.update(area_data)

        perf_path = os.path.join(build_dir, "report", "estimate_network_performance.json")
        if os.path.isfile(perf_path):
            try:
                with open(perf_path, 'r') as f:
                    perf_data = json.load(f)
                    summary["estimated_throughput_fps"] = perf_data.get("estimated_throughput_fps", None)
                    summary["max_cycles_node_name"] = perf_data.get("max_cycles_node_name", None)
            except Exception as e:
                print(f"[!] Erro ao ler estimated throughput: {e}")

        field_order = [
            "date", "hw_name", "status", "duration_in_seconds",
            "folding_summary", "folding_diff", "build_dir", "resource_limits",
            "Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs", "FFs", "BRAM (36k)", "DSP Blocks",
            "estimated_throughput_fps", "max_cycles_node_name"
        ]

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
            
            new_cfg = copy.deepcopy(cfg)
            f = feature_dims.get(layer, {})
            modified = False

            op = f.get("op_type", "")
            
            if "SIMD" in new_cfg:
                dim = f.get("MW") or f.get("IFMChannels") or f.get("NumChannels")
                simd0 = new_cfg["SIMD"]
                nxt = next_divisor(dim, simd0)
                if nxt is not None and (not op.startswith("MVAU") or (nxt * f.get("WBits",1)) <= new_cfg.get("PE",1) * mvau_wwidth_max):
                    new_cfg["SIMD"] = nxt
                    modified = True

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

        max_latency = max(estimate_layer_cycles.values()) if estimate_layer_cycles else 0
        critical_layers = [k for k, v in estimate_layer_cycles.items() if v == max_latency]
            
        for layer in folding:
            if layer == "Defaults":
                new_folding[layer] = {k: v for k, v in folding[layer].items() if k in ["PE", "SIMD"]}
                continue

            cfg = folding[layer]
            new_cfg = copy.deepcopy(cfg)

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

                if not only_pe and "SIMD" in new_cfg:
                    mw = layer_features.get("MW") or layer_features.get("IFMChannels") or layer_features.get("NumChannels")
                    if mw:
                        new_simd = new_cfg["SIMD"] * 2
                        while new_simd > 1 and mw % new_simd != 0:
                            new_simd -= 1
                        if new_simd > new_cfg["SIMD"] and mw % new_simd == 0:
                            new_cfg["SIMD"] = new_simd
                            modified = True

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
        
        def get_layer_features(path):
            model = onnx.load(path)
            
            # --- NOVO: Mapeamento de Shapes dos Tensores ---
            # Precisamos saber o shape real da entrada para não estourar o SIMD
            tensor_shapes = {}
            # Adiciona inputs do grafo (ex: imagem de entrada)
            for i in model.graph.input:
                dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
                tensor_shapes[i.name] = dims
            # Adiciona value_infos (tensores intermediários)
            for vi in model.graph.value_info:
                dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                tensor_shapes[vi.name] = dims
            # -----------------------------------------------

            feats = {}
            for node in model.graph.node:
                name = node.name
                op = node.op_type
                attr = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                f = {"op_type": op}
                
                # Variável para armazenar o limite físico do SIMD detectado
                simd_limit = None

                if op.startswith("MatrixVectorActivation") or op.startswith("MVAU"):
                    raw_mw = int(attr.get("MW", 0))
                    f["MW"] = raw_mw
                    f["MH"] = int(attr.get("MH", 0))
                    
                    # --- Lógica de Proteção de Dimensão ---
                    # Tenta descobrir o tamanho real do canal de entrada olhando o tensor input[0]
                    if len(node.input) > 0:
                        input_name = node.input[0]
                        if input_name in tensor_shapes:
                            shape = tensor_shapes[input_name]
                            # Pega a última dimensão (que no layout do FINN stream é o canal/folding)
                            if shape and len(shape) > 0:
                                last_dim = shape[-1]
                                if last_dim > 0:
                                    simd_limit = last_dim
                    
                    # Se não conseguiu detectar pelo tensor, usa o MW como fallback
                    if simd_limit is None: 
                        simd_limit = raw_mw

                elif op.startswith("ConvolutionInputGenerator") or op.startswith("Downsampler"):
                    val = int(attr.get("IFMChannels", 0))
                    f["MW"] = val
                    simd_limit = val # Limite é o número de canais de entrada
                    
                elif op.startswith("FMPadding"):
                    val = int(attr.get("NumChannels", 0))
                    f["MW"] = val
                    simd_limit = val
                    
                elif op.startswith("Thresholding") or op.startswith("StreamingMaxPool") or op.startswith("StreamingEltwise") or op.startswith("AddStreams") or op.startswith("ChannelwiseOp") or op.startswith("DuplicateStreams") or op.startswith("Globalaccpool"):
                    f["MH"] = int(attr.get("NumChannels", 0))
                    
                elif op.startswith("LabelSelect"):
                    f["MH"] = int(attr.get("Labels", 0))
                    
                elif op.startswith("VectorVectorActivation") or op.startswith("VVAU"):
                    k_dims = attr.get("Kernel", [1, 1])
                    f["MW"] = k_dims[0] * k_dims[1]
                    f["MH"] = int(attr.get("NumChannels", 0))

                # Salva o limite calculado para ser usado no modify_folding
                if simd_limit is not None:
                    f["SIMD_LIMIT"] = simd_limit

                feats[name] = f
            return feats

        def next_divisor(n, current):
            if n is None or current is None or n == 0: return None
            if current >= n: return None
            for d in range(current + 1, n + 1):
                if n % d == 0: return d
            return None

        feature_dims = get_layer_features(onnx_path)
        new_folding = {}
        layers_modified = False
        max_lat = max(estimate_layer_cycles.values()) if estimate_layer_cycles else 0
        critical = {k for k, v in estimate_layer_cycles.items() if v == max_lat}

        print(f"\n[DEBUG] Latência máxima (gargalo): {max_lat} ciclos")
        print(f"[DEBUG] Camada(s) crítica(s) identificada(s): {critical}")

        for layer, cfg in folding.items():
            if layer == "Defaults":
                new_folding[layer] = {k: cfg[k] for k in ("PE", "SIMD") if k in cfg}
                continue
            new_cfg = copy.deepcopy(cfg)
            f = feature_dims.get(layer, {})
            modified = False
            
            if layer in critical:
                print(f"\n[DEBUG] --- Processando camada crítica: {layer} ---")
                
                if not only_pe and "SIMD" in new_cfg:
                    dim_simd = f.get("SIMD_LIMIT", f.get("MW"))
                    simd0 = new_cfg.get("SIMD")
                    print(f"[DEBUG]     -> Tentando otimizar SIMD: Dimensão (MW) = {dim_simd}, Valor Atual = {simd0}")
                    if dim_simd is not None and simd0 is not None:
                        nxt_simd = next_divisor(dim_simd, simd0)
                        print(f"[DEBUG]       -> Próximo divisor para SIMD: {nxt_simd}")

                        if nxt_simd is None and new_cfg.get("parallel_window", 0) == 0 and ("ConvolutionInputGenerator" in f.get("op_type", "")):
                            print(f"[DEBUG]       -> SIMD já está no máximo para {layer}. Ativando parallel_window.")
                            new_cfg["parallel_window"] = 1
                            modified = True
                        elif nxt_simd is not None:
                            new_cfg["SIMD"] = nxt_simd
                            modified = True

                if not modified and not only_simd and "PE" in new_cfg:
                    dim_pe = f.get("MH")
                    pe0 = new_cfg.get("PE")
                    print(f"[DEBUG]     -> Tentando otimizar PE: Dimensão (MH) = {dim_pe}, Valor Atual = {pe0}")
                    if dim_pe is not None and pe0 is not None:
                        nxt_pe = next_divisor(dim_pe, pe0)
                        print(f"[DEBUG]       -> Próximo divisor para PE: {nxt_pe}")
                        if nxt_pe is not None and nxt_pe <= max_pe:
                            new_cfg["PE"] = nxt_pe
                            modified = True
            
            if modified:
                layers_modified = True
                pe_old, pe_new = cfg.get('PE'), new_cfg.get('PE')
                simd_old, simd_new = cfg.get('SIMD'), new_cfg.get('SIMD')
                pw_old, pw_new = cfg.get('parallel_window'), new_cfg.get('parallel_window')
                
                log_msg = f"[↑] {layer}:"
                if pe_new != pe_old:
                    log_msg += f" PE {pe_old}→{pe_new}"
                if simd_new != simd_old:
                    log_msg += f" SIMD {simd_old}→{simd_new}"
                if pw_new != pw_old:
                    log_msg += f" parallel_window {pw_old}→{pw_new}"
                print(log_msg)
                
            new_folding[layer] = new_cfg
            
        return folding if not layers_modified else new_folding

    @staticmethod
    def reset_folding(folding, onnx_path, fixed_resources=None):
        def _get_hw_layer_features(path):
            import onnx
            from onnx import helper
            model = onnx.load(path)
            feats = {}
            for node in model.graph.node:
                if node.op_type.startswith("MatrixVectorActivation") or node.op_type.startswith("VectorVectorActivation") or node.op_type.startswith("MVAU"):
                    attr = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                    mw = int(attr.get("MW", 0))
                    mh = int(attr.get("MH", 0))
                    feats[node.name] = {"MW": mw, "MH": mh}
            return feats

        def _get_min_valid_simd(mw):
            if mw is None or mw <= 0: return 1
            min_req_simd = math.ceil(mw / 1024.0)
            for d in range(int(min_req_simd), mw + 1):
                if mw % d == 0: return d
            return mw

        layer_features = _get_hw_layer_features(onnx_path)
        new_folding = copy.deepcopy(folding)

        for node_name, config in new_folding.items():
            if "PE" in config: config["PE"] = 1
            if "SIMD" in config: config["SIMD"] = 1
            if "MVAU" in node_name and node_name in layer_features:
                mw = layer_features[node_name].get("MW")
                if "SIMD" in config and mw is not None:
                    config["SIMD"] = _get_min_valid_simd(mw)
        
        if fixed_resources and isinstance(fixed_resources, dict):
            print(f"[i] Aplicando configurações de recursos fixos: {fixed_resources}")
            for node_name, config in new_folding.items():
                if node_name == "Defaults": continue
                for op_type_key, resource_config in fixed_resources.items():
                    if op_type_key in node_name:
                        print(f"    -> Aplicando {resource_config} em {node_name}")
                        config.update(resource_config)
                        
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
    def raise_if_exceeds_limits(resource_diffs):
        exceeded = {res: -diff for res, diff in resource_diffs.items() if diff < 0}
        if exceeded:
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
                def monitor_output():
                    for line in proc.stdout:
                        output_log.write(line)
                        log_file.write(line)
                        log_file.flush()

                t = threading.Thread(target=monitor_output)
                t.start()

                t.join(timeout=timeout_sec)

                if t.is_alive():
                    proc.kill()
                    t.join() 
                    log_file.write("\n⏱️ Build travado ou demorando demais\n")
                    log_file.flush()
                    raise RuntimeError("Build travado ou demorando demais")

        t.join()

        full_output = output_log.getvalue()

        if "Traceback" in full_output or "ValueError" in full_output:
            with open(log_path, "a") as log_file:
                log_file.write("\n❌ Erro detectado no build\n")
            raise RuntimeError("Erro detectado no build")

        return full_output
    
    @staticmethod
    def build_hardware(model_path, build_dir, hw_name, fpga_part, steps, folding_file=None, target_fps=None, **kwargs):
        def generate_hardware(model_file_local, build_dir_local, target_fps_local, steps_local, folding_file_local, hw_name_local):
            output_dir_for_this_run = os.path.join(build_dir_local, hw_name_local)

            if os.path.exists(output_dir_for_this_run):
                shutil.rmtree(output_dir_for_this_run)

            cfg_estimates = build.DataflowBuildConfig(
                output_dir                  = output_dir_for_this_run,
                folding_config_file         = folding_file_local,
                target_fps                  = target_fps_local,
                synth_clk_period_ns         = 10.0,
                fpga_part                   = fpga_part,
                split_large_fifos           = True,
                shell_flow_type             = build_cfg.ShellFlowType.VIVADO_ZYNQ,
                generate_outputs            = [
                    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                    build_cfg.DataflowOutputType.STITCHED_IP,
                    build_cfg.DataflowOutputType.OOC_SYNTH,
                    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
                ],
                steps                       = steps_local,
                stitched_ip_gen_dcp         = True,
                rtlsim_batch_size           = 1000,
            )

            build.build_dataflow_cfg(model_file_local, cfg_estimates)

        generate_hardware(
            model_file_local=model_path,
            build_dir_local=build_dir,
            target_fps_local=target_fps,
            steps_local=steps,
            folding_file_local=folding_file,
            hw_name_local=hw_name
        )

def load_pruned_model_flexible(model_path, topology_id, w_quant, a_quant, in_bit_width=8):
    """
    Carrega um modelo prunado (FlexibleCNV ou FlexibleFC) lendo a configuração
    salva no arquivo _config.py adjacente ao .pth.
    """
    # 1. Localizar o arquivo de configuração (_config.py)
    config_path = model_path.replace(".pth", "_config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuração do modelo prunado não encontrada: {config_path}")

    # 2. Carregar a configuração dinamicamente como um módulo Python
    spec = importlib.util.spec_from_file_location("pruned_config", config_path)
    pruned_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruned_config)

    model = None
    
    # --- Lógica para CNV (Grape, Soybean, Cifar10) ---
    if "CNV" in topology_id:
        # Recupera parâmetros do config
        stage_configs = getattr(pruned_config, 'stage_configs', None)
        fc_features = getattr(pruned_config, 'fc_features', [])
        flatten_size = getattr(pruned_config, 'flatten_size', 512) 
        conv_padding = getattr(pruned_config, 'conv_padding', 1) 
        
        # --- Lógica melhorada para número de classes ---
        # 1. Tenta ler do arquivo _config.py (gerado pelo novo pruner)
        num_classes = getattr(pruned_config, 'num_classes', None)
        
        # 2. Se não existir no config, usa o fallback hardcoded
        if num_classes is None:
            num_classes = 10 
            if "GRAPE" in topology_id:
                num_classes = 4
            elif "SOYBEAN" in topology_id:
                num_classes = 3

        print(f"   -> Reconstruindo FlexibleCNV: Classes={num_classes}, Padding={conv_padding}, Flatten={flatten_size}")
        
        model = FlexibleCNV(
            num_classes=num_classes,
            weight_bit_width=w_quant,
            act_bit_width=a_quant,
            in_bit_width=in_bit_width,
            in_ch=3,
            stage_configs=stage_configs,
            fc_features=fc_features,
            flatten_size=flatten_size,
            conv_padding=conv_padding
        )

    # --- Lógica para FC (TFC / MNIST) ---
    elif "TFC" in topology_id or "FC" in topology_id:
        out_features_list = getattr(pruned_config, 'out_features_list', [])
        
        print(f"   -> Reconstruindo FlexibleFC: Layers={out_features_list}")

        model = FlexibleFC(
            num_classes=10,
            weight_bit_width=w_quant,
            act_bit_width=a_quant,
            in_bit_width=in_bit_width, # TFC usa entrada binária
            in_channels=1,
            out_features_list=out_features_list,
            in_features_shape=(28, 28)
        )

    # 3. Carregar os pesos treinados
    if model:
        checkpoint = torch.load(model_path, map_location='cpu')
        # Verifica se é um checkpoint completo ou apenas state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        return model
    else:
        raise ValueError(f"Topologia {topology_id} não suportada para carregamento flexível.")
    
# --- Funções de Pipeline ---
def _transform_pipeline_t2(model, build_dir, topology_id, quant_str):
    print("-> Exportando modelo para ONNX com pipeline completo de transformações...")
    
    model_build_dir = os.path.join(build_dir, f"t{topology_id}_{quant_str}_model_files")
    os.makedirs(model_build_dir, exist_ok=True)

    temp_export_path = os.path.join(model_build_dir, f"t{topology_id}_{quant_str}_temp_exported.onnx")
    parent_model_path = os.path.join(model_build_dir, f"t{topology_id}_{quant_str}_dataflow_parent.onnx")
    final_dataflow_model_path = os.path.join(model_build_dir, f"t{topology_id}_{quant_str}_finn_ready.onnx")

    export_qonnx(model, torch.randn(1, 4, 32, 32), export_path=temp_export_path)
    qonnx_cleanup(temp_export_path, out_file=temp_export_path)
    model = ModelWrapper(temp_export_path)
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))
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
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    
    try:
        w_bits = int(quant_str.split('w')[0])
    except (ValueError, IndexError):
        raise ValueError(f"Formato de 'quant_str' inválido: '{quant_str}'. Esperado algo como '4w4a'.")

    if w_bits == 1:
        model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    else:
        model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())

    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferStreamingMaxPool())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(parent_model_path)
    
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename_in_build_dir = sdp_node.get_nodeattr("model")
    
    shutil.copy(dataflow_model_filename_in_build_dir, final_dataflow_model_path)

    os.remove(temp_export_path)

    return final_dataflow_model_path

def _transform_pipeline_cnv(model, build_dir, topology_id, quant_str):    
    print(f"-> Aplicando pipeline de transformação para Topologia CNV (ID: {topology_id})...")

    # Parsing das quantizações para nomes de arquivos (ex: 2w_2a)
    q_parts = quant_str.replace('a', '').replace('w', '').split('_')
    q_val = q_parts[0] if len(q_parts) > 0 else "unknown"
    a_val = q_parts[1] if len(q_parts) > 1 else q_val

    model_build_dir = os.path.join(build_dir, f"t{topology_id}w{quant_str}_model_files")
    os.makedirs(model_build_dir, exist_ok=True)

    temp_export_path = os.path.join(model_build_dir, f"temp_exported.onnx")
    final_dataflow_model_path = os.path.join(model_build_dir, f"t{topology_id}_{quant_str}_finn_ready.onnx")

    export_qonnx(model, torch.randn(1, 3, 32, 32), export_path=temp_export_path)
    qonnx_cleanup(temp_export_path, out_file=temp_export_path)
    model = ModelWrapper(temp_export_path)
    model = model.transform(ConvertQONNXtoFINN())
    
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    
    totensor_pyt = ToTensor()
    chkpt_preproc_name = os.path.join(model_build_dir, "preproc.onnx")
    export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
    qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
    
    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())                     

    model = model.transform(MergeONNXModels(pre_model))
    
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

    model = model.transform(InsertTopK(k=1))

    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    
    model = model.transform(to_hw.InferVectorVectorActivation())
    
    model.save(final_dataflow_model_path)
    
    # Limpeza de arquivo temporário
    if os.path.exists(temp_export_path):
        os.remove(temp_export_path)

    return final_dataflow_model_path

def _transform_pipeline_mnist(model, build_dir, topology_id, quant_str):
    print(f"-> Aplicando pipeline de transformação para Topologia TFC (ID: {topology_id})...")

    model_build_dir = os.path.join(build_dir, f"{topology_id}_{quant_str}_model_files")
    os.makedirs(model_build_dir, exist_ok=True)
    
    base_filename = f"{topology_id}_{quant_str}"
    initial_model_path = os.path.join(model_build_dir, f"{base_filename}_initial.onnx")
    final_model_path = os.path.join(model_build_dir, f"{base_filename}_finn_ready.onnx")

    export_qonnx(model, torch.randn(1, 1, 28, 28), initial_model_path)
    qonnx_cleanup(initial_model_path, out_file=initial_model_path)
    
    model = ModelWrapper(initial_model_path)
    
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    
    totensor_pyt = ToTensor()
    chkpt_preproc_name = os.path.join(model_build_dir, "preproc.onnx")
    export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
    qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
    
    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())                     

    model = model.transform(MergeONNXModels(pre_model))
    
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

    model = model.transform(InsertTopK(k=1))
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    
    w_bits, a_bits = [int(b.replace('w', '').replace('a', '')) for b in quant_str.split('_')]
    if w_bits == 1 and a_bits == 1:
        model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    else:
        model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
        
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(Streamline())
    model = model.transform(to_hw.InferThresholdingLayer())

    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = getCustomOp(parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0])
    dataflow_model_path = sdp_node.get_nodeattr("model")
    
    shutil.copy(dataflow_model_path, final_model_path)
    return final_model_path

def _transform_pipeline_cifar10(model, build_dir, topology_id, quant_str):
    print(f"-> Aplicando pipeline de transformação COMPLETO para Topologia CNV (ID: {topology_id})...")

    model_build_dir = os.path.join(build_dir, f"{topology_id}_{quant_str}_model_files")
    os.makedirs(model_build_dir, exist_ok=True)

    base_filename = f"{topology_id}_{quant_str}"
    initial_model_path = os.path.join(model_build_dir, f"{base_filename}_initial.onnx")
    final_model_path = os.path.join(model_build_dir, f"{base_filename}_finn_ready.onnx")
    
    export_qonnx(model, torch.randn(1, 3, 32, 32), initial_model_path)
    qonnx_cleanup(initial_model_path, out_file=initial_model_path)
    model = ModelWrapper(initial_model_path)
    
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())

    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))
    model.save(final_model_path)

    return final_model_path

def _transform_pipeline_mobilenet(model, build_dir, topology_id, quant_str):
    """
    Pipeline de transformação FINN para modelos MobileNet quantizados (Brevitas).

    MobileNet usa convoluções depthwise separáveis (VVAU no FINN) e
    global average pooling — requer InferVectorVectorActivation além do
    pipeline CNV padrão. Não usa conversão bipolar (pesos inteiros, não binários).

    Input esperado: (1, 3, 64, 64) — imagens RGB 64x64 (SAT6/agrícolas).
    Ajuste `input_shape` no registry_models.yaml se o modelo usar outra resolução.
    """
    print(f"-> Aplicando pipeline de transformação para MobileNet (ID: {topology_id})...")

    model_build_dir = os.path.join(build_dir, f"{topology_id}_{quant_str}_model_files")
    os.makedirs(model_build_dir, exist_ok=True)

    base_filename = f"{topology_id}_{quant_str}"
    initial_model_path = os.path.join(model_build_dir, f"{base_filename}_initial.onnx")
    final_model_path   = os.path.join(model_build_dir, f"{base_filename}_finn_ready.onnx")

    # Exporta com input shape 64x64 (padrão MobileNet no HARA)
    export_qonnx(model, torch.randn(1, 3, 64, 64), initial_model_path)
    qonnx_cleanup(initial_model_path, out_file=initial_model_path)
    model = ModelWrapper(initial_model_path)

    # --- Canonicalização inicial ---
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    # --- Pré-processamento (ToTensor) ---
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)

    totensor_pyt = ToTensor()
    chkpt_preproc_name = os.path.join(model_build_dir, "preproc.onnx")
    export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
    qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)

    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    model = model.transform(MergeONNXModels(pre_model))

    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))

    # --- Streamlining ---
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    # MobileNet usa pesos inteiros — sem conversão bipolar/xnor
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    # --- Mapeamento para camadas de HW ---
    # InferVectorVectorActivation: cobre convoluções depthwise (VVAU)
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(Streamline())
    model = model.transform(to_hw.InferThresholdingLayer())

    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = getCustomOp(
        parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    )
    dataflow_model_path = sdp_node.get_nodeattr("model")

    shutil.copy(dataflow_model_path, final_model_path)
    print(f"-> MobileNet FINN-ready salvo em: {final_model_path}")
    return final_model_path


def get_finn_ready_model(model_info, build_dir):
    topology_id = model_info["topology_id"]
    
    # Tratamento de quantização (suporta chaves antigas e novas do YAML)
    if "weight_quant" in model_info:
        w_quant = model_info["weight_quant"]
        a_quant = model_info["act_quant"]
    else:
        w_quant = model_info["quant"]
        a_quant = model_info["quant"]

    quant_str = f"{w_quant}w_{a_quant}a" 
    
    pytorch_model = None
    loader = model_info.get("loader")
    source = model_info.get("source")

    print(f"-> Carregando modelo '{topology_id}' (Loader: {loader}, Source: {source})")

    # === NOVO LOADER: hara_pruned ===
    if loader == "hara_pruned":
        model_path = model_info["model_path"]
        # Bit width de entrada: 1 para MNIST (TFC), 8 para imagens coloridas (CNV)
        in_bits = 2 if "TFC" in topology_id else 8
        pytorch_model = load_pruned_model_flexible(model_path, topology_id, w_quant, a_quant, in_bits)
    # ================================

    elif loader == "hara_internal":
        model_path = model_info["model_path"]
        pytorch_model = load_pruned_model(model_path)
    
    elif loader == "brevitas_example":
        model_name = model_info["model_name"]
        full_model_name = f"{model_name}_{w_quant}W{a_quant}A"

        if source == "pretrained":
            pytorch_model = get_test_model_trained(model_name, w_quant, a_quant)
        
        elif source == "local_checkpoint":
            checkpoint_path = model_info["checkpoint_path"]
            from brevitas_examples.bnn_pynq.models import model_with_cfg
            from brevitas_examples.bnn_pynq.extractor import ModelExtractor
            
            # Tenta carregar config pelo nome completo ou apenas nome base
            try:
                model_arch, _ = model_with_cfg(full_model_name, pretrained=False)
            except KeyError:
                model_arch, _ = model_with_cfg(model_name, pretrained=False) 

            # Usa o Extractor ou carrega direto se for apenas state_dict
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    model_arch.load_state_dict(checkpoint['state_dict'])
                else:
                    model_arch.load_state_dict(checkpoint)
                pytorch_model = model_arch
            except Exception as e:
                print(f"Erro ao carregar checkpoint local: {e}")
                raise e
    else:
        raise ValueError(f"Loader '{loader}' desconhecido.")

    if pytorch_model is None:
        raise RuntimeError("Falha ao carregar o modelo Pytorch.")

    # Seleção do Pipeline de Transformação FINN
    pipeline_map = {
        "SAT6_T1": _transform_pipeline_t2,
        "SAT6_T2": _transform_pipeline_t2,
        "MOBILENET": _transform_pipeline_mobilenet,
        "MNIST_TFC": _transform_pipeline_mnist,
        "CIFAR10_CNV": _transform_pipeline_cifar10,
        "SOYBEAN_CNV": _transform_pipeline_cnv, # Usa o pipeline CNV genérico
        "GRAPE_CNV": _transform_pipeline_cnv,    # Usa o pipeline CNV genérico
    }
    
    selected_pipeline = pipeline_map.get(topology_id)
    if selected_pipeline is None:
        raise ValueError(f"Pipeline FINN não encontrado para '{topology_id}'")
        
    return selected_pipeline(pytorch_model, build_dir, topology_id, quant_str)