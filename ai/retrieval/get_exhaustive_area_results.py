"""
get_exhaustive_area_results.py

Adaptação de get_total_area_results.py para os builds exaustivos do HARA V1.

Estrutura esperada em exhaustive_hw_builds/:
  MODEL_ID_TIMESTAMP/               (ex: MNIST_2W2A_2026-04-06_14-48-37)
    request.json                    ← metadados do modelo
    run1_baseline_folded/           ← cada run é uma config de HW
      stitched_ip/finn_design_partition_util.rpt
      intermediate_models/step_generate_estimate_reports.onnx
      report/estimate_layer_cycles.json
      final_hw_config.json
    run2_optimized/
    run3_optimized/
    ...

Saída:
  results/complete/vivado_all_submodules_area_attrs.csv   ← todos os submódulos
  results/splitted/vivado_<tipo>_area_attrs.csv           ← um CSV por tipo de submódulo
"""

import numpy as np
import pandas as pd
import os
import re
import json
import onnx
from onnx import helper
from collections import defaultdict, OrderedDict
import traceback

# =============================================================================
# CONFIGURAÇÃO — ajuste conforme necessário
# =============================================================================
EXHAUSTIVE_BUILDS_DIR = "/home/arthurely/Desktop/finn_chi2p/hara/exhaustive_hw_builds"

# Se quiser filtrar apenas builds específicos (lista de prefixos de model_id), 
# deixe vazio para processar todos.
FILTER_MODEL_IDS = ["MNIST", "SAT6_T2"]  # Exclui CIFAR10 (usado só como teste)

# Nomes dos arquivos procurados em cada run_dir
REPORT_FILENAME  = "finn_design_partition_util.rpt"
ONNX_FILENAME    = "step_generate_estimate_reports.onnx"
ONNX_SUBDIR      = "intermediate_models"
CYCLES_FILENAME  = "estimate_layer_cycles.json"
CYCLES_SUBDIR    = "report"
FOLDING_FILENAME = "final_hw_config.json"

# =============================================================================


def load_onnx_node_attrs(onnx_file_path):
    """Carrega atributos de todos os nós de um ONNX como dicionário node_name → attrs."""
    node_attrs_map = {}
    if not os.path.exists(onnx_file_path):
        return node_attrs_map
    try:
        model = onnx.load(onnx_file_path)
        for node in model.graph.node:
            attrs = {"op_type": node.op_type}
            for attr in node.attribute:
                val = helper.get_attribute_value(attr)
                if isinstance(val, list):
                    attrs[attr.name] = str(val)
                elif isinstance(val, bytes):
                    try:
                        attrs[attr.name] = val.decode("utf-8")
                    except UnicodeDecodeError:
                        attrs[attr.name] = str(val)
                elif isinstance(val, np.ndarray):
                    attrs[attr.name] = str(val.tolist())
                else:
                    attrs[attr.name] = str(val)
            node_attrs_map[node.name] = attrs
    except Exception as e:
        print(f"  [!] Erro ao carregar ONNX {onnx_file_path}: {e}")
    return node_attrs_map


def load_folding_config(folding_path):
    """Retorna o dicionário de folding (PE/SIMD/ram_style/resType por camada) preservando a ordem do pipeline."""
    if not os.path.exists(folding_path):
        return OrderedDict()
    try:
        with open(folding_path, "r") as f:
            return json.load(f, object_pairs_hook=OrderedDict)
    except Exception:
        return OrderedDict()


def load_estimate_cycles(cycles_path):
    """Retorna dicionário layer_name → ciclos estimados."""
    if not os.path.exists(cycles_path):
        return {}
    try:
        with open(cycles_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def parse_rpt_submodules(rpt_file_path):
    """
    Lê finn_design_partition_util.rpt e extrai recursos por submódulo
    direto abaixo de finn_design_i.

    Retorna lista de dicts:
        {instance_name, base_name, type_suffix, idx,
         Total LUTs, FFs, RAMB18, RAMB36, DSP Blocks, BRAM (36k eq.)}
    """
    submodule_type_pattern = re.compile(r"^(.*?)_(rtl|hls)_(\d+)(?:_(\d+))?$")
    results = []

    try:
        with open(rpt_file_path, "r", encoding="utf-8") as f:
            content = f.readlines()
    except Exception as e:
        print(f"  [!] Erro ao abrir RPT {rpt_file_path}: {e}")
        return results

    in_utilization_table = False
    header_indices = {}
    col_headers = []
    found_finn_design_i = False
    finn_design_i_indent = -1

    for line_raw in content:
        line_s = line_raw.strip()

        if not line_s.startswith("|") and "1. Utilization by Hierarchy" in line_s:
            in_utilization_table = True
            continue
        if not in_utilization_table:
            continue
        if not line_s.startswith("|"):
            if found_finn_design_i:
                break
            continue

        # Cabeçalho
        if not col_headers and "Instance" in line_s and "Module" in line_s:
            temp = [h.strip() for h in line_s.split("|") if h.strip()]
            if temp and temp[0] == "Instance" and temp[1] == "Module":
                col_headers = temp
                try:
                    for hdr in ["Instance", "Module", "Total LUTs", "FFs", "RAMB18", "RAMB36", "DSP Blocks"]:
                        if hdr in col_headers:
                            header_indices[hdr] = col_headers.index(hdr)
                        else:
                            raise ValueError(f"Cabeçalho ausente: {hdr}")
                except ValueError as ve:
                    print(f"  [!] {ve} em {rpt_file_path}")
                    return results
            continue

        if not header_indices:
            continue

        try:
            first_cell_raw = line_raw.split("|", 2)[1]
        except IndexError:
            continue

        indent = len(first_cell_raw) - len(first_cell_raw.lstrip(" "))
        instance_name = first_cell_raw.strip()
        data_parts = [p.strip() for p in line_s.split("|")[1:-1]]

        if len(data_parts) != len(col_headers):
            continue

        if not found_finn_design_i:
            if instance_name == "finn_design_i":
                found_finn_design_i = True
                finn_design_i_indent = indent
            continue

        # Submódulos diretos de finn_design_i
        if indent == finn_design_i_indent + 2:
            if instance_name == "(finn_design_i)":
                continue
            try:
                total_luts = int(data_parts[header_indices["Total LUTs"]])
                ffs        = int(data_parts[header_indices["FFs"]])
                ramb18     = float(data_parts[header_indices["RAMB18"]])
                ramb36     = float(data_parts[header_indices["RAMB36"]])
                dsp        = int(data_parts[header_indices["DSP Blocks"]])
                bram_eq    = ramb36 + ramb18 * 0.5

                m = submodule_type_pattern.match(instance_name)
                if m:
                    base_name    = m.group(1)
                    type_suffix  = m.group(2)
                    idx          = int(m.group(3))
                else:
                    base_name    = instance_name
                    type_suffix  = "unknown"
                    idx          = 0

                results.append({
                    "instance_name":  instance_name,
                    "base_name":      base_name,
                    "type_suffix":    type_suffix,
                    "layer_idx":      idx,
                    "Total LUT":      total_luts,
                    "Total FFs":      ffs,
                    "RAMB18":         ramb18,
                    "RAMB36":         ramb36,
                    "BRAM (36k eq.)": bram_eq,
                    "DSP Blocks":     dsp,
                })
            except (ValueError, KeyError) as e:
                print(f"  [!] Erro ao ler linha '{instance_name}': {e}")

        elif indent <= finn_design_i_indent:
            # Saiu do escopo de finn_design_i
            found_finn_design_i = False

    return results


def collect_exhaustive_builds():
    """
    Itera sobre exhaustive_hw_builds/MODEL_TIMESTAMP/runN_*/
    e coleta dados de área + atributos ONNX + folding + ciclos.
    """
    all_rows_complete = []
    all_rows_by_type  = defaultdict(list)
    all_onnx_attr_keys = set()

    builds_base = EXHAUSTIVE_BUILDS_DIR
    if not os.path.isdir(builds_base):
        print(f"[!] Diretório não encontrado: {builds_base}")
        return all_rows_complete, all_rows_by_type, all_onnx_attr_keys

    build_sessions = sorted(os.listdir(builds_base))

    for session_name in build_sessions:
        session_path = os.path.join(builds_base, session_name)
        if not os.path.isdir(session_path):
            continue

        if FILTER_MODEL_IDS:
            if not any(session_name.startswith(mid) for mid in FILTER_MODEL_IDS):
                continue

        model_id = session_name[:-20] if len(session_name) > 20 else session_name
        timestamp = session_name[-19:] if len(session_name) > 20 else ""

        request_meta = {}
        req_path = os.path.join(session_path, "request.json")
        if os.path.exists(req_path):
            try:
                with open(req_path) as f:
                    req_data = json.load(f)
                    request_meta["fixed_ram_style"] = (
                        req_data.get("fixed_resources", {})
                        .get("MVAU_hls", {})
                        .get("ram_style", "unknown")
                    )
                    request_meta["fixed_resType"] = (
                        req_data.get("fixed_resources", {})
                        .get("MVAU_hls", {})
                        .get("resType", "unknown")
                    )
            except Exception:
                pass

        print(f"\n[>] Sessão: {session_name}")

        run_dirs = sorted(
            [d for d in os.listdir(session_path)
             if os.path.isdir(os.path.join(session_path, d))
             and (d.startswith("run") and not d.endswith("_model_files"))],
            key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0
        )

        for run_name in run_dirs:
            run_path = os.path.join(session_path, run_name)

            rpt_path     = os.path.join(run_path, "stitched_ip", REPORT_FILENAME)
            onnx_path    = os.path.join(run_path, ONNX_SUBDIR, ONNX_FILENAME)
            cycles_path  = os.path.join(run_path, CYCLES_SUBDIR, CYCLES_FILENAME)
            folding_path = os.path.join(run_path, FOLDING_FILENAME)

            if not os.path.exists(rpt_path):
                continue

            run_num_match = re.search(r"\d+", run_name)
            run_number = int(run_num_match.group()) if run_num_match else -1
            is_baseline = 1 if "baseline" in run_name else 0

            print(f"  -> Processando {run_name}...")

            onnx_attrs   = load_onnx_node_attrs(onnx_path)
            folding      = load_folding_config(folding_path)
            cycles       = load_estimate_cycles(cycles_path)

            submodules = parse_rpt_submodules(rpt_path)

            if not submodules:
                print(f"     [!] Nenhum submódulo encontrado em {rpt_path}")
                continue

            for sm in submodules:
                instance = sm["instance_name"]
                base     = sm["base_name"]
                idx      = sm["layer_idx"]

                onnx_node_candidates = [f"{base}_{idx}", instance, base]
                sm_onnx_attrs = {}
                for cand in onnx_node_candidates:
                    if cand in onnx_attrs:
                        sm_onnx_attrs = onnx_attrs[cand].copy()
                        break

                layer_folding = folding.get(f"{base}_hls_{idx}", 
                               folding.get(f"{base}_rtl_{idx}",
                               folding.get(f"{base}_{idx}", {})))
                
                fifo_folding = folding.get(instance, {})
                fold_pe    = layer_folding.get("PE", None)
                fold_simd  = layer_folding.get("SIMD", None)
                fold_ram   = layer_folding.get("ram_style", fifo_folding.get("ram_style", None))
                fold_res   = layer_folding.get("resType", None)
                
                fifo_depth      = fifo_folding.get("depth", None) if "StreamingFIFO" in base else None
                fifo_impl_style = fifo_folding.get("impl_style", None) if "StreamingFIFO" in base else None

                # Resgate da largura do dado (inWidth) para StreamingFIFOs injetando no nó da FIFO
                if "StreamingFIFO" in base:
                    pipeline_keys = [k for k in folding.keys() if k != "Defaults"]
                    if instance in pipeline_keys:
                        pos = pipeline_keys.index(instance)
                        if pos > 0:
                            upstream_name = pipeline_keys[pos - 1]
                            m_up = re.match(r"^(.+?)_(rtl|hls)_(\d+)$", upstream_name)
                            up_base = m_up.group(1) if m_up else upstream_name
                            up_idx  = m_up.group(3) if m_up else "0"
                            up_cands = [f"{up_base}_{up_idx}", upstream_name, up_base]
                            
                            for cand in up_cands:
                                if cand in onnx_attrs:
                                    up_attrs = onnx_attrs[cand]
                                    fifo_inW = up_attrs.get("outWidth")
                                    if not fifo_inW and "outputDataType" in up_attrs:
                                        dt = str(up_attrs["outputDataType"]).upper()
                                        bits = 1 if "BINARY" in dt else (int(re.search(r"\d+", dt).group(0)) if re.search(r"\d+", dt) else 1)
                                        ofm = up_attrs.get("OFMChannels", 1)
                                        if isinstance(ofm, str) and "[" in ofm:
                                            try:
                                                import ast
                                                lst = ast.literal_eval(ofm)
                                                ofm = int(np.prod(lst))
                                            except:
                                                ofm = 1
                                        fifo_inW = int(ofm) * bits
                                    
                                    if fifo_inW:
                                        sm_onnx_attrs["inWidth"] = str(fifo_inW)
                                    break

                all_onnx_attr_keys.update(sm_onnx_attrs.keys())

                layer_cycles = cycles.get(f"{base}_hls_{idx}",
                               cycles.get(f"{base}_rtl_{idx}",
                               cycles.get(f"{base}_{idx}", None)))

                row = {
                    "model_id":         model_id,
                    "session":          session_name,
                    "timestamp":        timestamp,
                    "run_name":         run_name,
                    "run_number":       run_number,
                    "is_baseline":      is_baseline,
                    **request_meta,
                    "Submodule Instance": instance,
                    "base_name":        base,
                    "isRTL":            1 if sm["type_suffix"] == "rtl" else 0,
                    "isHLS":            1 if sm["type_suffix"] == "hls" else 0,
                    "layer_idx":        idx,
                    **sm_onnx_attrs,
                    "PE":               fold_pe,
                    "SIMD":             fold_simd,
                    "ram_style":        fold_ram,
                    "resType":          fold_res,
                    "depth":            fifo_depth,
                    "impl_style":       fifo_impl_style,
                    "estimated_cycles": layer_cycles,
                    "Total LUT":        sm["Total LUT"],
                    "Total FFs":        sm["Total FFs"],
                    "BRAM (36k eq.)":   sm["BRAM (36k eq.)"],
                    "DSP Blocks":       sm["DSP Blocks"],
                }

                all_rows_complete.append(row)
                all_rows_by_type[base].append(row)

    return all_rows_complete, all_rows_by_type, all_onnx_attr_keys


def main():
    print("=" * 60)
    print("HARA Exhaustive Builds — Area Retrieval Pipeline")
    print(f"Diretório base: {EXHAUSTIVE_BUILDS_DIR}")
    if FILTER_MODEL_IDS:
        print(f"Filtrando modelos: {FILTER_MODEL_IDS}")
    print("=" * 60)

    all_rows, by_type, onnx_keys = collect_exhaustive_builds()

    if not all_rows:
        print("\n[!] Nenhum dado coletado. Verifique o diretório e os builds disponíveis.")
        return

    print(f"\n[✓] Total de registros coletados: {len(all_rows)}")
    print(f"[✓] Tipos de submódulos encontrados: {sorted(by_type.keys())}")

    fixed_front = [
        "model_id", "session", "timestamp", "run_name", "run_number", "is_baseline",
        "fixed_ram_style", "fixed_resType",
        "Submodule Instance", "base_name", "isRTL", "isHLS", "layer_idx",
    ]
    onnx_sorted = sorted(onnx_keys)
    fixed_back = [
        "PE", "SIMD", "ram_style", "resType", "depth", "impl_style", "estimated_cycles",
        "Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks",
    ]
    col_order = fixed_front + [k for k in onnx_sorted if k not in fixed_front + fixed_back] + fixed_back

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_dir  = os.path.join(script_dir, "results")
    complete_dir = os.path.join(results_dir, "complete")
    split_dir    = os.path.join(results_dir, "splitted")
    os.makedirs(complete_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    df_all = pd.DataFrame(all_rows)
    for col in col_order:
        if col not in df_all.columns:
            df_all[col] = ""
    df_all = df_all[[c for c in col_order if c in df_all.columns]]

    out_complete = os.path.join(complete_dir, "exhaustive_all_submodules_area_attrs.csv")
    df_all.to_csv(out_complete, index=False)
    print(f"\n[✓] CSV completo salvo: {out_complete}")
    print(f"    {df_all.shape[0]} linhas × {df_all.shape[1]} colunas")

    saved = 0
    for submod_type, rows in by_type.items():
        df_type = pd.DataFrame(rows)
        for col in col_order:
            if col not in df_type.columns:
                df_type[col] = ""
        df_type = df_type[[c for c in col_order if c in df_type.columns]]

        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in submod_type)
        out_split = os.path.join(split_dir, f"exhaustive_{safe_name}_area_attrs.csv")
        df_type.to_csv(out_split, index=False)
        print(f"  [✓] {submod_type}: {df_type.shape[0]} registros → {out_split}")
        saved += 1

    print(f"\n[✓] {saved} CSV(s) por tipo salvo(s) em {split_dir}")
    print("\nPróximo passo: execute pre_processing.py para limpeza e one-hot encoding.")


if __name__ == "__main__":
    main()