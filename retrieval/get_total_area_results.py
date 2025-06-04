import numpy as np
import pandas as pd
import os
import re
import onnx
from onnx import helper
from collections import defaultdict
import traceback # Para debugging mais detalhado de exceções

def parse_submodule_utilization_with_onnx_modified():
    """
    Pesquisa recursivamente por relatórios de utilização do Vivado e ficheiros ONNX associados.
    Extrai utilização de recursos para submódulos de 'finn_design_i' e atributos ONNX.
    Salva dados em CSVs de duas formas:
    1. Um CSV único com todos os submódulos em 'results/complete/'.
    2. CSVs separados por nome base do submódulo em 'results/splitted/', com atributos ONNX antes dos dados de área.
    BRAM é reportado como equivalentes de 36k (RAMB36=1, RAMB18=0.5).
    """
    base_directory = "/home/arthurely/Desktop/finn/hara/builds" # ATENÇÃO: Verifique se este caminho está correto

    if not os.path.isdir(base_directory):
        print(f"Erro: Diretório '{base_directory}' não encontrado.")
        return

    report_filename_to_search = "finn_design_partition_util.rpt"
    onnx_filename = "step_generate_estimate_reports.onnx"
    onnx_subdir = "intermediate_models"

    ignore_onnx_attrs = [] # Mantido vazio conforme o último input

    all_submodules_data_by_base_name = defaultdict(list)
    all_submodules_data_for_complete_csv = [] # Nova lista para o CSV único e completo
    found_files_count = 0
    all_seen_onnx_attr_keys = set()

    submodule_type_pattern = re.compile(r'^(.*?)_(rtl|hls)_(\d+)$')

    print(f"A pesquisar por '{report_filename_to_search}' em '{base_directory}'...")

    potential_hw_config_dirs = []
    for entry in os.listdir(base_directory):
        path1 = os.path.join(base_directory, entry)
        if os.path.isdir(path1):
            if entry.startswith("run_"):
                for sub_entry in os.listdir(path1):
                    path2 = os.path.join(path1, sub_entry)
                    if os.path.isdir(path2):
                        potential_hw_config_dirs.append(path2)
            else:
                potential_hw_config_dirs.append(path1)

    for hw_config_dir_path in potential_hw_config_dirs:
        hardware_config_name = os.path.basename(hw_config_dir_path)
        rpt_file_path = os.path.join(hw_config_dir_path, "stitched_ip", report_filename_to_search)
        onnx_file_path = os.path.join(hw_config_dir_path, onnx_subdir, onnx_filename)

        if not os.path.exists(rpt_file_path):
            continue

        print(f"A processar config: {hardware_config_name}")
        found_files_count +=1

        current_hw_config_onnx_nodes_attrs = {}
        if os.path.exists(onnx_file_path):
            try:
                model = onnx.load(onnx_file_path)
                for node in model.graph.node:
                    node_attrs = {'op_type': node.op_type}
                    all_seen_onnx_attr_keys.add('op_type')
                    for attr in node.attribute:
                        if attr.name not in ignore_onnx_attrs:
                            attr_val = helper.get_attribute_value(attr)
                            if isinstance(attr_val, list):
                                node_attrs[attr.name] = str(attr_val)
                            elif isinstance(attr_val, bytes):
                                try:
                                    node_attrs[attr.name] = attr_val.decode('utf-8')
                                except UnicodeDecodeError:
                                    node_attrs[attr.name] = str(attr_val)
                            elif isinstance(attr_val, np.ndarray):
                                node_attrs[attr.name] = str(attr_val.tolist())
                            else:
                                node_attrs[attr.name] = str(attr_val)
                            all_seen_onnx_attr_keys.add(attr.name)
                    current_hw_config_onnx_nodes_attrs[node.name] = node_attrs
            except Exception as e:
                print(f"Erro ao carregar ou processar ficheiro ONNX {onnx_file_path}: {e}")
        else:
            print(f"Aviso: Ficheiro ONNX não encontrado para {hardware_config_name} em {onnx_file_path}")

        try:
            with open(rpt_file_path, 'r', encoding='utf-8') as my_file:
                content = my_file.readlines()

            in_utilization_table = False
            table_header_indices = {}
            report_column_headers_stripped = []
            found_finn_design_i = False
            finn_design_i_indent_level = -1

            for line_num, line_content_raw in enumerate(content):
                line_stripped_for_table_check = line_content_raw.strip()

                if not line_stripped_for_table_check.startswith("|") and "1. Utilization by Hierarchy" in line_stripped_for_table_check:
                    in_utilization_table = True; continue
                if not in_utilization_table: continue
                if not line_stripped_for_table_check.startswith("|"):
                    if found_finn_design_i: break
                    continue

                if not report_column_headers_stripped and "Instance" in line_stripped_for_table_check and "Module" in line_stripped_for_table_check:
                    temp_headers = [h.strip() for h in line_stripped_for_table_check.split('|') if h.strip()]
                    if temp_headers and temp_headers[0] == "Instance" and temp_headers[1] == "Module":
                        report_column_headers_stripped = temp_headers
                        try:
                            critical_headers = ["Instance", "Module", "Total LUTs", "FFs", "RAMB18", "RAMB36", "DSP Blocks"]
                            for header_name in critical_headers:
                                if header_name in report_column_headers_stripped:
                                    table_header_indices[header_name] = report_column_headers_stripped.index(header_name)
                                else: raise ValueError(f"Falta o cabeçalho crítico de parsing: {header_name}")
                        except ValueError as ve:
                            print(f"Aviso: Erro ao processar cabeçalhos em {rpt_file_path}: {ve}. Cabeçalhos: {report_column_headers_stripped}"); in_utilization_table = False; break
                        continue

                if table_header_indices and line_stripped_for_table_check.startswith("|"):
                    try: first_cell_raw_content_with_pipe = line_content_raw.split('|', 2)[1]
                    except IndexError: continue

                    current_indent = len(first_cell_raw_content_with_pipe) - len(first_cell_raw_content_with_pipe.lstrip(' '))
                    instance_name_stripped = first_cell_raw_content_with_pipe.strip()
                    data_parts_from_stripped_line = line_stripped_for_table_check.split('|')
                    data_values_stripped = [p.strip() for p in data_parts_from_stripped_line[1:-1]]

                    if len(data_values_stripped) != len(report_column_headers_stripped): continue

                    if not found_finn_design_i:
                        if instance_name_stripped == "finn_design_i":
                            found_finn_design_i = True; finn_design_i_indent_level = current_indent; continue
                    elif found_finn_design_i:
                        if current_indent == finn_design_i_indent_level + 2:
                            submodule_name_full_instance = instance_name_stripped
                            if submodule_name_full_instance == "(finn_design_i)": continue

                            is_rtl, is_hls = False, False
                            submodule_group_key = submodule_name_full_instance
                            onnx_node_name_candidate_priority1 = None
                            onnx_node_name_candidate_priority2 = submodule_name_full_instance
                            onnx_node_name_candidate_priority3 = None

                            match = submodule_type_pattern.match(submodule_name_full_instance)
                            if match:
                                base_name = match.group(1)
                                type_suffix = match.group(2)
                                digits = match.group(3)
                                submodule_group_key = base_name
                                onnx_node_name_candidate_priority1 = f"{base_name}_{digits}"
                                onnx_node_name_candidate_priority3 = base_name

                                if type_suffix == "rtl":
                                    is_rtl = 1
                                    is_hls = 0
                                elif type_suffix == "hls":
                                    is_rtl = 0
                                    is_hls = 1
                                # Se não for 'rtl' nem 'hls', ambos permanecem False
                            else:
                                print(f"Aviso: Submódulo '{submodule_name_full_instance}' em {hardware_config_name} não corresponde ao padrão _rtl_X/_hls_X. Será agrupado como '{submodule_group_key}'. 'isRTL' e 'isHLS' serão definidos como Falso.")

                            onnx_attrs_for_submodule = {}
                            found_onnx_node_name = None

                            if onnx_node_name_candidate_priority1 and onnx_node_name_candidate_priority1 in current_hw_config_onnx_nodes_attrs:
                                onnx_attrs_for_submodule = current_hw_config_onnx_nodes_attrs[onnx_node_name_candidate_priority1]
                                found_onnx_node_name = onnx_node_name_candidate_priority1
                            elif onnx_node_name_candidate_priority2 in current_hw_config_onnx_nodes_attrs:
                                onnx_attrs_for_submodule = current_hw_config_onnx_nodes_attrs[onnx_node_name_candidate_priority2]
                                found_onnx_node_name = onnx_node_name_candidate_priority2
                            elif onnx_node_name_candidate_priority3 and onnx_node_name_candidate_priority3 in current_hw_config_onnx_nodes_attrs:
                                onnx_attrs_for_submodule = current_hw_config_onnx_nodes_attrs[onnx_node_name_candidate_priority3]
                                found_onnx_node_name = onnx_node_name_candidate_priority3

                            if not found_onnx_node_name:
                                print(f"Aviso: Nenhum nó ONNX correspondente encontrado para HW '{submodule_name_full_instance}' (tentativas: P1='{onnx_node_name_candidate_priority1}', P2='{onnx_node_name_candidate_priority2}', P3='{onnx_node_name_candidate_priority3}') em {hardware_config_name}")
                                if current_hw_config_onnx_nodes_attrs:
                                    print(f"  Nós ONNX disponíveis para {hardware_config_name}: {list(current_hw_config_onnx_nodes_attrs.keys())}")
                                else:
                                    print(f"  Nenhum nó ONNX foi carregado para {hardware_config_name}.")

                            try:
                                total_luts = int(data_values_stripped[table_header_indices["Total LUTs"]])
                                total_ffs = int(data_values_stripped[table_header_indices["FFs"]])
                                ramb18 = float(data_values_stripped[table_header_indices["RAMB18"]])
                                ramb36 = float(data_values_stripped[table_header_indices["RAMB36"]])
                                dsp_blocks = int(data_values_stripped[table_header_indices["DSP Blocks"]])
                                bram_36k_eq = (ramb36 * 1.0) + (ramb18 * 0.5)

                                row_dict = {
                                    'Hardware config': hardware_config_name,
                                    'Submodule Instance': submodule_name_full_instance,
                                    'isRTL': is_rtl, 'isHLS': is_hls,
                                    **onnx_attrs_for_submodule,
                                    'Total LUT': total_luts, 'Total FFs': total_ffs,
                                    'BRAM (36k eq.)': bram_36k_eq, 'DSP Blocks': dsp_blocks
                                }
                                all_submodules_data_by_base_name[submodule_group_key].append(row_dict)
                                all_submodules_data_for_complete_csv.append(row_dict) # Adiciona à lista para o CSV completo
                            except (ValueError, KeyError) as ve:
                                print(f"Erro ao converter/aceder a dados de área para '{submodule_name_full_instance}' em {rpt_file_path}: {ve}")
                        elif current_indent <= finn_design_i_indent_level:
                            found_finn_design_i = False
        except FileNotFoundError:
            print(f"Erro: Ficheiro RPT {rpt_file_path} não encontrado durante o processamento de área.")
        except Exception as e:
            print(f"Erro geral ao processar dados de área do ficheiro {rpt_file_path}: {e}")
            traceback.print_exc()

    if found_files_count == 0:
        print(f"Nenhum ficheiro '{report_filename_to_search}' encontrado.")
        return
    if not all_submodules_data_for_complete_csv: # Se a lista completa estiver vazia, nada foi processado
        print("Nenhum dado de submódulo foi processado com sucesso.")
        return

    # --- Configuração dos diretórios de resultados ---
    base_results_dir = './results'
    complete_results_dir = os.path.join(base_results_dir, 'complete')
    splitted_results_dir = os.path.join(base_results_dir, 'splitted')

    os.makedirs(complete_results_dir, exist_ok=True)
    print(f"\nDiretório para CSV completo criado/existente: {os.path.abspath(complete_results_dir)}")
    os.makedirs(splitted_results_dir, exist_ok=True)
    print(f"Diretório para CSVs divididos criado/existente: {os.path.abspath(splitted_results_dir)}")

    # --- Definição dos cabeçalhos do CSV ---
    sorted_onnx_attr_headers = sorted(list(all_seen_onnx_attr_keys))
    final_csv_headers = ['Hardware config', 'Submodule Instance', 'isRTL', 'isHLS'] + \
                        sorted_onnx_attr_headers + \
                        ['Total LUT', 'Total FFs', 'BRAM (36k eq.)', 'DSP Blocks']

    # --- Cenário 1: Salvar CSV Completo ---
    if all_submodules_data_for_complete_csv:
        df_complete = pd.DataFrame(all_submodules_data_for_complete_csv)
        # Garantir que todas as colunas existam e estejam na ordem correta
        for col in final_csv_headers:
            if col not in df_complete.columns:
                df_complete[col] = "" # Ou pd.NA / np.nan se preferir valores nulos explícitos
        df_complete = df_complete[final_csv_headers]

        complete_output_filename = "vivado_all_submodules_area_attrs.csv"
        complete_output_csv_path = os.path.join(complete_results_dir, complete_output_filename)
        try:
            df_complete.to_csv(complete_output_csv_path, index=False)
            print(f"\nCSV completo com todos os submódulos guardado em: {os.path.abspath(complete_output_csv_path)}")
        except Exception as e:
            print(f"Erro ao guardar CSV completo em {complete_output_csv_path}: {e}")
            traceback.print_exc()
    else:
        print("\nNenhum dado disponível para gerar o CSV completo.")

    # --- Cenário 2: Salvar CSVs Divididos ---
    num_splitted_csv_saved = 0
    if all_submodules_data_by_base_name:
        print("\nA processar CSVs divididos por grupo de submódulo...")
        for submodule_group_key, list_of_row_dicts in all_submodules_data_by_base_name.items():
            if list_of_row_dicts:
                df_type = pd.DataFrame(list_of_row_dicts)
                # Garantir que todas as colunas existam e estejam na ordem correta
                for col in final_csv_headers:
                    if col not in df_type.columns:
                        df_type[col] = ""
                df_type = df_type[final_csv_headers]

                safe_filename_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in submodule_group_key)
                output_filename = f"vivado_{safe_filename_base}_area_attrs.csv"
                output_csv_path = os.path.join(splitted_results_dir, output_filename)

                try:
                    df_type.to_csv(output_csv_path, index=False)
                    print(f"  Dados divididos para '{submodule_group_key}' guardados em: {os.path.abspath(output_csv_path)}")
                    num_splitted_csv_saved += 1
                except Exception as e:
                    print(f"  Erro ao guardar CSV dividido para '{submodule_group_key}' em {output_csv_path}: {e}")
                    traceback.print_exc()
            else:
                # Esta condição não deve ocorrer se defaultdict(list) for usado corretamente e a chave existir
                print(f"  Nenhuma linha de dados encontrada para o grupo de submódulos '{submodule_group_key}'. CSV não criado.")

        if num_splitted_csv_saved > 0:
            print(f"\n{num_splitted_csv_saved} ficheiro(s) CSV dividido(s) guardado(s) com sucesso no diretório '{os.path.abspath(splitted_results_dir)}'.")
        elif all_submodules_data_by_base_name: # Havia chaves de grupo mas nenhum CSV foi salvo
             print(f"\nNenhum ficheiro CSV dividido foi gerado, apesar de existirem grupos de submódulos.")
        # else: A condição de não haver dados já foi tratada por all_submodules_data_for_complete_csv
    else:
         # Esta condição também deve ser coberta pela verificação de all_submodules_data_for_complete_csv
        print("\nNenhum dado agrupado por submódulo para gerar CSVs divididos.")

if __name__ == '__main__':
    parse_submodule_utilization_with_onnx_modified()