import pandas as pd
import numpy as np
import os
import re
import ast
from datetime import datetime
import shutil

# Funções auxiliares (mantidas como no original)
def is_listlike_string(val):
    return isinstance(val, str) and val.startswith("[") and val.endswith("]")

def clean_listlike_column(val):
    try:
        if is_listlike_string(val):
            lst = ast.literal_eval(val)
            if isinstance(lst, list) and all(isinstance(x, (int, float)) for x in lst):
                return np.prod(lst)
    except:
        pass
    return val

def expand_listlike_columns(df):
    cols_to_expand = [col for col in df.columns if df[col].apply(is_listlike_string).any()]
    for col in cols_to_expand:
        new_col_name = f"{col} (dimensions flattened)"
        idx = df.columns.get_loc(col)
        expanded_series = df[col].apply(clean_listlike_column)
        df.insert(idx + 1, new_col_name, expanded_series)
    
    df.drop(columns=cols_to_expand, inplace=True)
    return df

def extract_bitwidth(x):
    x_str = str(x).upper()
    if "B'BINARY" in x_str:
        return 1
    match = re.search(r'(UINT|INT)(\d+)', x_str)
    return int(match.group(2)) if match else None

# Constantes de configuração
area_cols_to_preserve = [
    "Total LUT", "Total FFs", "BRAM (36k eq.)", "DSP Blocks",
    "ConvKernelDim", "Dilation", "Stride", "SRLs", "inWidth", "Padding", "K",
    "Labels", "PE", "inFIFODepths", "outFIFODepths", "numInputVectors",
    "depth", "impl_style", "ram_style", "resType"
]

def clean_dataframe(df_input, df_name_key=""):
    df = df_input.copy()

    # --- FIX: INSERIR DEPTHS REAIS NAS FIFOS VIA MERGE ---
    if "StreamingFIFO" in df_name_key:
        fifo_depth_csv = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "results", "fifo_depth", "exhaustive_fifo_depths.csv"
        )
        if os.path.exists(fifo_depth_csv):
            df_depths = pd.read_csv(fifo_depth_csv)
            # Chaves para garantir que não vamos cruzar dados de runs diferentes
            keys = ["model_id", "session", "run_name"]
            df_depths = df_depths[keys + ["fifo_name", "depth"]]
            
            # Remove a coluna depth vazia atual para não duplicar no merge (depth_x, depth_y)
            if "depth" in df.columns:
                df = df.drop(columns=["depth"])
                
            # Cruza os dados casando as sessões e o nome da instância
            df = df.merge(
                df_depths, 
                how="left", 
                left_on=keys + ["Submodule Instance"], 
                right_on=keys + ["fifo_name"]
            )
            df.drop(columns=["fifo_name"], inplace=True, errors="ignore")
            # Preenche apenas as que realmente sobrarem nulas com o default
            df["depth"] = df["depth"].fillna(2)

    # --- FIX: REMOVER COLUNAS DE VAZAMENTO DE DADOS ANTES TRATAMENTOS ---
    leakage_cols = [
        "model_id", "session", "timestamp", "run_name", "run_number", 
        "is_baseline", "fixed_ram_style", "fixed_resType", 
        "Submodule Instance", "base_name", "layer_idx"
    ]
    df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True, errors='ignore')

    # Colunas a serem preservadas de certas etapas de limpeza
    existing_area_cols = [col for col in area_cols_to_preserve if col in df.columns]
    cols_to_preserve_current_df = list(existing_area_cols)

    # Inserir colunas "(bits)" a partir de colunas de tipo de dados
    datatype_cols = [col for col in df.columns if "datatype" in col.lower()]
    generated_bit_cols = []
    for col in datatype_cols:
        new_col_name = f"{col} (bits)"
        generated_bit_cols.append(new_col_name)
        bitwidths = df[col].apply(extract_bitwidth)
        col_idx = df.columns.get_loc(col)
        df.insert(col_idx + 1, new_col_name, bitwidths)
    
    cols_to_preserve_current_df.extend(generated_bit_cols)
    df.drop(columns=datatype_cols, inplace=True, errors='ignore')

    # 1. Remover colunas com valor único
    nunique = df.nunique()
    cols_to_drop_unique = [col for col in nunique[nunique <= 1].index if col not in cols_to_preserve_current_df]
    df.drop(columns=cols_to_drop_unique, inplace=True, errors='ignore')

    # 2. Tentar converter para numérico (movido para o final)

    # 3. Remover colunas altamente correlacionadas (apenas numéricas)
    numeric_df = df.select_dtypes(include=np.number)
    cols_to_drop_corr = []
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_to_drop_corr = [
            column for column in upper.columns
            if any(upper[column] > 0.98) and column not in cols_to_preserve_current_df
        ]
        if cols_to_drop_corr:
            print(f"  Removendo colunas altamente correlacionadas: {cols_to_drop_corr}")
        df.drop(columns=cols_to_drop_corr, inplace=True, errors='ignore')

    # 4. Remover duplicatas
    df.drop_duplicates(inplace=True)

    # 5. Lógica de remoção de colunas com valores ausentes foi desativada
    cols_to_drop_missing = [] 

    # 6. Lógica de remoção de colunas específicas foi desativada

    # 7. Renomear colunas IFM/OFM
    rename_map = {
        col: col.replace("IFM", "Input Feature Map ").replace("OFM", "Output Feature Map ")
        for col in df.columns
        if "IFM" in col or "OFM" in col
    }
    df.rename(columns=rename_map, inplace=True)

    # 8. Expandir colunas com valores list-like
    df = expand_listlike_columns(df)

    # --- INÍCIO DA LÓGICA DE ONE-HOT ENCODING (COM MANUTENÇÃO DE ORDEM) ---
    # 9. Aplicar One-Hot Encoding em colunas categóricas, ignorando as especificadas
    
    cols_to_ignore_encoding = ["Hardware config", "Submodule Instance"]
    all_object_cols = df.select_dtypes(include=['object']).columns
    categorical_cols_to_encode = [col for col in all_object_cols if col not in cols_to_ignore_encoding]
    
    if categorical_cols_to_encode:
        print(f"  Aplicando One-Hot Encoding com preservação de ordem nas colunas: {categorical_cols_to_encode}")

        for cat_col in categorical_cols_to_encode:
            # 1. Salvar a posição original da coluna
            original_idx = df.columns.get_loc(cat_col)

            # 2. Gerar as colunas dummy para a coluna categórica atual
            dummies = pd.get_dummies(df[cat_col], prefix=cat_col, prefix_sep='_', dtype=int)

            # 3. Renomear as colunas dummy para o formato "isColunaValor"
            rename_map = {}
            for dummy_col in dummies.columns:
                value = dummy_col.split(f"{cat_col}_", 1)[1]
                clean_value = ''.join(e for e in str(value) if e.isalnum()).capitalize()
                
                # Preserva o camelCase original da coluna, apenas capitaliza a primeira letra
                cat_col_pascal = cat_col[0].upper() + cat_col[1:] if cat_col else ""
                
                new_name = f"is{cat_col_pascal}{clean_value}"
                rename_map[dummy_col] = new_name
            dummies.rename(columns=rename_map, inplace=True)
            
            # 4. Remover a coluna categórica original do DataFrame
            df.drop(columns=[cat_col], inplace=True)

            # 5. Inserir as novas colunas na posição original
            cols_before = df.columns[:original_idx]
            cols_after = df.columns[original_idx:]
            
            df = pd.concat([df[cols_before], dummies, df[cols_after]], axis=1)

        print(f"  Colunas categóricas transformadas e reordenadas.")
    # --- FIM DA LÓGICA DE ONE-HOT ENCODING ---

    # 10. Tentar converter todas as colunas restantes para numérico (se possível)
    for col in df.columns:
        if col not in cols_to_ignore_encoding:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    stats = {
        "original_columns": len(nunique),
        "removed_constant": len(cols_to_drop_unique),
        "removed_high_corr": len(cols_to_drop_corr),
        "removed_missing": len(cols_to_drop_missing),
        "categorical_encoded": len(categorical_cols_to_encode),
        "final_columns": df.shape[1]
    }
    return df, stats

def main_pipeline():
    project_root_dir = "." 
    scenario_roots = [
        os.path.join(project_root_dir, "results/complete"),
        os.path.join(project_root_dir, "results/splitted")
    ]
    print(f"Execução do pipeline de pré-processamento iniciada.\n")
    for scenario_input_dir in scenario_roots:
        scenario_name = os.path.basename(scenario_input_dir)
        print(f"--- Processando Cenário: {scenario_name.upper()} ---")
        abs_scenario_input_dir = os.path.abspath(scenario_input_dir)
        print(f"Diretório de entrada: {abs_scenario_input_dir}")
        if not os.path.isdir(abs_scenario_input_dir):
            print(f"⚠️  Aviso: Diretório de entrada não encontrado: {abs_scenario_input_dir}. Pulando cenário.\n")
            continue
        try:
            input_csv_files = [f for f in os.listdir(abs_scenario_input_dir) if os.path.isfile(os.path.join(abs_scenario_input_dir, f)) and f.endswith(".csv")]
        except FileNotFoundError:
            print(f"⚠️  Aviso: Erro ao listar arquivos em {abs_scenario_input_dir}. Pulando cenário.\n")
            continue
        if not input_csv_files:
            print(f"ℹ️  Nenhum arquivo CSV encontrado em {abs_scenario_input_dir}. Pulando cenário.\n")
            continue
        print(f"Encontrados {len(input_csv_files)} arquivo(s) CSV para processar: {', '.join(input_csv_files)}")
        output_dir = os.path.join(abs_scenario_input_dir, "preprocessed")
        os.makedirs(output_dir, exist_ok=True) 
        print(f"Diretório de saída: {output_dir}")
        scenario_cleaned_dfs = {}
        scenario_summary_stats = {}
        for csv_file in input_csv_files:
            file_path = os.path.join(abs_scenario_input_dir, csv_file)
            df_name_key = csv_file.replace(".csv", "") 
            print(f"\n  Processando arquivo: {csv_file}...")
            try:
                original_df = pd.read_csv(file_path)
                cleaned_df, processing_stats = clean_dataframe(original_df, df_name_key)
                scenario_cleaned_dfs[df_name_key] = cleaned_df
                scenario_summary_stats[df_name_key] = processing_stats
                print(f"  Arquivo '{csv_file}' limpo com sucesso.")
            except Exception as e:
                print(f"  ❌ Erro ao processar o arquivo {csv_file}: {e}")
        if not scenario_cleaned_dfs:
            print(f"ℹ️  Nenhum DataFrame foi limpo com sucesso para o cenário '{scenario_name}'. Operações de salvamento puladas.\n")
            continue
        
        summary_df = pd.DataFrame(scenario_summary_stats).T
        summary_df = summary_df.reset_index().rename(columns={'index': 'OriginalFile'})
        summary_cols_order = ['OriginalFile', 'original_columns', 'removed_constant', 'removed_high_corr', 'removed_missing', 'categorical_encoded', 'final_columns']
        summary_df = summary_df[[col for col in summary_cols_order if col in summary_df.columns]]
        
        for df_key, df_content in scenario_cleaned_dfs.items():
            cleaned_output_filename = f"{df_key}_cleaned.csv"
            full_output_df_path = os.path.join(output_dir, cleaned_output_filename)
            try:
                df_content.to_csv(full_output_df_path, index=False)
                print(f"  ✅ Salvo: {full_output_df_path}")
            except Exception as e:
                print(f"  ❌ Erro ao salvar {cleaned_output_filename} em {output_dir}: {e}")
        
        summary_output_filename = "summary_preprocessing.csv"
        full_output_summary_path = os.path.join(output_dir, summary_output_filename)
        try:
            summary_df.to_csv(full_output_summary_path, index=False)
            print(f"  📝 Resumo salvo: {full_output_summary_path}")
        except Exception as e:
            print(f"  ❌ Erro ao salvar {summary_output_filename} em {output_dir}: {e}")
        print(f"--- Cenário {scenario_name.upper()} finalizado ---\n")

    print("✅ Pipeline de pré-processamento concluído para todos os cenários.")

if __name__ == '__main__':
    main_pipeline()