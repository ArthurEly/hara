import pandas as pd
import numpy as np
import os
import re
import ast
from datetime import datetime
import shutil # Importado, mas não usado ativamente para rmtree nesta versão

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
    "Total LUTs", "LUTRAMs", "Logic LUTs", "FFs", "RAMB36", "RAMB18", "DSP Blocks",
    "ConvKernelDim", "Dilation", "Stride", "SRLs", "inWidth", "Padding", "K",
    "Labels", "PE", "inFIFODepths", "outFIFODepths", "numInputVectors",
]
# max_resources não é mais usado para normalização, mas pode ser mantido se útil para outros fins
# max_resources = {
# "Total LUTs": 53200, "LUTRAMs": 17400, "Logic LUTs": 53200,
# "FFs": 106400, "RAMB36": 140, "RAMB18": 280, "DSP Blocks": 220
# }


def clean_dataframe(df_input):
    df = df_input.copy()

    existing_area_cols = [col for col in area_cols_to_preserve if col in df.columns]
    cols_to_preserve_current_df = list(existing_area_cols)

    # Inserir colunas "(bits)"
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

    # 2. Tentar converter para numérico
    for col in df.columns:
        if col in df.select_dtypes(include='object').columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

    # 3. Remover colunas altamente correlacionadas
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

    # 5. REMOVIDO: Não remover colunas com >50% de valores ausentes
    # missing_threshold = 0.5
    # cols_to_drop_missing = [
    # col for col in df.columns
    # if df[col].isnull().mean() > missing_threshold and col not in cols_to_preserve_current_df
    # ]
    # df.drop(columns=cols_to_drop_missing, inplace=True, errors='ignore')
    # Manter a contagem para o sumário, mesmo que zero
    cols_to_drop_missing = [] 

    # 6. REMOVIDO: Não remover colunas específicas "noActivation", "impl_style"
    # for col_to_remove in ["noActivation", "impl_style"]:
    # if col_to_remove in df.columns:
    # df.drop(columns=[col_to_remove], inplace=True, errors='ignore')

    # 7. Renomear colunas IFM/OFM
    rename_map = {
        col: col.replace("IFM", "Input Feature Map ").replace("OFM", "Output Feature Map ")
        for col in df.columns
        if "IFM" in col or "OFM" in col
    }
    df.rename(columns=rename_map, inplace=True)

    # 8. Expandir colunas com valores list-like
    df = expand_listlike_columns(df)

    # 9. REMOVIDO: Normalizar colunas de área
    # for col, max_val in max_resources.items():
    # if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
    # df[col] = ((df[col].astype(float) / max_val) * 100).round(4)
    # elif col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
    # print(f"  Aviso: Coluna de recurso '{col}' não é numérica, não pode normalizar.")

    stats = {
        "original_columns": len(nunique),
        "removed_constant": len(cols_to_drop_unique),
        "removed_high_corr": len(cols_to_drop_corr),
        "removed_missing": len(cols_to_drop_missing), # Será 0, mas mantido para consistência do sumário
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
            input_csv_files = [
                f for f in os.listdir(abs_scenario_input_dir)
                if os.path.isfile(os.path.join(abs_scenario_input_dir, f)) and f.endswith(".csv")
            ]
        except FileNotFoundError:
            print(f"⚠️  Aviso: Erro ao listar arquivos em {abs_scenario_input_dir}. Pulando cenário.\n")
            continue
            
        if not input_csv_files:
            print(f"ℹ️  Nenhum arquivo CSV encontrado em {abs_scenario_input_dir}. Pulando cenário.\n")
            continue
        
        print(f"Encontrados {len(input_csv_files)} arquivo(s) CSV para processar: {', '.join(input_csv_files)}")

        # Diretório de saída simplificado dentro de "preprocessed"
        output_dir = os.path.join(abs_scenario_input_dir, "preprocessed")
        
        # Não há mais distinção entre "timestamped_run" e "last_run", apenas um diretório de saída.
        # Se a pasta "preprocessed" deve ser limpa a cada execução, adicione a lógica aqui.
        # Por padrão, os arquivos serão sobrescritos.
        # Exemplo de limpeza (use com cautela):
        # if os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
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
                cleaned_df, processing_stats = clean_dataframe(original_df)
                
                scenario_cleaned_dfs[df_name_key] = cleaned_df
                scenario_summary_stats[df_name_key] = processing_stats
                print(f"  Arquivo '{csv_file}' limpo com sucesso.")
            except Exception as e:
                print(f"  ❌ Erro ao processar o arquivo {csv_file}: {e}")
                # import traceback
                # traceback.print_exc()

        if not scenario_cleaned_dfs:
            print(f"ℹ️  Nenhum DataFrame foi limpo com sucesso para o cenário '{scenario_name}'. Operações de salvamento puladas.\n")
            continue

        summary_df = pd.DataFrame(scenario_summary_stats).T
        summary_df.index.name = "OriginalFile"
        summary_df.reset_index(inplace=True)

        # Salvar DataFrames limpos
        for df_key, df_content in scenario_cleaned_dfs.items():
            cleaned_output_filename = f"{df_key}_cleaned.csv"
            full_output_df_path = os.path.join(output_dir, cleaned_output_filename)
            try:
                df_content.to_csv(full_output_df_path, index=False)
                print(f"  ✅ Salvo: {full_output_df_path}")
            except Exception as e:
                print(f"  ❌ Erro ao salvar {cleaned_output_filename} em {output_dir}: {e}")
        
        # Salvar DataFrame de resumo
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