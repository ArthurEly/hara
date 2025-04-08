import pandas as pd
import numpy as np
import os
import re
import ast

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
        print(f"Expandindo coluna: {col}")
        new_col = f"{col} (dimensions flattened)"
        idx = df.columns.get_loc(col)
        expanded_series = df[col].apply(clean_listlike_column)
        df.insert(idx + 1, new_col, expanded_series)
    df.drop(columns=cols_to_expand, inplace=True)
    return df

def extract_bitwidth(x):
    match = re.search(r'(UINT|INT)(\d+)', str(x))
    return int(match.group(2)) if match else None

file_paths = [
    "./results_onnx/StreamingFIFO_rtl_merged.csv",
    "./results_onnx/FMPadding_hls_merged.csv",
    "./results_onnx/ConvolutionInputGenerator_hls_merged.csv",
    "./results_onnx/MVAU_hls_merged.csv",
    "./results_onnx/StreamingDataWidthConverter_rtl_merged.csv",
    "./results_onnx/LabelSelect_hls_merged.csv",
    "./results_onnx/StreamingDataWidthConverter_hls_merged.csv"
]

area_cols_to_preserve = [
    "Total LUTs",
    "LUTRAMs",
    "Logic LUTs",
    "FFs",
    "RAMB36",
    "RAMB18",
    "DSP Blocks",
    "ConvKernelDim",
    "Dilation",
    "Stride",
    "SRLs",
    "inWidth",
    "Padding",
    "K",
    "Labels",
    "PE",
    "inFIFODepths",
    "outFIFODepths",
    "numInputVectors",
]

cleaned_dfs = {}
summary = {}

def clean_dataframe(df):
    existing_area_cols = [col for col in area_cols_to_preserve if col in df.columns]

    # Inserir colunas "(bits)" logo após as colunas de tipo de dado
    datatype_cols = [col for col in df.columns if "datatype" in col.lower()]
    for col in datatype_cols:
        print(f"Expandindo bits: {col}")
        new_col = f"{col} (bits)"
        bitwidths = df[col].apply(extract_bitwidth)
        col_idx = df.columns.get_loc(col)
        df.insert(col_idx + 1, new_col, bitwidths)
    
    # Atualiza a lista de colunas a preservar com as colunas de bitwidth geradas
    generated_bit_cols = [f"{col} (bits)" for col in datatype_cols]
    preserve_cols = existing_area_cols + generated_bit_cols

    # Agora podemos remover as colunas originais de datatype
    df.drop(columns=datatype_cols, inplace=True)

    # 1. Remover colunas com valor único, exceto colunas de preservação
    nunique = df.nunique()
    cols_to_drop = [col for col in nunique[nunique <= 1].index if col not in preserve_cols]
    df.drop(columns=cols_to_drop, inplace=True)

    # 2. Tentar converter colunas para numérico onde possível
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # 3. Remover colunas altamente correlacionadas, exceto colunas de preservação
    numeric_df = df.select_dtypes(include='number')
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [
        column for column in upper.columns
        if any(upper[column] > 0.98) and column not in preserve_cols
    ]
    df.drop(columns=to_drop_corr, inplace=True)

    # 4. Remover duplicatas
    df.drop_duplicates(inplace=True)

    # 5. Remover colunas com mais de 50% de valores ausentes, exceto colunas de preservação
    missing_threshold = 0.5
    cols_to_drop_missing = [
        col for col in df.columns
        if df[col].isnull().mean() > missing_threshold and col not in preserve_cols
    ]
    df.drop(columns=cols_to_drop_missing, inplace=True)

    # 6. Remover coluna noActivation, se existir
    for col_to_remove in ["noActivation", "impl_style"]:
        if col_to_remove in df.columns:
            df.drop(columns=[col_to_remove], inplace=True)

    # 7. Renomear colunas IFM/OFM para nomes explícitos
    rename_map = {
        col: col.replace("IFM", "Input Feature Map ").replace("OFM", "Output Feature Map")
        for col in df.columns
        if "IFM" in col or "OFM" in col
    }
    df.rename(columns=rename_map, inplace=True)

    # 8. Expandir colunas com valores list-like
    df = expand_listlike_columns(df)

    return df, {
        "original_columns": len(nunique),
        "removed_constant": len(cols_to_drop),
        "removed_high_corr": len(to_drop_corr),
        "removed_missing": len(cols_to_drop_missing),
        "final_columns": df.shape[1]
    }

# Processar todos os arquivos
for path in file_paths:
    name = os.path.basename(path).replace(".csv", "")
    df = pd.read_csv(path)
    cleaned_df, stats = clean_dataframe(df)
    cleaned_dfs[name] = cleaned_df
    summary[name] = stats

# Criar DataFrame de resumo
summary_df = pd.DataFrame(summary).T
summary_df.index.name = "Operator"
summary_df.reset_index(inplace=True)

print(summary_df)

# Salvar os arquivos limpos
os.makedirs("./results_cleaned", exist_ok=True)
for name, df in cleaned_dfs.items():
    df.to_csv(f"./results_cleaned/{name}_cleaned.csv", index=False)

print("✅ Todos os arquivos limpos foram salvos.")
