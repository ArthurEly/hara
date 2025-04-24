import pandas as pd
from pathlib import Path

# Função a ser testada
def check_resource_usage(csv_path, limits):
    df = pd.read_csv(csv_path)
    last_row = df.tail(1).to_dict(orient='records')[0]
    diffs = {}

    for res, limit in limits.items():
        used = last_row.get(res, 0)
        if pd.isna(used):
            used = 0
        diff = limit - used
        diffs[res] = diff
        print(f"[🧮] {res}: usado={used}, limite={limit}, restante={diff}")
    
    return diffs

# --- Criação de um CSV fictício ---
csv_path = "fake_summary.csv"
data = {
    "HW Name": ["test_hw"],
    "Total LUTs": [280000],
    "FFs": [500000],
    "BRAM (36k)": [480],
    "DSP Blocks": [850],  # este aqui vai estourar o limite
}

df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)

# --- Limites ---
limits = {
    "Total LUTs": 300000,
    "FFs": 600000,
    "BRAM (36k)": 500,
    "DSP Blocks": 800
}

# --- Teste ---
print("\n[🔎] Rodando check_resource_usage...")
resource_diffs = check_resource_usage(csv_path, limits)

# --- Verificação de resultado ---
print("\n[📊] Resultado final:")
for res, diff in resource_diffs.items():
    status = "✅ OK" if diff >= 0 else f"❌ Excedeu em {-diff}"
    print(f"  {res}: {diff} ({status})")

# --- Limpeza (opcional) ---
Path(csv_path).unlink()
