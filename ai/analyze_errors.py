import os
import re

import pandas as pd


def extract_topology(build_name):
    match = re.search(r"^(.*?)_202\d", build_name)
    if match:
        return match.group(1)
    return build_name.split("/")[0]


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hara_validation_results.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Arquivo não encontrado: {csv_path}")
        return

    metric_cols = ["Err%_LUTs", "Err%_FFs", "Err%_BRAM"]
    metric_cols_with_dsp = metric_cols + (["Err%_DSP"] if "Err%_DSP" in df.columns else [])

    df["Erro_Medio"] = df[metric_cols].mean(axis=1)
    df["Erro_Medio_Com_DSP"] = df[metric_cols_with_dsp].mean(axis=1)
    df["Topology"] = df["Build Name"].apply(extract_topology)

    grp = df.groupby("Topology")
    agg_cols = metric_cols_with_dsp + ["Erro_Medio", "Erro_Medio_Com_DSP"]
    topology_stats = grp[agg_cols].mean().reset_index()
    topology_stats["Count"] = grp.size().values
    topology_stats = topology_stats.sort_values(by="Erro_Medio")

    print("\n" + "=" * 90)
    print("📊 MAPE MÉDIO POR TOPOLOGIA")
    print("=" * 90)
    for _, row in topology_stats.iterrows():
        line = (
            f"📌 {row['Topology']:<15} | Builds: {int(row['Count']):<3} "
            f"| Erro Médio: {row['Erro_Medio']:>6.2f}%"
        )
        if "Err%_DSP" in topology_stats.columns:
            line += f" | Erro Médio c/ DSP: {row['Erro_Medio_Com_DSP']:>6.2f}%"
        print(line)

        detail = (
            f"   LUTs: {row['Err%_LUTs']:>6.2f}%"
            f" | FFs: {row['Err%_FFs']:>6.2f}%"
            f" | BRAM: {row['Err%_BRAM']:>6.2f}%"
        )
        if "Err%_DSP" in topology_stats.columns:
            detail += f" | DSP: {row['Err%_DSP']:>6.2f}%"
        print(detail)
        print("-" * 90)

    df_sorted = df.sort_values(by="Erro_Medio")

    print("\n" + "=" * 90)
    print("🏆 TOP 5 MELHORES PREDIÇÕES")
    print("=" * 90)
    for _, row in df_sorted.head(5).iterrows():
        print(f"🔹 {row['Build Name']}")
        print(f"   LUTs: Real {row['Real_LUTs']:>8.0f} | Pred {row['Pred_LUTs']:>8.0f} (Erro: {row['Err%_LUTs']:>6.2f}%)")
        print(f"   FFs : Real {row['Real_FFs']:>8.0f} | Pred {row['Pred_FFs']:>8.0f} (Erro: {row['Err%_FFs']:>6.2f}%)")
        print(f"   BRAM: Real {row['Real_BRAM']:>8.1f} | Pred {row['Pred_BRAM']:>8.1f} (Erro: {row['Err%_BRAM']:>6.2f}%)")
        if "Err%_DSP" in df.columns:
            print(f"   DSP : Real {row['Real_DSP']:>8.0f} | Pred {row['Pred_DSP']:>8.0f} (Erro: {row['Err%_DSP']:>6.2f}%)")
        print("-" * 60)

    print("\n" + "=" * 90)
    print("🚨 TOP 5 PIORES PREDIÇÕES")
    print("=" * 90)
    for _, row in df_sorted.tail(5).sort_values(by="Erro_Medio", ascending=False).iterrows():
        print(f"🔻 {row['Build Name']}")
        print(f"   LUTs: Real {row['Real_LUTs']:>8.0f} | Pred {row['Pred_LUTs']:>8.0f} (Erro: {row['Err%_LUTs']:>6.2f}%)")
        print(f"   FFs : Real {row['Real_FFs']:>8.0f} | Pred {row['Pred_FFs']:>8.0f} (Erro: {row['Err%_FFs']:>6.2f}%)")
        print(f"   BRAM: Real {row['Real_BRAM']:>8.1f} | Pred {row['Pred_BRAM']:>8.1f} (Erro: {row['Err%_BRAM']:>6.2f}%)")
        if "Err%_DSP" in df.columns:
            print(f"   DSP : Real {row['Real_DSP']:>8.0f} | Pred {row['Pred_DSP']:>8.0f} (Erro: {row['Err%_DSP']:>6.2f}%)")
        print("-" * 60)


if __name__ == "__main__":
    main()