import pandas as pd
import re

def extract_topology(build_name):
    """
    Extrai a topologia base do nome do build.
    Exemplo: 'SAT6_T2W8_2026-04-04_21-27-41/run31_optimized' -> 'SAT6_T2W8'
    """
    match = re.search(r"^(.*?)_202\d", build_name)
    if match:
        return match.group(1)
    return build_name.split('/')[0] # Fallback caso não tenha a data no formato esperado

def main():
    try:
        df = pd.read_csv('hara_validation_results.csv')
    except FileNotFoundError:
        print("[!] Arquivo hara_validation_results.csv não encontrado.")
        return

    # Criar uma métrica de erro médio (ignorando DSPs) para ranqueamento
    df['Erro_Medio'] = df[['Err%_LUTs', 'Err%_FFs', 'Err%_BRAM']].mean(axis=1)
    
    # 1. NOVA SEÇÃO: Análise por Topologia
    df['Topology'] = df['Build Name'].apply(extract_topology)
    
    # Agrupar e calcular a média e a contagem
    grp = df.groupby('Topology')
    topology_stats = grp[['Err%_LUTs', 'Err%_FFs', 'Err%_BRAM', 'Erro_Medio']].mean().reset_index()
    topology_stats['Count'] = grp.size().values
    
    # Ordenar pelas topologias com o menor erro geral
    topology_stats = topology_stats.sort_values(by='Erro_Medio')

    print("\n" + "="*80)
    print("📊 MAPE MÉDIO POR TOPOLOGIA (Desempenho do Modelo)")
    print("="*80)
    for _, row in topology_stats.iterrows():
        print(f"📌 {row['Topology']:<15} | Builds: {int(row['Count']):<3} | Erro Médio Geral: {row['Erro_Medio']:>5.1f}%")
        print(f"   LUTs: {row['Err%_LUTs']:>5.1f}%  |  FFs: {row['Err%_FFs']:>5.1f}%  |  BRAM: {row['Err%_BRAM']:>5.1f}%")
        print("-" * 80)

    # 2. SEÇÃO ORIGINAL: Top 5 e Piores 5
    df_sorted = df.sort_values(by='Erro_Medio')

    print("\n" + "="*80)
    print("🏆 TOP 5 MELHORES PREDIÇÕES (Mais próximas da realidade)")
    print("="*80)
    best = df_sorted.head(5)
    for _, row in best.iterrows():
        print(f"🔹 {row['Build Name']}")
        print(f"   LUTs: Real {row['Real_LUTs']:>6.0f} | Pred {row['Pred_LUTs']:>6.0f}  (Erro: {row['Err%_LUTs']:>6.2f}%)")
        print(f"   FFs : Real {row['Real_FFs']:>6.0f} | Pred {row['Pred_FFs']:>6.0f}  (Erro: {row['Err%_FFs']:>6.2f}%)")
        print(f"   BRAM: Real {row['Real_BRAM']:>6.1f} | Pred {row['Pred_BRAM']:>6.1f}  (Erro: {row['Err%_BRAM']:>6.2f}%)")
        print("-" * 50)

    print("\n" + "="*80)
    print("🚨 TOP 5 PIORES PREDIÇÕES (Maior discrepância)")
    print("="*80)
    worst = df_sorted.tail(5).sort_values(by='Erro_Medio', ascending=False)
    for _, row in worst.iterrows():
        print(f"🔻 {row['Build Name']}")
        print(f"   LUTs: Real {row['Real_LUTs']:>6.0f} | Pred {row['Pred_LUTs']:>6.0f}  (Erro: {row['Err%_LUTs']:>6.2f}%)")
        print(f"   FFs : Real {row['Real_FFs']:>6.0f} | Pred {row['Pred_FFs']:>6.0f}  (Erro: {row['Err%_FFs']:>6.2f}%)")
        print(f"   BRAM: Real {row['Real_BRAM']:>6.1f} | Pred {row['Pred_BRAM']:>6.1f}  (Erro: {row['Err%_BRAM']:>6.2f}%)")
        print("-" * 50)

if __name__ == "__main__":
    main()