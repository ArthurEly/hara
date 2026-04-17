#!/usr/bin/env python3
"""
analyze_fifo_depths.py — Análise exploratória dos padrões de FIFO depth
Foco no CIFAR10 para entender por que o preditor falha.

v2: adiciona diagnóstico do classificador stage1 e análise em log-scale.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "results", "fifo_depth", "fifo_backpressure_dataset.csv")


def main():
    df = pd.read_csv(DATASET_PATH)

    def get_topo(s):
        if "MNIST" in s:   return "MNIST_TFC"
        if "SAT6" in s:    return "SAT6_T2"
        if "CIFAR10" in s: return "CIFAR10_CNV"
        return "OTHER"
    df["topology"] = df["session"].apply(get_topo)

    cifar = df[df["topology"] == "CIFAR10_CNV"].copy()

    print("="*80)
    print(f" ANÁLISE DE FIFO DEPTHS — CIFAR10_CNV ({len(cifar)} amostras)")
    print("="*80)

    # 1. Distribuição de depths
    print("\n[1] DISTRIBUIÇÃO DE DEPTHS")
    print("-"*50)
    depth_bins = [
        (0,   2,      "depth == 2 (mínimo)"),
        (3,   32,     "depth 3-32"),
        (33,  256,    "depth 33-256"),
        (257, 1024,   "depth 257-1024"),
        (1025,8192,   "depth 1k-8k"),
        (8193,65536,  "depth 8k-64k"),
        (65537,500000,"depth 64k+"),
    ]
    for lo, hi, label in depth_bins:
        n   = len(cifar[(cifar["real_depth"] >= lo) & (cifar["real_depth"] <= hi)])
        pct = n / len(cifar) * 100
        print(f"  {label:<25}: {n:>5} ({pct:>5.1f}%)")

    # 2. Depth por par produtor→consumidor
    print("\n[2] DEPTH MÉDIO POR PAR PRODUTOR → CONSUMIDOR")
    print("-"*80)
    pairs = cifar.groupby(["produtor_op", "consumidor_op"]).agg(
        count       =("real_depth", "size"),
        mean_depth  =("real_depth", "mean"),
        median_depth=("real_depth", "median"),
        min_depth   =("real_depth", "min"),
        max_depth   =("real_depth", "max"),
    ).reset_index().sort_values("mean_depth", ascending=False)

    print(f"  {'Produtor':<35} {'Consumidor':<35} {'N':>5} {'Média':>10} {'Mediana':>8} {'Min':>6} {'Max':>8}")
    print("  " + "-"*108)
    for _, row in pairs.iterrows():
        print(f"  {row['produtor_op']:<35} {row['consumidor_op']:<35} {row['count']:>5} "
              f"{row['mean_depth']:>10.0f} {row['median_depth']:>8.0f} "
              f"{row['min_depth']:>6.0f} {row['max_depth']:>8.0f}")

    # 3. Relação throughput vs depth
    print("\n[3] RELAÇÃO THROUGHPUT E DEPTH")
    print("-"*80)
    cifar["speed_ratio"] = cifar["p_throughput"] / cifar["c_throughput"].clip(lower=1e-12)
    cifar["prod_faster"] = cifar["speed_ratio"] > 1.0
    cifar["cons_faster"] = cifar["speed_ratio"] < 1.0
    cifar["balanced"]    = cifar["speed_ratio"] == 1.0

    for label, mask in [
        ("Produtor mais rápido (speed_ratio > 1)", cifar["prod_faster"]),
        ("Consumidor mais rápido (speed_ratio < 1)", cifar["cons_faster"]),
        ("Balanceado (speed_ratio == 1)", cifar["balanced"]),
    ]:
        subset = cifar[mask]
        if len(subset) == 0:
            print(f"  {label}: 0 amostras"); continue
        print(f"  {label}:")
        print(f"    N = {len(subset)} | depth médio = {subset['real_depth'].mean():.0f} | "
              f"mediana = {subset['real_depth'].median():.0f} | max = {subset['real_depth'].max():.0f}")
        d2    = len(subset[subset["real_depth"] == 2])
        d_gt2 = len(subset[subset["real_depth"] > 2])
        print(f"    depth==2: {d2} ({d2/len(subset)*100:.1f}%) | depth>2: {d_gt2} ({d_gt2/len(subset)*100:.1f}%)")

    # 4. Cycle ratio vs depth
    print("\n[4] CYCLE RATIO vs DEPTH")
    print("-"*80)
    cifar["cycle_ratio_bin"] = pd.cut(
        cifar["cycle_ratio"],
        bins=[0, 0.01, 0.1, 0.5, 1.0, 2.0, 10, 100, 1e10],
        labels=["<0.01","0.01-0.1","0.1-0.5","0.5-1.0","1.0-2.0","2-10","10-100",">100"]
    )
    cr_stats = cifar.groupby("cycle_ratio_bin", observed=True).agg(
        count      =("real_depth", "size"),
        mean_depth =("real_depth", "mean"),
        median_depth=("real_depth","median"),
        pct_depth2 =("real_depth", lambda x: (x == 2).sum() / len(x) * 100),
    ).reset_index()

    print(f"  {'Cycle Ratio':<15} {'N':>6} {'Depth Médio':>12} {'Mediana':>10} {'% depth=2':>10}")
    print("  " + "-"*55)
    for _, row in cr_stats.iterrows():
        print(f"  {str(row['cycle_ratio_bin']):<15} {row['count']:>6} {row['mean_depth']:>12.0f} "
              f"{row['median_depth']:>10.0f} {row['pct_depth2']:>9.1f}%")

    # -----------------------------------------------------------------------
    # [4b] DIAGNÓSTICO STAGE 1 — regra determinística cycle_ratio > 10
    # -----------------------------------------------------------------------
    print("\n[4b] DIAGNÓSTICO STAGE 1 — regra cycle_ratio > 10 & speed_ratio < 1")
    print("-"*80)
    cifar["rule_pred2"] = (cifar["cycle_ratio"] > 10) & (cifar["speed_ratio"] < 1.0)
    cifar["actual_is2"] = cifar["real_depth"] == 2

    tp = ((cifar["rule_pred2"]) & (cifar["actual_is2"])).sum()
    fp = ((cifar["rule_pred2"]) & (~cifar["actual_is2"])).sum()
    tn = ((~cifar["rule_pred2"]) & (~cifar["actual_is2"])).sum()
    fn = ((~cifar["rule_pred2"]) & (cifar["actual_is2"])).sum()

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1        = 2 * precision * recall / max(1e-9, precision + recall)
    coverage  = (tp + fp) / len(cifar)

    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print(f"  Cobertura (% amostras marcadas como depth==2): {coverage*100:.1f}%")
    print(f"  FP: {fp} amostras marcadas como depth==2 mas depth>{2} (erro do preditor)")
    if fp > 0:
        fp_depths = cifar[cifar["rule_pred2"] & ~cifar["actual_is2"]]["real_depth"]
        print(f"  FP depths (max={fp_depths.max():.0f}, median={fp_depths.median():.0f}):")
        print(f"    {sorted(fp_depths.tolist())[:20]}")

    # 5. FIFOs difíceis (depth > 1000)
    print("\n[5] FIFOS COM DEPTH > 1000 (os casos difíceis)")
    print("-"*80)
    big_fifos = cifar[cifar["real_depth"] > 1000].copy()
    print(f"  Total: {len(big_fifos)} FIFOs com depth > 1000\n")
    big_pairs = big_fifos.groupby(["produtor_op", "consumidor_op"]).agg(
        count         =("real_depth", "size"),
        mean_depth    =("real_depth", "mean"),
        mean_cycle_ratio=("cycle_ratio","mean"),
        mean_p_thr    =("p_throughput","mean"),
        mean_c_thr    =("c_throughput","mean"),
        mean_tensor_vol=("tensor_volume","mean"),
    ).reset_index().sort_values("count", ascending=False)

    print(f"  {'Produtor→Consumidor':<50} {'N':>5} {'Depth':>8} {'CycleR':>8} {'TensorVol':>10}")
    print("  " + "-"*85)
    for _, row in big_pairs.iterrows():
        pair = f"{row['produtor_op']}→{row['consumidor_op']}"
        print(f"  {pair:<50} {row['count']:>5} {row['mean_depth']:>8.0f} "
              f"{row['mean_cycle_ratio']:>8.3f} {row['mean_tensor_vol']:>10.0f}")

    # 6. Correlação com real_depth (linear e log)
    print("\n[6] CORRELAÇÃO DAS FEATURES COM real_depth (linear e log)")
    print("-"*70)
    cifar["log_real_depth"] = np.log1p(cifar["real_depth"])

    numeric_cols = [
        "dataType_bits", "tensor_volume", "produtor_PE", "produtor_cycles",
        "p_throughput", "p_transfers", "consumidor_SIMD", "consumidor_cycles",
        "c_throughput", "c_transfers", "parallelism_mismatch",
        "cycle_ratio", "theoretical_accumulation", "theoretical_fifo_depth",
        "chain_length",
        "tensor_H", "tensor_W", "tensor_C", "tensor_spatial",
        "comp_produtor_PE", "comp_produtor_SIMD", "comp_produtor_cycles", "comp_p_throughput",
        "comp_consumidor_PE", "comp_consumidor_SIMD", "comp_consumidor_cycles", "comp_c_throughput",
        "comp_cycle_ratio", "comp_theo_accumulation", "comp_theo_depth",
        "comp_parallelism_mismatch",
        "p_IFMDim", "p_OFMDim", "p_KernelDim", "p_Stride",
        "p_IFMChannels", "p_OFMChannels",
        "c_IFMDim", "c_OFMDim", "c_KernelDim", "c_IFMChannels",
        "window_volume", "speed_ratio",
    ]
    numeric_cols = [c for c in numeric_cols if c in cifar.columns]

    corr_raw = cifar[numeric_cols + ["real_depth"]].corr()["real_depth"].drop("real_depth").fillna(0)
    corr_log = cifar[numeric_cols + ["log_real_depth"]].corr()["log_real_depth"].drop("log_real_depth").fillna(0)

    combined = pd.DataFrame({"corr_raw": corr_raw, "corr_log": corr_log})
    combined = combined.reindex(combined["corr_log"].abs().sort_values(ascending=False).index)

    print(f"  {'Feature':<35} {'Corr(raw)':>10} {'Corr(log)':>10}  bar(log)")
    print("  " + "-"*75)
    for feat, row in combined.iterrows():
        bar  = "█" * int(abs(row["corr_log"]) * 30)
        sign = "+" if row["corr_log"] > 0 else "-"
        print(f"  {feat:<35} {row['corr_raw']:>+10.4f} {row['corr_log']:>+10.4f}  {sign}{bar}")

    # 7. Theoretical vs real depth
    print("\n[7] THEORETICAL vs REAL DEPTH (original)")
    print("-"*60)
    cifar_big = cifar[cifar["real_depth"] > 2].copy()
    if len(cifar_big) > 0:
        cifar_big["theo_error"] = cifar_big["theoretical_fifo_depth"] - cifar_big["real_depth"]
        cifar_big["theo_ratio"] = cifar_big["theoretical_fifo_depth"] / cifar_big["real_depth"].clip(lower=1)
        print(f"  Amostras com depth > 2: {len(cifar_big)}")
        print(f"  Erro médio (theo - real): {cifar_big['theo_error'].mean():.0f}")
        print(f"  Ratio médio (theo/real):  {cifar_big['theo_ratio'].mean():.3f}")
        print(f"  Ratio mediano:            {cifar_big['theo_ratio'].median():.3f}")

    if "comp_theo_depth" in cifar.columns:
        print("\n[7b] COMP THEORETICAL vs REAL DEPTH (olhando através de DWC)")
        print("-"*60)
        if len(cifar_big) > 0:
            cifar_big["comp_theo_error"] = cifar_big["comp_theo_depth"] - cifar_big["real_depth"]
            cifar_big["comp_theo_ratio"] = cifar_big["comp_theo_depth"] / cifar_big["real_depth"].clip(lower=1)
            print(f"  Amostras com depth > 2: {len(cifar_big)}")
            print(f"  Erro médio (comp_theo - real): {cifar_big['comp_theo_error'].mean():.0f}")
            print(f"  Ratio médio (comp_theo/real):  {cifar_big['comp_theo_ratio'].mean():.3f}")
            print(f"  Ratio mediano:                 {cifar_big['comp_theo_ratio'].median():.3f}")

    # 8. Dimensão espacial vs depth
    if "tensor_spatial" in cifar.columns:
        print("\n[8] DIMENSÃO ESPACIAL vs DEPTH")
        print("-"*80)
        spatial_bins = [
            (1,    1,     "1×1 (FC layers)"),
            (2,    9,     "≤3×3"),
            (10,   49,    "4×4 - 7×7"),
            (50,   256,   "8×8 - 16×16"),
            (257,  1024,  "17×17 - 32×32"),
            (1025, 100000,">32×32"),
        ]
        print(f"  {'Spatial Area':<20} {'N':>6} {'Depth Médio':>12} {'Mediana':>10} {'Max':>10} {'% d=2':>8}")
        print("  " + "-"*70)
        for lo, hi, label in spatial_bins:
            subset = cifar[(cifar["tensor_spatial"] >= lo) & (cifar["tensor_spatial"] <= hi)]
            if len(subset) == 0: continue
            d2_pct = (subset["real_depth"] == 2).sum() / len(subset) * 100
            print(f"  {label:<20} {len(subset):>6} {subset['real_depth'].mean():>12.0f} "
                  f"{subset['real_depth'].median():>10.0f} {subset['real_depth'].max():>10.0f} "
                  f"{d2_pct:>7.1f}%")

        print(f"\n  Foco: SWG→DWC — tensor_spatial vs depth:")
        swg_dwc = cifar[
            (cifar["produtor_op"] == "ConvolutionInputGenerator_rtl") &
            (cifar["consumidor_op"] == "StreamingDataWidthConverter_rtl")
        ]
        if len(swg_dwc) > 0:
            for lo, hi, label in spatial_bins:
                sub = swg_dwc[(swg_dwc["tensor_spatial"] >= lo) & (swg_dwc["tensor_spatial"] <= hi)]
                if len(sub) == 0: continue
                d2_pct = (sub["real_depth"] == 2).sum() / len(sub) * 100
                print(f"    {label:<20} N={len(sub):>4} | depth médio={sub['real_depth'].mean():>8.0f} "
                      f"| mediana={sub['real_depth'].median():>8.0f} | %d=2: {d2_pct:.0f}%")

    # -----------------------------------------------------------------------
    # [9] ANÁLISE LOG-SCALE: onde o modelo ganha mais com log-transform
    # -----------------------------------------------------------------------
    print("\n[9] ANÁLISE LOG-SCALE — distribuição de log1p(depth)")
    print("-"*60)
    log_depths = np.log1p(cifar["real_depth"])
    log_bins = [0, 1, 2, 3, 5, 7, 9, 12, 15]
    print(f"  {'log1p(depth) bin':<20} {'N':>6} {'% total':>8}")
    print("  " + "-"*38)
    for i in range(len(log_bins) - 1):
        lo_b, hi_b = log_bins[i], log_bins[i+1]
        n    = ((log_depths >= lo_b) & (log_depths < hi_b)).sum()
        pct  = n / len(cifar) * 100
        d_lo = int(np.expm1(lo_b))
        d_hi = int(np.expm1(hi_b))
        print(f"  [{lo_b:>4},{hi_b:>4}) → [{d_lo:>7},{d_hi:>7}) {n:>6} {pct:>7.1f}%")

    print(f"\n  log-scale stats:")
    print(f"    mean   = {log_depths.mean():.3f}  (depth ≈ {np.expm1(log_depths.mean()):.0f})")
    print(f"    median = {log_depths.median():.3f}  (depth ≈ {np.expm1(log_depths.median()):.0f})")
    print(f"    std    = {log_depths.std():.3f}")
    print(f"\n  >> Treinar com log1p(target) torna a distribuição muito mais uniforme.")
    print(f"     Na escala linear, o modelo otimiza MAE/RMSE dominado pelo max={cifar['real_depth'].max():.0f}.")
    print(f"     Na escala log, todos os bins acima têm representação similar.")

    # -----------------------------------------------------------------------
    # [10] FEATURES ENGENHEIRADAS v2 — correlação com log_real_depth
    # -----------------------------------------------------------------------
    print("\n[10] FEATURES ENGENHEIRADAS v2 — correlação com log1p(depth)")
    print("-"*60)

    cifar_eng = cifar.copy()
    if "p_throughput" in cifar_eng.columns and "c_throughput" in cifar_eng.columns:
        cifar_eng["speed_ratio"]     = cifar_eng["p_throughput"] / cifar_eng["c_throughput"].clip(lower=1e-12)
        cifar_eng["log_speed_ratio"] = np.log1p(cifar_eng["speed_ratio"].clip(lower=0))

    if "chain_length" in cifar_eng.columns and "consumidor_cycles" in cifar_eng.columns:
        cifar_eng["chain_x_cycles"]  = np.log1p(
            cifar_eng["chain_length"].fillna(0) * cifar_eng["consumidor_cycles"].fillna(0)
        )
    if "tensor_spatial" in cifar_eng.columns and "cycle_ratio" in cifar_eng.columns:
        cifar_eng["spatial_x_cycle"] = np.log1p(
            cifar_eng["tensor_spatial"].fillna(0) * cifar_eng["cycle_ratio"].fillna(0).clip(lower=0)
        )
    if "tensor_spatial" in cifar_eng.columns:
        cifar_eng["is_spatial_large"] = (cifar_eng["tensor_spatial"] > 256).astype(int)

    for col in ["cycle_ratio", "chain_length", "consumidor_cycles", "theoretical_fifo_depth"]:
        if col in cifar_eng.columns:
            cifar_eng[f"log_{col}"] = np.log1p(cifar_eng[col].fillna(0).clip(lower=0))

    eng_cols = [
        "speed_ratio", "log_speed_ratio", "chain_x_cycles", "spatial_x_cycle",
        "is_spatial_large", "log_cycle_ratio", "log_chain_length",
        "log_consumidor_cycles", "log_theoretical_fifo_depth",
    ]
    eng_cols = [c for c in eng_cols if c in cifar_eng.columns]

    corr_eng = cifar_eng[eng_cols + ["log_real_depth"]].corr()["log_real_depth"].drop("log_real_depth").fillna(0)
    corr_eng = corr_eng.sort_values(ascending=False)

    for feat, corr in corr_eng.items():
        bar  = "█" * int(abs(corr) * 40)
        sign = "+" if corr > 0 else "-"
        print(f"  {feat:<30} {sign}{abs(corr):.4f}  {bar}")


if __name__ == "__main__":
    main()