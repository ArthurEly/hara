import matplotlib
using_agg = False
from platform import python_implementation
if "pypy" in python_implementation().casefold():
    using_agg = True
    matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import math

def show(name: str = 'output.pdf') -> None:
    backend = matplotlib.get_backend().lower()
    if 'agg' in backend or 'pdf' in backend or 'svg' in backend:
        plt.savefig(name, format='pdf')
    else:
        plt.show()

def format_thousands(x, pos):
    # Formatar números maiores que 10.000 com sufixo 'k'
    if x >= 1000:
        return f"{int(x / 1000)}k"
    else:
        return f"{int(x):,}"

def format_fps(fps):
    # Arredondar para baixo e colocar 'k' se maior que 10.000
    if fps >= 1000:
        return f"{math.floor(fps / 100) / 10}k"
    else:
        return f"{math.floor(fps)}"

def generate_bar_graphs(
    x_ticks: list[str],
    hara_values: list[float],
    hara_throughput_values: list[float],
    auto_folding_values: list[float],
    greedy_values: list[float],
    naive_values: list[float],
    auto_folding_throughput_values: list[float],
    greedy_throughput_values: list[float],
    naive_throughput_values: list[float],
    limit_values: list[float],
    colors: list[str],
    x_label: str = 'Configurations',
    y_label: str = 'Value',
    legend_names: list[str] = ['HARA', 'Auto-Folding', 'greedy', 'naive'],
    limit_bar_color: str = 'red',
    file_name: str = 'fpgas.png',
    show_legend: bool = False  
) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))

    FONTSIZE = 34
    LABELSIZE = 34
    width = 0.18
    x = np.arange(len(x_ticks))
    INSIDE_BAR_SIZE = 28

    bars1 = ax.bar(x - 1.5*width, hara_values, width,
                   color=colors[0], edgecolor='black', label=legend_names[0])
    bars2 = ax.bar(x - 0.5*width, auto_folding_values, width,
                   color=colors[1], edgecolor='black', label=legend_names[1])
    bars3 = ax.bar(x + 0.5*width, greedy_values, width,
                   color=colors[2], edgecolor='black', label=legend_names[2])
    # Deixa as barras normais
    bars4 = ax.bar(x + 1.5*width, naive_values, width,
                color=colors[3], edgecolor='black', label=legend_names[3])

    # Agora, destacar os Naives com falha (nos índices 2 e 3)
    failed_indices = [2, 3]
    for idx in failed_indices:
        bar = bars4[idx]
        bar.set_facecolor('red')  # cor de falha
        bar.set_hatch('///')      # textura para reforçar visualmente

    # Depois, adicionar à legenda
    handles = [
        Patch(facecolor=colors[0], edgecolor='black', label=legend_names[0]),
        Patch(facecolor=colors[1], edgecolor='black', label=legend_names[1]),
        Patch(facecolor=colors[2], edgecolor='black', label=legend_names[2]),
        Patch(facecolor=colors[3], edgecolor='black', label=legend_names[3]),
        Patch(facecolor='red', hatch='///', label=f'{legend_names[3]} (Failed)', edgecolor='black'),  # novo item na legenda
        Line2D([], [], color=limit_bar_color, linestyle='--', linewidth=2,
            label='Resource limit')
    ]

    limit_handles = []
    for bar1, bar4, lim in zip(bars1, bars4, limit_values):
        left = bar1.get_x()
        right = bar4.get_x() + bar4.get_width()
        line, = ax.plot([left, right], [lim, lim],
                        linestyle='--', color=limit_bar_color, linewidth=4)
        limit_handles.append(line)

    def annotate_bars(bars, throughput_values, is_hara=False):
        ylim = ax.get_ylim()
        for bar, throughput in zip(bars, throughput_values):
            height = bar.get_height()
            # Usar a função de formatação para FPS
            formatted_fps = format_fps(throughput)
            y = height + 0.02 * (ylim[1] - ylim[0])
            
            # Se for HARA, colocar o texto em negrito
            fontweight = 'bold' if is_hara else 'normal'

            ax.text(bar.get_x() + bar.get_width()/2, y, f"{formatted_fps} FPS",
                    ha='center', va='bottom', fontsize=INSIDE_BAR_SIZE,
                    rotation=90,
                    fontweight=fontweight,  # aplicar negrito se for HARA
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.05', alpha=0.7))

    # Na função `generate_bar_graphs`, modifique a chamada para `annotate_bars`:
    annotate_bars(bars1, hara_throughput_values, is_hara=True)  # HARA em negrito
    annotate_bars(bars2, auto_folding_throughput_values, is_hara=False)
    annotate_bars(bars3, greedy_throughput_values, is_hara=False)
    annotate_bars(bars4, naive_throughput_values, is_hara=False)

    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks, fontsize=FONTSIZE)
    ax.tick_params(axis='y', labelsize=LABELSIZE)
    ax.set_facecolor('none')

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    # Exibir legenda apenas se show_legend for True
    if show_legend:
        fig.legend(
            handles=handles, 
            loc='upper center',
            bbox_to_anchor=(0.547, 0.9),  # centralizado e um pouco abaixo do topo
            ncol=len(handles)/2,
            fontsize=26,
            frameon=True,             
            facecolor='white',     # amarelo bem fraquinho
            edgecolor='#aaaaaa',     # cinza médio para a borda
            framealpha=0.6           
        )
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # ajuste para liberar espaço no topo

    # Adicionar linhas horizontais atrás das barras em cada marcação de Y
    #for y in ax.get_yticks():
    #    ax.axhline(y, color='gray', linestyle='-', linewidth=1, alpha=0.4)
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))

    show(file_name)

max_resources = {
    "Total LUTs": 53200,
    "LUTRAMs": 17400,
    "Logic LUTs": 53200,
    "FFs": 106400,
    "RAMB36": 140,
    "RAMB18": 280,
    "DSP Blocks": 220
}

RESOURCE_LIMITS = {
    "Total LUTs": max_resources["Total LUTs"],
    "FFs": max_resources["FFs"],
    "BRAM (36k)": max_resources["RAMB36"],
    "DSP Blocks": max_resources["DSP Blocks"]
}

def main() -> None:
    x_ticks = ["10% limit", "20% limit", "40% limit", "80% limit"]

    hara_luts = [5270, 10173, 15213, 15213]
    auto_luts = [5970, 8535, 17052, 17052]
    greedy_luts = [4696, 9757, 17872, 17872]  # Exemplo fictício
    naive_luts = [5193, 10226, 10226, 10226]  # Exemplo fictício
    limit_luts = [
        RESOURCE_LIMITS["Total LUTs"]*0.1, 
        RESOURCE_LIMITS["Total LUTs"]*0.2, 
        RESOURCE_LIMITS["Total LUTs"]*0.4, 
        RESOURCE_LIMITS["Total LUTs"]*0.8
    ]

    hara_ffs = [5398, 13167, 19825, 19825]
    auto_ffs = [5653, 11389, 20286, 20286]
    greedy_ffs = [4781, 10491, 17847, 17847]
    naive_ffs = [4987, 10892, 10892, 10892]
    limit_ffs = [
        RESOURCE_LIMITS["FFs"]*0.1, 
        RESOURCE_LIMITS["FFs"]*0.2, 
        RESOURCE_LIMITS["FFs"]*0.4, 
        RESOURCE_LIMITS["FFs"]*0.8
    ]

    hara_brams = [6.5, 6.5, 11, 11]
    auto_brams = [10.5, 8.5, 11, 11]
    greedy_brams = [10.5, 8.5, 13, 13]
    naive_brams = [8.5, 12, 12, 12]
    limit_brams = [
        RESOURCE_LIMITS["BRAM (36k)"]*0.1, 
        RESOURCE_LIMITS["BRAM (36k)"]*0.2, 
        RESOURCE_LIMITS["BRAM (36k)"]*0.4, 
        RESOURCE_LIMITS["BRAM (36k)"]*0.8
    ]

    hara_throughput = [6103.52, 43252.60, 86355.79, 86355.79]
    auto_throughput = [678.16, 24414.06, 86355.79, 86355.79]
    greedy_throughput = [2712.67, 32552.08, 86355.79, 86355.79]
    naive_throughput = [2712.67, 21701.38, 21701.38, 21701.38]

    colors = [
        '#A8E6CF',  # HARA
        '#33A1C9',  # Auto-Folding
        '#FFD700',  # greedy (gold)
        '#FF69B4',  # naive (pink)
    ]
    legend_names = [
        'HARA',
        'Auto-Folding',
        
        # 'Greedy',
        # "Critical-Layer Uniform Expansion",
        "CLUE",
         
        # 'Naive',
        # "Uniform Parallelism Increase",
        "UPI",
    ]
    limit_bar_color = 'red'

    generate_bar_graphs(
        x_ticks,
        hara_luts,
        hara_throughput,
        auto_luts,
        greedy_luts,
        naive_luts,
        auto_throughput,
        greedy_throughput,
        naive_throughput,
        limit_luts,
        colors,
        x_label='Configurations',
        y_label='Total LUTs',
        legend_names=legend_names,
        limit_bar_color=limit_bar_color,
        file_name='fpgas_luts.pdf',
        show_legend=True        
    )

    generate_bar_graphs(
        x_ticks,
        hara_ffs,
        hara_throughput,
        auto_ffs,
        greedy_ffs,
        naive_ffs,
        auto_throughput,
        greedy_throughput,
        naive_throughput,
        limit_ffs,
        colors,
        x_label='Configurations',
        y_label='Total FFs',
        legend_names=legend_names,
        limit_bar_color=limit_bar_color,
        file_name='fpgas_ffs.pdf',
        show_legend=True
    )

    generate_bar_graphs(
        x_ticks,
        hara_brams,
        hara_throughput,
        auto_brams,
        greedy_brams,
        naive_brams,
        auto_throughput,
        greedy_throughput,
        naive_throughput,
        limit_brams,
        colors,
        x_label='Configurations',
        y_label='Total BRAMs',
        legend_names=legend_names,
        limit_bar_color=limit_bar_color,
        file_name='fpgas_brams.pdf',
        show_legend=True
    )
    
if __name__ == "__main__":
    main()