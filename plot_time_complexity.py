import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

def plot_time_complexity():
    # Parâmetros
    L = np.arange(1, 11)
    F = 10
    B = 1
    
    # Complexidades
    exhaustive_complexity = F**L * B
    K = L
    hara_complexity = K * B

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(L, exhaustive_complexity, label="Exhaustive Search", color="#FFB000", lw=3)
    ax.plot(L, hara_complexity, label="HARA Framework", color="#FF7C00", lw=3, linestyle="--")

    ax.set_yscale("log")
    ax.set_xlabel("Number of Layers (L)", fontsize=14)
    ax.set_ylabel("Number of Synthesis Builds (log scale)", fontsize=14)

    # Limites do eixo Y
    ax.set_ylim(1, 1e7)  # De 10^1 até 10^7

    # Grades principais (10^0, 10^1, 10^2, etc.)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))

    # Grades secundárias (2*10^x, 3*10^x, ..., 9*10^x)
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=100))
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Legenda
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.92),
        ncol=1,
        fontsize=13,
        frameon=True,
        facecolor='#fff9e6',
        edgecolor='#aaaaaa',
        framealpha=0.8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    # Salvar e mostrar
    plt.savefig("time_complexity_analysis_with_minor_ticks.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    plot_time_complexity()
