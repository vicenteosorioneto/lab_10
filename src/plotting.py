"""
plotting.py — Geração de gráficos profissionais dos resultados de benchmark.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("lab10.plotting")

ASSETS_DIR = Path("assets")


def _setup_style():
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        }
    )
    return plt


def plot_vram_comparison(results: list, model_vram_mb: float):
    """Gráfico de barras: VRAM pico por cenário."""
    plt = _setup_style()
    import numpy as np

    labels = [m.label for m in results]
    vram_peaks = [m.vram_peak_mb for m in results]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vram_peaks, color=colors[: len(labels)], width=0.5, edgecolor="white", linewidth=1.2)

    # Linha de referência: VRAM apenas do modelo
    ax.axhline(model_vram_mb, color="#F39C12", linestyle="--", linewidth=1.5, label=f"Modelo base: {model_vram_mb:.0f} MB")

    for bar, val in zip(bars, vram_peaks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vram_peaks) * 0.015,
            f"{val:.0f} MB",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Comparação de VRAM Pico por Cenário de Inferência", pad=14)
    ax.set_ylabel("VRAM Pico (MB)")
    ax.set_ylim(0, max(vram_peaks) * 1.2 + 50)
    ax.legend(loc="upper right")

    ASSETS_DIR.mkdir(exist_ok=True)
    path = ASSETS_DIR / "vram_comparison.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Gráfico VRAM salvo: {path}")


def plot_generation_time(results: list):
    """Gráfico de barras horizontal: tempo de geração por cenário."""
    plt = _setup_style()

    labels = [m.label for m in results]
    times = [m.elapsed_sec for m in results]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, times, color=colors[: len(labels)], edgecolor="white", linewidth=1.2, height=0.45)

    for bar, val in zip(bars, times):
        ax.text(
            val + max(times) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s",
            va="center", fontsize=10, fontweight="bold",
        )

    ax.set_title("Tempo Total de Geração (100 tokens) por Cenário", pad=14)
    ax.set_xlabel("Tempo (segundos)")
    ax.set_xlim(0, max(times) * 1.25)
    ax.invert_yaxis()

    ASSETS_DIR.mkdir(exist_ok=True)
    path = ASSETS_DIR / "generation_time.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Gráfico de tempo salvo: {path}")


def plot_throughput(results: list):
    """Gráfico de barras: throughput (tokens/s) por cenário."""
    plt = _setup_style()

    labels = [m.label for m in results]
    throughputs = [m.throughput_tok_per_sec for m in results]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        labels, throughputs, color=colors[: len(labels)],
        width=0.5, edgecolor="white", linewidth=1.2,
    )

    for bar, val in zip(bars, throughputs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughputs) * 0.015,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Throughput de Geração (tokens/segundo) por Cenário", pad=14)
    ax.set_ylabel("Tokens / Segundo")
    ax.set_ylim(0, max(throughputs) * 1.25)

    ASSETS_DIR.mkdir(exist_ok=True)
    path = ASSETS_DIR / "throughput.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Gráfico de throughput salvo: {path}")


def generate_all_plots(results: list, model_vram_mb: float):
    """Gera todos os gráficos de uma vez."""
    try:
        plot_vram_comparison(results, model_vram_mb)
        plot_generation_time(results)
        plot_throughput(results)
        logger.info("Todos os gráficos gerados em assets/")
    except Exception as e:
        logger.error(f"Erro ao gerar gráficos: {e}")
        logger.warning("Continuando sem gráficos. Verifique a instalação do matplotlib.")
