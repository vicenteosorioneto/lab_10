"""
benchmark.py — Orquestra os três cenários de benchmark e gera o relatório comparativo.

Cenários:
  1. Sem KV Cache (use_cache=False) — baseline
  2. Com KV Cache (use_cache=True)
  3. Com FlashAttention-2 (se disponível, senão SDPA)
"""

import logging
from typing import List, Optional

import torch

from src.metrics import InferenceMetrics, run_inference_benchmark
from src.utils import save_json, cuda_available, bytes_to_mb

logger = logging.getLogger("lab10.benchmark")

MAX_NEW_TOKENS = 100
BENCHMARK_OUTPUT = "benchmark_results.json"


def _safe_input_ids(tokenizer, prompt: str, max_length: int = 512) -> torch.Tensor:
    """Tokeniza e trunca o prompt para caber no modelo sem OOM."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"]


def run_all_benchmarks(
    model,
    tokenizer,
    prompt: str,
    attn_implementation: str,
    max_input_tokens: int = 512,
) -> List[InferenceMetrics]:
    """
    Executa todos os cenários de benchmark em sequência e retorna lista de métricas.
    """
    results: List[InferenceMetrics] = []
    input_ids = _safe_input_ids(tokenizer, prompt, max_length=max_input_tokens)

    logger.info(f"Tokens de entrada: {input_ids.shape[1]}")
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # Cenário 1: SEM KV Cache
    # -----------------------------------------------------------------
    logger.info("CENÁRIO 1: Inferência SEM KV Cache")
    logger.info(
        "  Gargalo: a cada token gerado, QKV é recalculado para TODA a sequência. "
        "Complexidade O(n²) em atenção. Alta latência e VRAM volátil."
    )
    m1 = run_inference_benchmark(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=False,
        label="Sem KV Cache",
        attn_implementation=attn_implementation,
    )
    results.append(m1)

    # -----------------------------------------------------------------
    # Cenário 2: COM KV Cache
    # -----------------------------------------------------------------
    logger.info("CENÁRIO 2: Inferência COM KV Cache")
    logger.info(
        "  Otimização: K e V são calculados apenas uma vez por token e armazenados. "
        "O decoder auto-regressivo reutiliza o cache → redução drástica de FLOPs."
    )
    m2 = run_inference_benchmark(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        label="Com KV Cache",
        attn_implementation=attn_implementation,
    )
    results.append(m2)

    # -----------------------------------------------------------------
    # Cenário 3: FlashAttention / SDPA (já configurado no carregamento)
    # -----------------------------------------------------------------
    label_fa = (
        "FlashAttention-2"
        if attn_implementation == "flash_attention_2"
        else f"SDPA ({attn_implementation})"
    )
    logger.info(f"CENÁRIO 3: Inferência com {label_fa} + KV Cache")
    logger.info(
        "  FlashAttention-2 recomputa atenção em blocos (tiling SRAM), eliminando "
        "a materialização da matriz NxN de atenção na HBM — reduz VRAM O(n²) → O(n)."
    )
    m3 = run_inference_benchmark(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        label=label_fa,
        attn_implementation=attn_implementation,
    )
    results.append(m3)

    return results


def build_report(
    results: List[InferenceMetrics],
    model_vram_mb: float,
    total_context_tokens: int,
) -> dict:
    """Constrói dicionário completo de resultados para serialização JSON."""
    report = {
        "model_vram_loaded_mb": round(model_vram_mb, 2),
        "total_context_tokens": total_context_tokens,
        "max_new_tokens": MAX_NEW_TOKENS,
        "scenarios": [m.to_dict() for m in results],
        "comparison": _compute_comparison(results),
    }
    save_json(report, BENCHMARK_OUTPUT)
    logger.info(f"Relatório salvo em {BENCHMARK_OUTPUT}")
    return report


def _compute_comparison(results: List[InferenceMetrics]) -> dict:
    if len(results) < 2:
        return {}
    base = results[0]
    comparison = {}
    for m in results[1:]:
        speedup = base.elapsed_sec / m.elapsed_sec if m.elapsed_sec > 0 else 0
        vram_delta = m.vram_peak_mb - base.vram_peak_mb
        comparison[f"{base.label} vs {m.label}"] = {
            "speedup_x": round(speedup, 2),
            "vram_delta_mb": round(vram_delta, 2),
            "throughput_gain_pct": round(
                (m.throughput_tok_per_sec - base.throughput_tok_per_sec)
                / max(base.throughput_tok_per_sec, 1e-9) * 100, 1
            ),
        }
    return comparison


def print_comparison_table(results: List[InferenceMetrics]):
    """Imprime tabela comparativa formatada no terminal."""
    header = f"{'Cenário':<25} {'Tokens':>8} {'Tempo(s)':>10} {'VRAM pico(MB)':>14} {'Throughput(tok/s)':>18}"
    sep = "-" * len(header)
    logger.info("\n" + sep)
    logger.info(header)
    logger.info(sep)
    for m in results:
        logger.info(
            f"{m.label:<25} {m.tokens_generated:>8} {m.elapsed_sec:>10.3f} "
            f"{m.vram_peak_mb:>14.1f} {m.throughput_tok_per_sec:>18.1f}"
        )
    logger.info(sep)
