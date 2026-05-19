"""
main.py — Pipeline completo: QLoRA + RAG + KV Cache + FlashAttention Benchmark

Partes deste laboratório foram geradas/complementadas com IA,
revisadas e validadas por Antonio Vicente da Costa Osorio Neto.

Execução:
    python main.py [--model MODEL_ID] [--cpu] [--no-flash]
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Configuração de logging antes de qualquer import local
# ---------------------------------------------------------------------------
from src.utils import setup_logging

logger = setup_logging("INFO")


def parse_args():
    p = argparse.ArgumentParser(description="Lab 10 — LLM Optimization Benchmark")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HuggingFace model ID")
    p.add_argument("--cpu", action="store_true", help="Força execução em CPU")
    p.add_argument("--no-flash", action="store_true", help="Desativa FlashAttention-2")
    p.add_argument("--max-input-tokens", type=int, default=512,
                   help="Número máximo de tokens de entrada (evita OOM)")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 65)
    logger.info("  LAB 10 — RAG + QLoRA + KV Cache + FlashAttention Benchmark")
    logger.info("=" * 65)
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability()
        logger.info(f"Compute capability: {cap[0]}.{cap[1]}")

    # ------------------------------------------------------------------
    # PASSO 1 — Carregar modelo com QLoRA
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 1] Carregando modelo com QLoRA 4-bit...")
    from src.model_loader import load_tokenizer, load_model_qlora
    from src.utils import get_vram_allocated_mb, reset_vram_peak

    reset_vram_peak()
    vram_before = get_vram_allocated_mb()

    tokenizer = load_tokenizer(args.model)
    model, attn_impl = load_model_qlora(
        model_name=args.model,
        use_flash_attention=not args.no_flash,
        force_cpu=args.cpu,
    )

    vram_after_model = get_vram_allocated_mb()
    model_vram_mb = max(vram_after_model - vram_before, 0.0)
    logger.info(f"VRAM alocada pelo modelo: {model_vram_mb:.2f} MB")
    logger.info(f"Implementação de atenção: {attn_impl}")

    # ------------------------------------------------------------------
    # PASSO 2 — Gerar corpus médico RAG massivo
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 2] Gerando corpus RAG massivo (10.000–15.000 tokens)...")
    from src.rag_simulator import generate_medical_corpus, build_rag_prompt

    corpus = generate_medical_corpus(target_chars=55_000)
    logger.info(f"Corpus gerado: {len(corpus):,} caracteres")

    # ------------------------------------------------------------------
    # PASSO 3 — Tokenizar e contar tokens do contexto completo
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 3] Tokenizando corpus completo...")
    full_enc = tokenizer(corpus, return_tensors="pt", truncation=False)
    total_tokens = full_enc["input_ids"].shape[1]
    logger.info(f"Total de tokens no corpus: {total_tokens:,}")

    # ------------------------------------------------------------------
    # PASSO 4 — Montar prompt RAG
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 4] Construindo prompt RAG com contexto truncado...")
    question = (
        "Quais são os principais mecanismos fisiopatológicos da insuficiência cardíaca "
        "e como os inibidores SGLT2 interferem na progressão da doença renal crônica?"
    )
    rag_prompt = build_rag_prompt(question, corpus, max_context_chars=10_000)
    prompt_enc = tokenizer(rag_prompt, return_tensors="pt", truncation=False)
    prompt_tokens = prompt_enc["input_ids"].shape[1]
    logger.info(f"Tokens do prompt RAG (antes do truncamento de entrada): {prompt_tokens:,}")

    # ------------------------------------------------------------------
    # PASSO 5–7 — Benchmarks: sem cache / com cache / FlashAttention
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 5-7] Executando benchmarks de inferência...")
    from src.benchmark import run_all_benchmarks, build_report, print_comparison_table

    results = run_all_benchmarks(
        model=model,
        tokenizer=tokenizer,
        prompt=rag_prompt,
        attn_implementation=attn_impl,
        max_input_tokens=args.max_input_tokens,
    )

    # ------------------------------------------------------------------
    # PASSO 8 — Tabela comparativa e relatório JSON
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 8] Resultados comparativos:")
    print_comparison_table(results)
    report = build_report(results, model_vram_mb, total_tokens)

    # ------------------------------------------------------------------
    # PASSO 9 — Gráficos
    # ------------------------------------------------------------------
    logger.info("\n[PASSO 9] Gerando gráficos...")
    from src.plotting import generate_all_plots

    generate_all_plots(results, model_vram_mb)

    # ------------------------------------------------------------------
    # Sumário final
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("  BENCHMARK CONCLUÍDO")
    logger.info("=" * 65)
    logger.info(f"  Relatório JSON : benchmark_results.json")
    logger.info(f"  Gráficos       : assets/")
    logger.info(f"  VRAM do modelo : {model_vram_mb:.1f} MB")
    logger.info(f"  Tokens corpus  : {total_tokens:,}")
    if results:
        best = max(results, key=lambda m: m.throughput_tok_per_sec)
        logger.info(f"  Melhor throughput: {best.label} — {best.throughput_tok_per_sec:.1f} tok/s")
    logger.info("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
