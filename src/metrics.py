"""
metrics.py — Coleta e estruturação de métricas de inferência.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional

import torch

from src.utils import Timer, get_vram_allocated_mb, get_vram_max_mb, reset_vram_peak, cuda_available

logger = logging.getLogger("lab10.metrics")


@dataclass
class InferenceMetrics:
    label: str
    tokens_generated: int
    elapsed_sec: float
    vram_start_mb: float
    vram_peak_mb: float
    throughput_tok_per_sec: float
    attn_implementation: str
    use_cache: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def print_summary(self):
        logger.info(
            f"[{self.label}] "
            f"tokens={self.tokens_generated} | "
            f"tempo={self.elapsed_sec:.3f}s | "
            f"throughput={self.throughput_tok_per_sec:.1f} tok/s | "
            f"VRAM_pico={self.vram_peak_mb:.1f} MB"
        )


def run_inference_benchmark(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool,
    label: str,
    attn_implementation: str,
) -> InferenceMetrics:
    """
    Executa uma geração completa e coleta métricas de tempo e VRAM.
    """
    device = next(model.parameters()).device

    # Garante que os ids estejam no dispositivo correto
    if input_ids.device != device:
        input_ids = input_ids.to(device)

    # Configura cache no modelo
    model.config.use_cache = use_cache

    # Aquece CUDA (evita overhead de alocação no primeiro forward)
    if cuda_available():
        with torch.no_grad():
            _ = model(input_ids[:, :1])
        torch.cuda.synchronize()

    reset_vram_peak()
    vram_start = get_vram_allocated_mb()

    logger.info(f"Iniciando benchmark [{label}] | use_cache={use_cache} | new_tokens={max_new_tokens}")

    with torch.no_grad(), Timer() as t:
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            do_sample=False,          # greedy para reproducibilidade
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = t.elapsed
    vram_peak = get_vram_max_mb()
    tokens_gen = outputs.shape[1] - input_ids.shape[1]
    throughput = tokens_gen / elapsed if elapsed > 0 else 0.0

    metrics = InferenceMetrics(
        label=label,
        tokens_generated=tokens_gen,
        elapsed_sec=round(elapsed, 4),
        vram_start_mb=vram_start,
        vram_peak_mb=round(vram_peak, 2),
        throughput_tok_per_sec=round(throughput, 2),
        attn_implementation=attn_implementation,
        use_cache=use_cache,
    )
    metrics.print_summary()
    return metrics
