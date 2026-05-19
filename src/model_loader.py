"""
model_loader.py — Carregamento do modelo LLM com QLoRA (4 bits) e fallbacks.

Suporta:
  - QLoRA 4-bit via bitsandbytes
  - FlashAttention-2 (com fallback automático para SDPA/eager)
  - Modo CPU (sem CUDA)
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger("lab10.model_loader")

# Modelo padrão (pequeno o suficiente para GPUs com ≥ 4 GB VRAM)
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _detect_flash_attention() -> bool:
    """Verifica se flash_attn está instalado e a GPU suporta."""
    try:
        import flash_attn  # noqa: F401
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability()
        # FlashAttention-2 requer compute capability >= 8.0 (Ampere+)
        return cap[0] >= 8
    except ImportError:
        return False


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    """Carrega AutoTokenizer com padding_side correto para geração."""
    from transformers import AutoTokenizer

    logger.info(f"Carregando tokenizador: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model_qlora(
    model_name: str = DEFAULT_MODEL,
    use_flash_attention: bool = True,
    force_cpu: bool = False,
) -> Tuple[object, str]:
    """
    Carrega modelo com QLoRA 4 bits quando CUDA disponível,
    ou em fp32 na CPU como fallback.

    Retorna:
        (model, attn_implementation)
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cuda_ok = torch.cuda.is_available() and not force_cpu

    # --- Configuração de quantização ---
    bnb_config = None
    if cuda_ok:
        try:
            import bitsandbytes  # noqa: F401

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("QLoRA 4-bit configurado (NF4 + double quant).")
        except ImportError:
            logger.warning("bitsandbytes não encontrado. Usando fp16 sem quantização.")

    # --- Seleção de implementação de atenção ---
    attn_impl = "eager"
    if cuda_ok and use_flash_attention:
        if _detect_flash_attention():
            attn_impl = "flash_attention_2"
            logger.info("FlashAttention-2 ativado.")
        else:
            attn_impl = "sdpa"
            logger.warning(
                "FlashAttention-2 indisponível (GPU ou pacote ausente). "
                "Usando SDPA (scaled_dot_product_attention) como fallback."
            )
    elif cuda_ok:
        attn_impl = "sdpa"

    # --- Carregamento do modelo ---
    logger.info(f"Carregando modelo: {model_name} | device={'cuda' if cuda_ok else 'cpu'} | attn={attn_impl}")

    kwargs = dict(
        pretrained_model_name_or_path=model_name,
        attn_implementation=attn_impl,
        trust_remote_code=False,
    )

    if cuda_ok:
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if bnb_config is not None:
            kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.eval()

    device_str = "cuda" if cuda_ok else "cpu"
    logger.info(f"Modelo carregado com sucesso em {device_str}.")
    return model, attn_impl
