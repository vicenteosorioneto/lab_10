"""
utils.py — Utilitários gerais do pipeline RAG/LLM.
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Any

import torch


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("lab10")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cuda_available() -> bool:
    return torch.cuda.is_available()


def bytes_to_mb(b: int) -> float:
    return round(b / (1024 ** 2), 2)


def get_vram_allocated_mb() -> float:
    if not cuda_available():
        return 0.0
    return bytes_to_mb(torch.cuda.memory_allocated())


def get_vram_max_mb() -> float:
    if not cuda_available():
        return 0.0
    return bytes_to_mb(torch.cuda.max_memory_allocated())


def reset_vram_peak():
    if cuda_available():
        torch.cuda.reset_peak_memory_stats()


def save_json(data: Any, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Context manager simples para medir tempo de execução."""

    def __init__(self):
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        if cuda_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if cuda_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
