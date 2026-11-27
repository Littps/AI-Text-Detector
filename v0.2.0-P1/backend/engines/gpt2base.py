"""Shared GPT-2 causal-LM base for zero-shot method engines.

gltr and fast-detectgpt both reason over the *same* GPT-2 small base
(models/gpt2, probed 2026-11-17: sha256 matches HF LFS OID, fp16 243 MB).
This base owns loading + a feature-extraction helper; the method engines
turn those features into verdicts.

Lazy imports: load() / helpers run inside the engine lifecycle, never at
module import time, so the package stays torch-free for tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import EngineBase, WeightsMissing
from .registry import EngineSpec

VERSION_ROOT = Path(__file__).resolve().parent.parent.parent


class Gpt2BaseEngine(EngineBase):
    """GPT-2 base residency + next-token feature extraction."""

    WEIGHTS_DIRNAME = "gpt2"  # shared base dir, NOT models/<engine_id>
    MAX_TOKENS = 1024

    def __init__(self, spec: EngineSpec) -> None:
        self.engine_id = spec.engine_id
        self.lang = spec.lang
        self.kind = spec.kind
        self.est_vram_mb = spec.est_vram_mb
        self.exclusive = spec.exclusive
        self._weights = VERSION_ROOT / "models" / self.WEIGHTS_DIRNAME
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self._weights / "config.json").is_file():
            raise WeightsMissing(f"{self.engine_id}: no config.json in {self._weights.name}/")
        if self.load_variant == "fp16":
            if not (self._weights / "model.fp16.safetensors").is_file():
                raise WeightsMissing(f"{self.engine_id}: model.fp16.safetensors not cast")
        elif not (self._weights / "model.safetensors").is_file():
            raise WeightsMissing(f"{self.engine_id}: model.safetensors not downloaded")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.load_dtype == "float16":
            dtype = torch.float16
        elif self.load_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._weights)
            model = AutoModelForCausalLM.from_pretrained(
                self._weights, dtype=dtype, variant=self.load_variant
            ).to(device)
        except Exception as e:
            self._model = self._tokenizer = None
            raise WeightsMissing(
                f"{self.engine_id}: weight load failed ({type(e).__name__}: {e})"
            ) from e
        model.eval()
        self._model = model
        self._device = device

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
