import os
import torch
from typing import Any, List, Optional
from transformers import CLIPTokenizer, T5Tokenizer
from . import strategy_base, flux_utils

class FluxTokenizeStrategy(strategy_base.TokenizeStrategy):
    def __init__(self, t5_xxl_max_token_length: int = 512, tokenizer_cache_dir: Optional[str] = None):
        self.t5_xxl_max_token_length = t5_xxl_max_token_length
        # Preserved Fix: Attributes match train_network.py
        self.clip_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length=77)
        self.t5xxl = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=t5_xxl_max_token_length)

    def tokenize(self, text: str | List[str]) -> List[Any]:
        if isinstance(text, str):
            text = [text]
        clip_tokens = self.clip_l(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, padding="max_length", max_length=self.t5_xxl_max_token_length, truncation=True, return_tensors="pt")
        return [clip_tokens, t5_tokens]

class FluxTextEncodingStrategy(strategy_base.TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: bool = False):
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def encode_tokens(self, tokenize_strategy: strategy_base.TokenizeStrategy, models: List[Any], tokens: List[Any]) -> List[Any]:
        clip_tokens, t5_tokens = tokens
        clip_l, t5_xxl = models
        
        if hasattr(clip_l, "transformer"):
            clip_enc = clip_l(clip_tokens["input_ids"].to(clip_l.device), output_hidden_states=False)
            vec = clip_enc.pooler_output
        else:
            vec = clip_l(clip_tokens["input_ids"].to(clip_l.device))[1]

        t5_enc = t5_xxl(t5_tokens["input_ids"].to(t5_xxl.device), attention_mask=t5_tokens["attention_mask"].to(t5_xxl.device) if self.apply_t5_attn_mask else None)
        txt = t5_enc.last_hidden_state
        return [vec, txt]

class FluxLatentsCachingStrategy(strategy_base.LatentsCachingStrategy):
    def __init__(self, cache_to_disk: bool = False, batch_size: int = 1, num_workers: int = 1):
        super().__init__(cache_to_disk, batch_size, num_workers)

    # --- VULCAN FIX: The Missing Property ---
    @property
    def cache_suffix(self) -> str:
        return ".npz"

    def get_latents_npz_path(self, absolute_path: str, image_size: Optional[Any]) -> str:
        return os.path.splitext(absolute_path)[0] + ".npz"

    def cache_batch_latents(self, model, batch: List[Any], accelerator=None):
        # We can pass here because the main training loop handles generation if cache is missing
        pass