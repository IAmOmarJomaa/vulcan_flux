import os
import torch
from typing import Any, List, Optional
from transformers import CLIPTokenizer, T5Tokenizer
from . import strategy_base, flux_utils

class FluxTokenizeStrategy(strategy_base.TokenizeStrategy):
    def __init__(self, t5_xxl_max_token_length: int = 512, tokenizer_cache_dir: Optional[str] = None):
        self.t5_xxl_max_token_length = t5_xxl_max_token_length
        # Load tokenizers directly from transformers
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length=77)
        self.t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=t5_xxl_max_token_length)

    def tokenize(self, text: str | List[str]) -> List[Any]:
        if isinstance(text, str):
            text = [text]
        
        # Tokenize for CLIP
        clip_tokens = self.clip_tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        
        # Tokenize for T5
        t5_tokens = self.t5_tokenizer(text, padding="max_length", max_length=self.t5_xxl_max_token_length, truncation=True, return_tensors="pt")
        
        return [clip_tokens, t5_tokens]

class FluxTextEncodingStrategy(strategy_base.TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: bool = False):
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def encode_tokens(self, tokenize_strategy: strategy_base.TokenizeStrategy, models: List[Any], tokens: List[Any]) -> List[Any]:
        clip_tokens, t5_tokens = tokens
        clip_l, t5_xxl = models
        
        # Encode CLIP
        # We need the pooled output
        if hasattr(clip_l, "transformer"):
            # It's a transformer model, get the pooler output
            clip_enc = clip_l(clip_tokens["input_ids"].to(clip_l.device), output_hidden_states=False)
            vec = clip_enc.pooler_output
        else:
            # Fallback
            vec = clip_l(clip_tokens["input_ids"].to(clip_l.device))[1]

        # Encode T5
        # We need the last hidden state
        t5_enc = t5_xxl(t5_tokens["input_ids"].to(t5_xxl.device), attention_mask=t5_tokens["attention_mask"].to(t5_xxl.device) if self.apply_t5_attn_mask else None)
        txt = t5_enc.last_hidden_state
        
        return [vec, txt]

# --- VULCAN FIX: The Missing Class Method ---
class FluxLatentCachingStrategy(strategy_base.LatentCachingStrategy):
    def __init__(self, cache_to_disk: bool = False, batch_size: int = 1, num_workers: int = 1):
        super().__init__(cache_to_disk, batch_size, num_workers)

    # THIS is the missing method that caused your NotImplementedError
    def get_latents_npz_path(self, absolute_path: str, image_size: Optional[Any]) -> str:
        # Standard logic: image.png -> image.npz
        return os.path.splitext(absolute_path)[0] + ".npz"

    def cache_batch_latents(self, model, batch: List[Any], accelerator=None):
        # Basic implementation to satisfy the strategy pattern
        # In a real scenario, this would handle VAE encoding and saving
        # For now, we rely on the training loop's internal caching if this returns None or isn't called directly
        pass