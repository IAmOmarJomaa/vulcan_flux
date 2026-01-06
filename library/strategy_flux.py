import torch
import transformers
from typing import Optional, List, Any
from . import strategy_base
import logging

logger = logging.getLogger(__name__)

class FluxTokenizeStrategy(strategy_base.TokenizeStrategy):
    def __init__(self, t5xxl_max_token_length: int = 512, tokenizer_cache_dir: Optional[str] = None):
        self.t5xxl_max_token_length = t5xxl_max_token_length
        self.clip_l = self._load_tokenizer(transformers.CLIPTokenizer, "openai/clip-vit-large-patch14", tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(transformers.T5Tokenizer, "google/t5-v1_1-xxl", tokenizer_cache_dir)

    def _load_tokenizer(self, tokenizer_class, model_id, cache_dir):
        return tokenizer_class.from_pretrained(model_id, cache_dir=cache_dir)

    def tokenize(self, text: str):
        clip_tokens = self.clip_l(
            text, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        t5_tokens = self.t5xxl(
            text, max_length=self.t5xxl_max_token_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return [clip_tokens, t5_tokens]

class FluxTextEncodingStrategy(strategy_base.TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: bool = False):
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def encode_tokens(self, tokenize_strategy, models, tokens, apply_t5_attn_mask=False):
        clip_l, t5xxl = models
        clip_tokens, t5_tokens = tokens

        with torch.no_grad():
            l_pooled = clip_l(clip_tokens.to(clip_l.device))["pooler_output"]

        with torch.no_grad():
            t5_out = t5xxl(t5_tokens.to(t5xxl.device))["last_hidden_state"]

        return [l_pooled, t5_out, t5_tokens, None]

class FluxLatentsCachingStrategy(strategy_base.LatentsCachingStrategy):
    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_check: bool):
        super().__init__(cache_to_disk, batch_size, skip_disk_check)

class FluxTextEncoderOutputsCachingStrategy(strategy_base.TextEncoderOutputsCachingStrategy):
    def __init__(self, cache_to_disk, batch_size, skip_disk_check, is_partial, apply_t5_attn_mask):
        super().__init__(cache_to_disk, batch_size, skip_disk_check, is_partial)
        self.apply_t5_attn_mask = apply_t5_attn_mask

    @property
    def cache_suffix(self):
        return "_te_outputs.npz"