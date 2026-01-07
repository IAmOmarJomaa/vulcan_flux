import os
import torch
import numpy as np
from typing import Any, List, Optional, Tuple
from transformers import CLIPTokenizer, T5Tokenizer
from PIL import Image # VULCAN FIX: Switched to PIL for safety
from . import strategy_base, flux_utils

class FluxTokenizeStrategy(strategy_base.TokenizeStrategy):
    def __init__(self, t5_xxl_max_token_length: int = 512, tokenizer_cache_dir: Optional[str] = None):
        self.t5_xxl_max_token_length = t5_xxl_max_token_length
        self.clip_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length=77)
        self.t5xxl = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=t5_xxl_max_token_length)

    def tokenize(self, text: str | List[str]) -> List[Any]:
        if isinstance(text, str): text = [text]
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
            vec = clip_l(clip_tokens["input_ids"].to(clip_l.device), output_hidden_states=False).pooler_output
        else:
            vec = clip_l(clip_tokens["input_ids"].to(clip_l.device))[1]
        
        t5_enc = t5_xxl(t5_tokens["input_ids"].to(t5_xxl.device), attention_mask=t5_tokens["attention_mask"].to(t5_xxl.device) if self.apply_t5_attn_mask else None)
        return [vec, t5_enc.last_hidden_state]

class FluxLatentsCachingStrategy(strategy_base.LatentsCachingStrategy):
    def __init__(self, cache_to_disk: bool = False, batch_size: int = 1, num_workers: int = 1):
        super().__init__(cache_to_disk, batch_size, num_workers)

    @property
    def cache_suffix(self) -> str:
        return ".npz"

    def get_latents_npz_path(self, absolute_path: str, image_size: Optional[Any]) -> str:
        if not isinstance(absolute_path, str): return ""
        return os.path.splitext(absolute_path)[0] + ".npz"

    def is_disk_cached_latents_expected(self, *args, **kwargs) -> bool:
        absolute_path = None
        for arg in args:
            if isinstance(arg, str) and (os.path.exists(arg) or os.path.isabs(arg)):
                absolute_path = arg
                break
        if absolute_path is None: return False
        return os.path.exists(self.get_latents_npz_path(absolute_path, None))

    def load_latents_from_disk(self, absolute_path: str, image_size: Tuple[int, int]) -> Any:
        npz_path = self.get_latents_npz_path(absolute_path, image_size)
        data = np.load(npz_path)
        return torch.from_numpy(data['latents'])

    # --- VULCAN FIX: Signature updated to accept *args (flip, alpha, etc) ---
    def cache_batch_latents(self, model, batch: List[Any], *args, **kwargs):
        if model is None or not batch: return

        device = model.device
        dtype = model.dtype

        for info in batch:
            # Handle possible object types
            image_path = getattr(info, 'absolute_path', None)
            if image_path is None and isinstance(info, str): image_path = info
            
            if not image_path or not os.path.exists(image_path): continue

            # Load and Preprocess
            try:
                img = Image.open(image_path).convert("RGB")
                # Basic resize to target bucket size if available in info
                target_size = getattr(info, 'image_size', None)
                if target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                
                # To Tensor [-1, 1]
                img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device, dtype)
                
                # Encode
                with torch.no_grad():
                    latents = model.encode(img_tensor).latent_dist.sample()
                
                # Save
                npz_path = self.get_latents_npz_path(image_path, None)
                np.savez(npz_path, latents=latents.float().cpu().numpy())
                # print(f"Cached: {npz_path}") # Uncomment for debug
            except Exception as e:
                print(f"Failed to cache {image_path}: {e}")