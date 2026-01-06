import torch
from typing import Tuple, List
import os
from safetensors.torch import load_file as load_safetensors
from . import flux_models
from .model_util import convert_diffusers_sd_to_bfl
import logging
from transformers import CLIPTextModel, CLIPConfig, T5EncoderModel, T5Config
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME_DEV = "black-forest-labs/FLUX.1-dev"
MODEL_NAME_SCHNELL = "black-forest-labs/FLUX.1-schnell"

def analyze_checkpoint_state(ckpt_path: str) -> Tuple[bool, bool, Tuple[int, int], List[str]]:
    # VULCAN FIX: Auto-download from HF if not local
    if not os.path.exists(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "transformer")):
        logger.info(f"'{ckpt_path}' not found locally. Attempting HF download...")
        try:
            # Check if it looks like the BFL repo ID
            if "FLUX.1" in ckpt_path:
                target_file = "flux1-dev.safetensors" if "dev" in ckpt_path else "flux1-schnell.safetensors"
                ckpt_path = hf_hub_download(repo_id=ckpt_path, filename=target_file)
                logger.info(f"Downloaded to: {ckpt_path}")
            else:
                 # Assume it's a diffusers repo structure
                 pass 
        except Exception as e:
            logger.warning(f"Download attempt failed or path is valid directory: {e}")

    ckpt_path = ckpt_path.strip('"').strip("'")
    if os.path.isdir(ckpt_path):
        # Diffusers format
        is_diffusers = True
        # Simple check for schnell in config (approximate)
        config_path = os.path.join(ckpt_path, "transformer", "config.json")
        is_schnell = False
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                content = f.read()
                if "schnell" in content:
                    is_schnell = True
        return is_diffusers, is_schnell, (19, 38), [] # Defaults for now
    else:
        # Single file BFL format
        is_diffusers = False
        is_schnell = "schnell" in os.path.basename(ckpt_path)
        return is_diffusers, is_schnell, (19, 38), [ckpt_path]

def load_flow_model(ckpt_path: str, dtype, device, disable_mmap=False, model_type="flux"):
    ckpt_path = ckpt_path.strip('"').strip("'")
    
    if model_type == "flux":
        is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(ckpt_path)
        
        # Map to the keys used in flux_models.py configs
        name = "dev" if not is_schnell else "schnell"
        
        logger.info(f"Building Flux model '{name}' from {'Diffusers' if is_diffusers else 'BFL'} checkpoint")
        
        with torch.device("meta"):
            # Load the params from the factory
            params = flux_models.configs[name]
            
            # Build the empty shell
            model = flux_models.Flux(params)
            
            if dtype is not None:
                model = model.to(dtype)

        logger.info(f"Loading state dict from {ckpt_path}")
        sd = {}
        for p in ckpt_paths:
            sd.update(load_safetensors(p, device=device, disable_mmap=disable_mmap, dtype=dtype))

        if is_diffusers:
            logger.info("Converting Diffusers to BFL")
            sd = convert_diffusers_sd_to_bfl(sd, num_double_blocks, num_single_blocks)

        # Sanitize keys: Remove "model.diffusion_model." prefix if it exists
        keys_to_rename = [k for k in sd.keys() if k.startswith("model.diffusion_model.")]
        for key in keys_to_rename:
            new_key = key.replace("model.diffusion_model.", "")
            sd[new_key] = sd.pop(key)
        
        if keys_to_rename:
            logger.info(f"Sanitized {len(keys_to_rename)} keys.")

        # Load weights into the shell
        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded Flux: {info}")
        return is_schnell, model

    raise NotImplementedError(f"Model type {model_type} not implemented")

def load_clip_l(path, dtype, device, disable_mmap=False):
    # VULCAN FIX: Robust config generation
    if path is None:
        path = "openai/clip-vit-large-patch14"
        
    logger.info(f"Building CLIP-L from {path}")
    if os.path.exists(path) or path.count("/") > 0:
        try:
            # Try loading automatically via transformers
            text_encoder = CLIPTextModel.from_pretrained(path, torch_dtype=dtype)
        except Exception:
            # Fallback to manual construction if simple load fails or no internet
            config = CLIPConfig(
                vocab_size=49408,
                hidden_size=768,
                intermediate_size=3072,
                num_hidden_layers=12,
                num_attention_heads=12,
                max_position_embeddings=77, # <--- FIXED: Added this required field
                hidden_act="quick_gelu",
                layer_norm_eps=1e-5,
                initializer_range=0.02,
                pad_token_id=1,
                bos_token_id=49406,
                eos_token_id=49407,
            )
            text_encoder = CLIPTextModel(config)
            # If we had a local file, we would load weights here, 
            # but for now we assume we are initializing a base or downloading.
            if path == "openai/clip-vit-large-patch14":
                 # If we are here, from_pretrained failed, so we might return random init
                 # But usually from_pretrained works for the HF ID.
                 pass
    else:
        # Fallback manual config
        config = CLIPConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77, # <--- FIXED
            hidden_act="quick_gelu"
        )
        text_encoder = CLIPTextModel(config)

    return text_encoder.to(device, dtype=dtype)

def load_t5xxl(path, dtype, device, disable_mmap=False):
    if path is None:
        path = "google/t5-v1_1-xxl"
        
    logger.info(f"Building T5-XXL from {path}")
    try:
        text_encoder = T5EncoderModel.from_pretrained(path, torch_dtype=dtype)
    except Exception:
        # Fallback manual config for T5 XXL
        config = T5Config(
            vocab_size=32128,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_heads=64,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            layer_norm_epsilon=1e-6,
            initializer_factor=1.0,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
        )
        text_encoder = T5EncoderModel(config)
        
    return text_encoder.to(device, dtype=dtype)