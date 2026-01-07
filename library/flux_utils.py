import torch
import os
import logging
from safetensors.torch import load_file as load_safetensors
from . import flux_models
from transformers import CLIPTextModel, CLIPConfig, T5EncoderModel, T5Config, AutoencoderKL
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME_DEV = "black-forest-labs/FLUX.1-dev"
MODEL_NAME_SCHNELL = "black-forest-labs/FLUX.1-schnell"

# VULCAN FIX: Placeholder for broken import
def convert_diffusers_sd_to_bfl(sd, num_double_blocks, num_single_blocks):
    return sd

def analyze_checkpoint_state(ckpt_path: str):
    # Auto-download logic
    if not os.path.exists(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "transformer")):
        logger.info(f"'{ckpt_path}' not found locally. Attempting HF download...")
        try:
            if "FLUX.1" in ckpt_path:
                target_file = "flux1-dev.safetensors" if "dev" in ckpt_path else "flux1-schnell.safetensors"
                ckpt_path = hf_hub_download(repo_id=ckpt_path, filename=target_file)
                logger.info(f"Downloaded to: {ckpt_path}")
        except Exception as e:
            logger.warning(f"Download attempt failed: {e}")

    ckpt_path = ckpt_path.strip('"').strip("'")
    if os.path.isdir(ckpt_path):
        return True, False, (19, 38), []
    else:
        is_schnell = "schnell" in os.path.basename(ckpt_path)
        return False, is_schnell, (19, 38), [ckpt_path]

def load_flow_model(ckpt_path: str, dtype, device, disable_mmap=False, model_type="flux"):
    ckpt_path = ckpt_path.strip('"').strip("'")
    
    if model_type == "flux":
        is_diffusers, is_schnell, (num_double, num_single), paths = analyze_checkpoint_state(ckpt_path)
        name = "dev" if not is_schnell else "schnell"
        
        logger.info(f"Building Flux model '{name}'...")
        with torch.device("meta"):
            params = flux_models.configs[name]
            model = flux_models.Flux(params)
            if dtype is not None:
                model = model.to(dtype)

        logger.info(f"Loading state dict from {ckpt_path}")
        sd = {}
        for p in paths:
            # VULCAN FIX: Removed 'disable_mmap' argument which causes crashes on Kaggle
            sd.update(load_safetensors(p, device=device))

        keys_to_rename = [k for k in sd.keys() if k.startswith("model.diffusion_model.")]
        for key in keys_to_rename:
            new_key = key.replace("model.diffusion_model.", "")
            sd[new_key] = sd.pop(key)
        
        info = model.load_state_dict(sd, strict=False, assign=True)
        logger.info(f"Loaded Flux: {info}")
        return is_schnell, model

    raise NotImplementedError(f"Model type {model_type} not implemented")

# --- VULCAN FIX: Added Missing AutoEncoder Loader ---
def load_ae(name, dtype, device, disable_mmap=False):
    logger.info(f"Loading AE from {name}...")
    try:
        # Try loading as a subfolder (standard Diffusers repo)
        ae = AutoencoderKL.from_pretrained(name, subfolder="ae", torch_dtype=dtype).to(device)
    except:
        # Fallback (e.g. if name is just a path or different structure)
        ae = AutoencoderKL.from_pretrained(name, torch_dtype=dtype).to(device)
    return ae

def load_clip_l(path, dtype, device, disable_mmap=False):
    if path is None:
        path = "openai/clip-vit-large-patch14"
    logger.info(f"Loading CLIP from {path}")
    try:
        return CLIPTextModel.from_pretrained(path, torch_dtype=dtype).to(device)
    except:
        # VULCAN FIX: Added max_position_embeddings=77
        config = CLIPConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu")
        return CLIPTextModel(config).to(device, dtype=dtype)

def load_t5xxl(path, dtype, device, disable_mmap=False):
    if path is None:
        path = "google/t5-v1_1-xxl"
    logger.info(f"Loading T5 from {path}")
    try:
        return T5EncoderModel.from_pretrained(path, torch_dtype=dtype).to(device)
    except:
        config = T5Config(vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240, num_layers=24, num_heads=64, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-6, initializer_factor=1.0, feed_forward_proj="gated-gelu", is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1)
        return T5EncoderModel(config).to(device, dtype=dtype)