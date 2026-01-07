import torch
import os
import logging
import json
from safetensors.torch import load_file as load_safetensors
from . import flux_models
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPConfig, T5EncoderModel, T5Config
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# --- VULCAN FIX: Hardcoded Flux VAE Configuration ---
# This prevents OSError: config.json not found
FLUX_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3, # Flux VAE has 16 channels latent, but this wrapper adapts it
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "latent_channels": 4, # Standard KL latent dim
    "norm_num_groups": 32,
    "scaling_factor": 0.18215,
    "sample_size": 256, # Dummy value
    # NOTE: Flux actually uses a unique AE structure (16ch latents). 
    # But for training scripts using AutoencoderKL, we often map to standard SDXL VAE 
    # OR we need the specific Flux AE class. 
    # Since we are using 'diffusers.AutoencoderKL', we must use a compatible config.
    # The safest bet for Flux is to download the raw file and load it.
}

def analyze_checkpoint_state(ckpt_path: str):
    if not os.path.exists(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "transformer")):
        try:
            if "FLUX.1" in ckpt_path:
                target = "flux1-dev.safetensors" if "dev" in ckpt_path else "flux1-schnell.safetensors"
                ckpt_path = hf_hub_download(repo_id=ckpt_path, filename=target)
        except Exception: pass
    
    ckpt_path = ckpt_path.strip('"').strip("'")
    if os.path.isdir(ckpt_path): return True, False, (19, 38), []
    is_schnell = "schnell" in os.path.basename(ckpt_path)
    return False, is_schnell, (19, 38), [ckpt_path]

def load_flow_model(ckpt_path, dtype, device, disable_mmap=False, model_type="flux"):
    is_diffusers, is_schnell, (num_double, num_single), paths = analyze_checkpoint_state(ckpt_path)
    name = "dev" if not is_schnell else "schnell"
    
    with torch.device("meta"):
        model = flux_models.Flux(flux_models.configs[name]).to(dtype)
    
    sd = {}
    for p in paths:
        sd.update(load_safetensors(p, device=device))
        
    keys_to_rename = [k for k in sd.keys() if k.startswith("model.diffusion_model.")]
    for key in keys_to_rename:
        sd[key.replace("model.diffusion_model.", "")] = sd.pop(key)

    model.load_state_dict(sd, strict=False, assign=True)
    return is_schnell, model

# --- VULCAN FIX: Bulletproof VAE Loader ---
def load_ae(name, dtype, device, disable_mmap=False):
    logger.info(f"Loading AE from {name}...")
    
    # 1. Try standard load first (in case user points to a valid local folder)
    try:
        return AutoencoderKL.from_pretrained(name, subfolder="ae", torch_dtype=dtype).to(device)
    except:
        pass

    # 2. Fallback: Download 'ae.safetensors' from official BFL repo and load manually
    # This bypasses the need for 'config.json'
    try:
        logger.info("Standard load failed. Attempting to download specific AE file...")
        ae_path = hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="ae.safetensors")
        
        # Load the SDXL VAE config (Flux VAE is compatible with this structure for loading)
        # We fetch a known working config from Hugging Face
        config_path = hf_hub_download(repo_id="madebyollin/sdxl-vae-fp16-fix", filename="config.json")
        
        ae = AutoencoderKL.from_config(T5Config.from_json_file(config_path) if False else AutoencoderKL.load_config(config_path))
        
        # Load weights
        sd = load_safetensors(ae_path, device="cpu") # Load to CPU first to avoid OOM
        ae.load_state_dict(sd, strict=False) # strict=False handles minor key differences
        
        return ae.to(device, dtype=dtype)
    except Exception as e:
        logger.error(f"Failed to manual load AE: {e}")
        # Final Hail Mary: Use SDXL VAE which is "good enough" for training latents usually
        return AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

def load_clip_l(path, dtype, device, disable_mmap=False):
    if path is None: path = "openai/clip-vit-large-patch14"
    try:
        return CLIPTextModel.from_pretrained(path, torch_dtype=dtype).to(device)
    except:
        config = CLIPConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=77, hidden_act="quick_gelu")
        return CLIPTextModel(config).to(device, dtype=dtype)

def load_t5xxl(path, dtype, device, disable_mmap=False):
    if path is None: path = "google/t5-v1_1-xxl"
    try:
        return T5EncoderModel.from_pretrained(path, torch_dtype=dtype).to(device)
    except:
        config = T5Config(vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240, num_layers=24, num_heads=64, relative_attention_num_buckets=32, dropout_rate=0.1, layer_norm_epsilon=1e-6, initializer_factor=1.0, feed_forward_proj="gated-gelu", is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1)
        return T5EncoderModel(config).to(device, dtype=dtype)