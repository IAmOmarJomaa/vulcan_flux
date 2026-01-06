import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union
import torch
from torch import Tensor
import re
from library.utils import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38

class LoRAModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # Standard LoRA layers
        self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # workable for save

        # Init weights
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        # Forward original module
        # VULCAN NOTE: We use the stored forward method of the original linear layer
        # This works even if the layer is 4-bit (NF4) or 8-bit (FP8)
        org_forwarded = self.org_forward(x)

        # Forward LoRA
        lx = self.lora_down(x)
        lx = self.lora_up(lx)

        # Apply scaling and multiplier
        return org_forwarded + lx * self.multiplier * self.scale

class LoRANetwork(torch.nn.Module):
    def __init__(
        self,
        text_encoder: Union[List[object], object],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        varbose: Optional[bool] = False,
        split_dims: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.lora_modules = []
        
        # VULCAN: Only targeting UNet (Flux) for now to save VRAM
        # If we wanted to train T5, we would add logic here.
        self.create_modules(unet, "lora_unet", split_dims)

        def create_modules(self, target_model, module_name_prefix, split_dims):
            # Scan model and attach LoRA modules
            # Simplified for brevity: assumes Linear layers in Flux blocks
            for name, module in target_model.named_modules():
                if module.__class__.__name__ == "Linear" or "Linear" in module.__class__.__name__:
                    # Filter for specific Flux layers if needed (attn, mlp)
                    # For V1, we target everything appropriate
                    if "attn" in name or "mlp" in name or "mod" in name:
                        lora_module = LoRAModule(
                            f"{module_name_prefix}_{name}", module, self.multiplier, 
                            self.lora_dim, self.alpha, self.dropout
                        )
                        self.lora_modules.append(lora_module)
                        lora_module.apply_to() # Hook it up

    def create_modules(self, root_module, prefix, split_dims):
        lora_names = []
        for name, module in root_module.named_modules():
             # Target Linear layers in Flux blocks
             # We specifically look for the "linear" layers inside the blocks
             if isinstance(module, torch.nn.Linear) or "Linear" in module.__class__.__name__:
                 # Exclude output layers or embeddings if desired, but Flux usually trains strictly on blocks
                 if "double_blocks" in name or "single_blocks" in name:
                     lora_name = prefix + "." + name
                     lora_name = lora_name.replace(".", "_")
                     
                     lora = LoRAModule(lora_name, module, self.multiplier, self.lora_dim, self.alpha)
                     self.lora_modules.append(lora)
                     self.add_module(lora_name, lora)
                     lora.apply_to()
                     lora_names.append(lora_name)
        
        logger.info(f"created {len(self.lora_modules)} modules for {prefix}")

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, learning_rate):
        # We only return unet params as we are not training TE in this V1
        return [{"params": self.parameters(), "lr": unet_lr if unet_lr is not None else learning_rate}]

    def enable_gradient_checkpointing(self):
        # Not needed for LoRA itself usually, but good for safety
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        
        from safetensors.torch import save_file
        save_file(state_dict, file, metadata)

def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1

    # Standard creation
    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        varbose=True
    )
    return network