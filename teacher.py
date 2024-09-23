"""
Added by FS.

Script for loading the loss values of a pre-trained teacher model.
"""

import torch
import os

from timm.loss import LabelSmoothingCrossEntropy
# from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_teacher_output(model_name: str,
                       loss_fn: torch.nn.Module,
                       device: str="cpu",
                       seed: int=42
    ):
    
    save_dir = f"output/teacher/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        loss_name = "CE"
    elif isinstance(loss_fn, LabelSmoothingCrossEntropy):
        loss_name =f"SmoothingCE_{loss_fn.smoothing}"
    else:
        raise KeyError("Unknown loss.")

    fname = os.path.join(save_dir, f"losses_{model_name}_{loss_name}_seed_{seed}.pkl")

    # Check if outputs have been precomputed
    if os.path.isfile(fname):
        print(f"Teacher losses have been pre-computed, loading from {fname}")
        losses = torch.load(f=fname).to(device)

        assert len(losses) == 1_281_167
        print("Zero entries in teacher losses:", len(losses[losses==0.0]))
        print(f"Teacher loss {losses.min().item()} (min), {losses.max().item()} (max), {losses.mean().item()} (mean).")
    
    else:
        raise KeyError(f"Teacher losses for model {model_name} are unknown, first run validate.py")
        
    return losses







