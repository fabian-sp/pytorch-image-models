"""
Added by FS.

Script for computing the loss values of a pre-trained teacher model.
"""

import torch
import os
import tqdm

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader

# Test 1: If shuffle=False, the order of the train set is not altered

# dl = torch.utils.data.DataLoader(torch.arange(100), batch_size=5, shuffle=False)
# for d in dl:
#     print(d)


# Test 2: we need to wrap the DataLoader such that it returns the index of the inputs and labels.

# ds = torch.utils.data.TensorDataset(torch.arange(100)*0.01, torch.arange(100))
# dl = torch.utils.data.DataLoader(DataClass(ds, split='train'), batch_size=5, shuffle=True)
# for d in dl:
#     print(d)

def compute_teacher_output(model_name: str,
                           loader: DataLoader,
                           device: str="cpu",
                           train_aug_seed: int=0,
                           train_set_size: int=1281167
    ):
    
    save_dir = f"output/teacher/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = os.path.join(save_dir, f"losses_{model_name}_seed_{train_aug_seed}.pkl")

    # Check if outputs have been precomputed
    if os.path.isfile(fname):
        print(f"Teacher losses have been pre-computed, loading from {fname}")

        losses = torch.load(f=fname)

    else:
        print(f"Compute teacher losses for model {model_name}")
        
        # Prepare
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        losses = torch.zeros(train_set_size, requires_grad=False)

        # loader = torch.utils.data.DataLoader(ds, 
        #                                      batch_size=batch_size, 
        #                                      shuffle=False                          # IMPORTANT!
        # )

        # Get model
        if model_name == "vit_b_16":
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            raise KeyError("Unknown teacher model name.")
        
        model.to(device)
        model.eval()
        
        # Compute losses
        with torch.no_grad():
            for batch in tqdm.tqdm(loader):
                data, targets, ind = batch['data'].to(device), batch['targets'].to(device), batch['ind'].to(device)
                out = model(data)

                loss = loss_fn(out, targets)
                losses[ind] = loss

        # Store tensor
        torch.save(losses, f=fname)

    return losses







