import torch
import torch.nn as nn


@torch.no_grad()
def get_ffi_structure(mod: nn.Linear, sparsity: float) -> nn.Linear:
    n_zeros = int(mod.weight.numel() * (sparsity))
    n_zeros_per_neuron = n_zeros // mod.weight.shape[0]
    for idx, neuron in enumerate(mod.weight):
        rand_idx = torch.randint(low=0, high=len(neuron))
        mod.weight[idx, rand_idx[:n_zeros_per_neuron]] = 0
    return mod
