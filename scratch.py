# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F

__DEVICE = torch.device("cuda")
# __DEVICE = torch.device("cpu")
__INPUT_DIM = 762
__OUTPUT_DIM = 4096
__BATCH_SIZE = 128
__SPARSITY = 0.95


# +


@torch.no_grad()
def get_ffi_structure(mod: nn.Linear, sparsity: float) -> nn.Linear:
    n_zeros = int(mod.weight.numel() * (sparsity))
    n_zeros_per_neuron = n_zeros // mod.weight.shape[0]
    for idx, neuron in enumerate(mod.weight):
        rand_idx = torch.randperm(n=len(neuron))
        mod.weight[idx, rand_idx[: n_zeros_per_neuron - 1]] = 0
    assert_ffi(mod)
    print_sparsity(mod)
    return mod


def assert_ffi(mod: nn.Linear):
    ffi = (mod.weight[0] != 0).sum()
    for n in mod.weight:
        assert (n != 0).sum() == ffi


def print_sparsity(mod: nn.Linear):
    print(
        f"Mod sparsity: {1-((mod.weight!=0).sum()/mod.weight.numel()).item():.4f}"
    )


# -

linear = nn.Linear(
    in_features=__INPUT_DIM, out_features=__OUTPUT_DIM, device=__DEVICE
)
sparse_linear = get_ffi_structure(linear, __SPARSITY)

x = torch.rand(size=(__BATCH_SIZE, __INPUT_DIM), device=__DEVICE)
x.shape


class FFILinearNaive(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        dtype: torch.typename = torch.float32,
        transpose: bool = True,
        vectorize: bool = False,
        index_dtype: torch.typename = torch.int32,
    ):
        super().__init__()
        if dtype is None:
            dtype = module.weight.dtype

        self.transpose = transpose
        with torch.no_grad():
            fine_grained_idx = (module.weight != 0).to(torch.bool)
            _, self.input_mask = fine_grained_idx.nonzero(as_tuple=True)
            self.input_mask = self.input_mask.reshape(
                shape=(module.weight.shape[0], -1)
            ).to(index_dtype)
            weight = module.weight.detach().type(dtype)
            weight = torch.clone(
                weight[fine_grained_idx]
                .reshape(shape=(weight.shape[0], -1))
                .detach()
                .type(dtype)
            )
            # padding to multiple of 4
            if vectorize:
                pad = (
                    self.input_mask.shape[1] + 3
                ) // 4 * 4 - self.input_mask.shape[1]
                self.input_mask = F.pad(self.input_mask, [0, pad])
                weight = F.pad(weight, [0, pad])

            self.condensed_weight = nn.Parameter(
                weight,
                requires_grad=False,
            )

            if hasattr(module, "bias"):
                self.bias = nn.Parameter(
                    torch.clone(module.bias.detach().type(dtype)),
                    requires_grad=False,
                )
            else:
                self.register_parameter("bias", None)
            self.to(module.weight.device)
            self.input_mask.to(module.weight.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            size=(input.shape[0], self.condensed_weight.shape[0]),
            device=x.device,
        )
        output_size, nnz_el_per_neuron = self.input_mask.shape
        for batch in range(input.shape[0]):
            for out in range(output_size):
                output[batch, out] = self.bias[out]
                for index in range(nnz_el_per_neuron):
                    output[batch, out] += (
                        input[batch, self.input_mask[out, index]]
                        * self.condensed_weight[out, index]
                    )
        return output


ffi_naive = FFILinearNaive(sparse_linear)
# sparse_linear(x).allclose(ffi_naive(x), atol=1e-07)


# +
class FFILinearVmap(FFILinearNaive):
    def __init__(
        self,
        module: nn.Module,
        dtype: torch.typename = torch.float32,
        transpose: bool = True,
        vectorize: bool = False,
        index_dtype: torch.typename = torch.int32,
    ):
        super().__init__(module, dtype, transpose, vectorize, index_dtype)

    def batch_kernel(self, input, input_masks, weights, biases):
        return torch.vmap(self.output_kernel, in_dims=(None, 0, 0, 0))(
            input, input_masks, weights, biases
        )

    def output_kernel(self, input, input_mask, weight, bias):
        return bias + torch.sum(input[input_mask] * weight)

    # @override
    def forward(self, input: torch.Tensor):
        return torch.vmap(
            self.batch_kernel, in_dims=(0, None, None, None), out_dims=0
        )(x, self.input_mask, self.condensed_weight, self.bias)


ffi_vmap = FFILinearVmap(sparse_linear, vectorize=True)
ffi_vmap(x).allclose(sparse_linear(x), atol=1e-06)
# -

# ffi_compiled = torch.compile(ffi_vmap, mode="reduce-overhead")
ffi_compiled = torch.compile(ffi_vmap, mode="max-autotune")
for _ in range(10):
    _ = ffi_compiled(x)

# +
N_ITERS = 10


def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(10):
        _ = fn()
    start.record()
    for _ in range(N_ITERS):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


# -

with torch.no_grad():
    print("Dense:", timed(lambda: linear(x))[1])
    # print("FFI_Naive:", timed(lambda: ffi_naive(x))[1])
    print("FFI_VMAP:", timed(lambda: ffi_vmap(x))[1])
    print("FFI_Compiled:", timed(lambda: ffi_compiled(x))[1])

ffi_vmap.condensed_weight.shape

ffi_vmap.input_mask[0]

# +
## Triton
import triton
import triton.language as tl
import os
import pdb

os.environ["TRITON_INTERPRET"] = "0"
torch.backends.cuda.matmul.allow_tf32 = False


@triton.jit
def ffi_triton(
    # Output / Input pointers
    output_p,
    input_p,
    # Pointers to weights, bias, and mask
    weight_p,
    bias_p,
    mask_p,
    # number of outputs channels, input channels, and batch_size
    n_out: tl.constexpr,
    n_in: tl.constexpr,
    n_batch: tl.constexpr,
    n_weights: tl.constexpr,
    # Number of nnz_el per neuron and grid block size
    FFI_PER_NEURON: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    block_idx_x, block_idx_y = tl.program_id(axis=0), tl.program_id(axis=1)
    batch_idx = block_idx_x * BLOCK_SIZE_X
    unit_idx = block_idx_y * BLOCK_SIZE_Y

    batch_offsets = batch_idx + tl.arange(0, BLOCK_SIZE_X)

    unit_offsets = unit_idx + tl.arange(0, BLOCK_SIZE_Y)
    unit_mask = unit_offsets < n_out * n_batch

    mask_offsets = tl.expand_dims(
        unit_offsets, 1
    ) * FFI_PER_NEURON + tl.expand_dims(tl.arange(0, FFI_PER_NEURON), 0)
    mask_mask = mask_offsets < n_weights
    weight_offsets = (
        tl.expand_dims(tl.arange(0, FFI_PER_NEURON), 1)
        + tl.expand_dims(unit_offsets, 0) * FFI_PER_NEURON
    )
    weight_mask = weight_offsets < n_weights

    output_offsets = tl.expand_dims(
        batch_offsets * BLOCK_SIZE_X, 1
    ) + tl.expand_dims(unit_offsets, 0)
    output_mask = output_offsets < n_batch * n_out

    bias = tl.load(bias_p + unit_offsets, mask=unit_mask)
    output = tl.load(output_p + output_offsets, output_mask)
    output += bias
    weights = tl.load(weight_p + weight_offsets, mask=weight_mask)
    mask = tl.load(mask_p + mask_offsets, mask=mask_mask)

    input_offset = tl.expand_dims(batch_offsets, 1) * n_in + mask
    input_mask = input_offset < n_in * n_batch
    inputs = tl.load(input_p + input_offset, input_mask)
    tl.dot(inputs, weights, acc=output, allow_tf32=False)
    tl.store(output_p + output_offsets, output, output_mask)


output = torch.zeros(size=(__BATCH_SIZE, __OUTPUT_DIM), device=__DEVICE)
input = x
weight = ffi_vmap.condensed_weight.T  # shape is ffi, output
bias = ffi_vmap.bias
mask = ffi_vmap.input_mask
n_elements = input.numel()
n_weights = weight.numel()
FFI_PER_NEURON = weight.shape[0]  # first dim is now ffi
BLOCK_SIZE_X = 32
BLOCK_SIZE_Y = 32
n_out = __OUTPUT_DIM
n_in = __INPUT_DIM
n_batch = __BATCH_SIZE
grid = triton.cdiv(input.shape[0], BLOCK_SIZE_X), triton.cdiv(
    weight.shape[1], BLOCK_SIZE_Y
)  # 32 threads per grid block
print(grid)
ffi_triton[grid](
    output,
    input,
    weight,
    bias,
    mask,
    n_out,
    n_in,
    n_batch,
    n_weights,
    FFI_PER_NEURON,
    BLOCK_SIZE_X,
    BLOCK_SIZE_Y,
)
print(output)
