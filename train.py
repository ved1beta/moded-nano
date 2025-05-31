import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial # Added partial for hook registration
from pathlib import Path
try:
    import numpy as np
except ImportError:
    print("Warning: NumPy not available, falling back to torch arrays for synthetic data")
    np = None  # Will use torch arrays instead

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
if torch.cuda.is_available():
    torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist

# Check if the GPU supports various features
def check_gpu_features():
    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()
    
    # Check if FP8 is supported
    has_fp8 = has_cuda and hasattr(torch, 'float8_e4m3fn')
    
    # Check if FlexAttention is available
    try:
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
        has_flex_attention = True
    except ImportError:
        has_flex_attention = False
        
    return has_cuda, has_fp8, has_flex_attention

# Check GPU features
HAS_CUDA, HAS_FP8, HAS_FLEX_ATTENTION = check_gpu_features()

# Import or define dummy FlexAttention if not available
if HAS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
else:
    print("FlexAttention not available, will use standard attention")
    
    # Dummy BlockMask class for compatibility
    class BlockMask:
        @classmethod
        def from_kv_blocks(cls, *args, **kwargs):
            return None

#torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng
# These operations will only be used if FP8 is supported

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    def impl(x: Tensor, w: Tensor):
        if not HAS_FP8:
            # Fallback for non-FP8 systems
            return x @ w.T, x, w
            
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    if HAS_FP8:
        return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)
    else:
        return x @ w.T, x, w

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        if not HAS_FP8:
            # Fallback for non-FP8 systems
            return grad @ w_f8.T, x_f8.T @ grad
            
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    if HAS_FP8:
        return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)
    else:
        return x_f8, w_f8.T.contiguous().T

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if hasattr(torch, 'bfloat16') and torch.cuda.is_available():
        X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        
        # Create empty param groups for single-GPU case
        if world_size == 1:
            for size in {p.numel() for p in params}:
                group = dict(params=[p for p in params if p.numel() == size])
                param_groups.append(group)
        else:
            # For multi-GPU case, maintain original logic
            for size in {p.numel() for p in params}:
                b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
                group = dict(params=[p for p in params if p.numel() == size],
                             update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
                param_groups.append(group)
                
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        if self.world_size == 1:
            # Simplified version for single-GPU training
            for group in self.param_groups:
                params: list[Tensor] = group["params"]
                for p in params:
                    if p.grad is None:
                        continue
                    
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    
                    # Apply the gradient
                    p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
        else:
            # Original multi-GPU implementation
            for group in self.param_groups:
                update_buffer: Tensor = group["update_buffer"]
                update_buffer_views: list[Tensor] = group["update_buffer_views"]
                # generate weight updates in distributed fashion
                params: list[Tensor] = group["params"]
                handle = None
                params_world = None
                def update_prev():
                    handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        p_world.add_(g_world.view_as(p_world),
                                    alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
                for base_i in range(len(params))[::self.world_size]:
                    if base_i + self.rank < len(params):
                        p = params[base_i + self.rank]
                        g = p.grad
                        assert g is not None
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        buf.lerp_(g, 1 - group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                    else:
                        g = update_buffer_views[self.rank]
                    if base_i > 0:
                        update_prev()
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                    params_world = params[base_i : base_i + self.world_size]
                update_prev()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        # Only use FP8 if the hardware supports it
        self.use_fp8 = use_fp8 and HAS_FP8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        # Try FP8 operations only if supported
        if self.use_fp8 and self.training and HAS_FP8:
            try:
                _x = x.flatten(0, -2)
                out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
                return out.reshape(*x.shape[:-1], -1)
            except Exception as e:
                # Fall back to regular linear if FP8 operations fail
                return F.linear(x, self.weight.type_as(x))
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12
        self.register_buffer('mask', None, persistent=False)

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        
        # Only require batch size 1 for FlexAttention
        if HAS_FLEX_ATTENTION:
            assert B == 1, "Must use batch size = 1 for FlexAttention"
            
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm
        q, k = self.rotary(q), self.rotary(k)
        
        # Handle value embeddings safely
        if ve is not None:
            try:
                v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
            except Exception as e:
                v = self.lambdas[0] * v
        else:
            v = self.lambdas[0] * v
        
        # Use FlexAttention only if available AND sequence length is at least 128
        # For smaller sequences, use standard attention which is more efficient
        if HAS_FLEX_ATTENTION and T >= 128:
            y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                               block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        else:
            # Standard attention implementation
            q = q.transpose(1, 2)  # [B, nh, T, hd]
            k = k.transpose(1, 2)  # [B, nh, T, hd]
            v = v.transpose(1, 2)  # [B, nh, T, hd]
            
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * self.attn_scale
            
            # Create causal mask if not already created or if too small
            if self.mask is None or self.mask.size(0) < T:
                mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
                mask = mask.bool().view(1, 1, T, T)
                self.register_buffer('mask', mask, persistent=False)
            
            # Apply causal mask
            att = att.masked_fill(self.mask[:, :, :T, :T], float('-inf'))
            
            # Softmax and apply to values
            att = F.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous()  # [B, T, nh, hd]
            
        y = y.reshape(B, T, self.num_heads * self.head_dim)  # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # Skip attention for certain layers if needed to save compute
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v, n):
    return ((int(v) + n - 1) // n) * n

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=HAS_FP8, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_()
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        if not HAS_FLEX_ATTENTION:
            return None, None
            
        BLOCK_SIZE = 128
        
        # Pad the input sequence to be a multiple of BLOCK_SIZE
        pad_size = (BLOCK_SIZE - len(input_seq) % BLOCK_SIZE) % BLOCK_SIZE
        if pad_size > 0:
            input_seq = torch.cat([input_seq, torch.ones(pad_size, dtype=input_seq.dtype, device=input_seq.device) * 50256])
        
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=input_seq.device)
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        # Long-short SWA block masks
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        # Get value embeddings
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # Distribute value embeddings across layers
        num_layers = len(self.blocks)
        
        # Always have value embeddings at the beginning and end
        if num_layers <= 6:
            # For small models, use value embeddings in all layers
            ve_list = ve * (num_layers // 3 + 1)  # Repeat to ensure we have enough
            ve_list = ve_list[:num_layers]  # Truncate to match number of layers
        else:
            # For larger models, distribute value embeddings: beginning, middle, and end
            ve_list = [None] * num_layers
            # First 3 layers
            for i in range(min(3, num_layers)):
                ve_list[i] = ve[i % 3]
            # Last 3 layers
            for i in range(max(0, num_layers - 3), num_layers):
                ve_list[i] = ve[i % 3]
        
        assert len(ve_list) == len(self.blocks)

        # Create block masks based on whether flex attention is available
        if HAS_FLEX_ATTENTION:
            long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
            # Make sure we have the right number of block masks
            if len(self.blocks) == 12:
                block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
            else:
                # For different numbers of blocks, alternate between long and short
                block_masks = []
                for i in range(len(self.blocks)):
                    block_masks.append(long_bm if i % 2 == 0 else short_bm)
        else:
            # For non-FlexAttention systems, use None for blockmasks
            block_masks = [None] * len(self.blocks)
            
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None])

        # U-net design
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve_list[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        # softcapping
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        return loss

# -----------------------------------------------------------------------------
# Flexible Data Loaders for different dataset formats

def _load_data_shard(file: Path):
    """Load a binary data shard in our specific format with 256 int32 header"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: Hugging Face datasets library not found. Install with 'pip install datasets'")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Warning: tiktoken not found. Install with 'pip install tiktoken'")

def prepare_huggingface_dataset(dataset_name, tokenizer_name="gpt2", text_column="text", max_length=None, split="train", config=None):
    """Load and tokenize a dataset from Hugging Face Hub"""
    if not HAS_DATASETS or not HAS_TIKTOKEN:
        raise ImportError("Hugging Face datasets and tiktoken libraries are required")
    
    print(f"Loading dataset {dataset_name} (config: {config}, split: {split})")
    dataset = load_dataset(dataset_name, config, split=split, trust_remote_code=True)
    
    # Get tokenizer
    enc = tiktoken.get_encoding(tokenizer_name)
    
    # Handle special tokens properly
    try:
        # Try with allowed_special="all" to avoid errors with special tokens
        eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    except Exception as e:
        print(f"Warning: Error with special token encoding: {e}")
        # Fallback to a standard token if special token fails
        eot_token = enc.encode(".", allowed_special=set())[0]
    
    # Tokenize the dataset
    all_tokens = []
    for item in dataset:
        if text_column not in item:
            raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {list(item.keys())}")
        
        # Add end of text token before each text
        tokens = [eot_token]
        
        # Encode with allowed_special=set() to avoid errors with special tokens in the text
        tokens.extend(enc.encode(item[text_column], allowed_special=set()))
        
        # Optionally truncate
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
            
        all_tokens.extend(tokens)
    
    # Convert to tensor
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.int32)
    
    # Save as binary file in our format
    os.makedirs("data/cache", exist_ok=True)
    
    # Split into train and validation
    val_size = min(len(tokens_tensor) // 10, 100000)  # 10% or max 100k tokens for validation
    
    # Save validation set
    save_tokens_to_bin(tokens_tensor[:val_size], "data/cache/hf_val_000000.bin")
    
    # Save training set
    save_tokens_to_bin(tokens_tensor[val_size:], "data/cache/hf_train_000000.bin")
    
    return "data/cache/hf_train_*.bin", "data/cache/hf_val_*.bin"

def save_tokens_to_bin(tokens, filename):
    """Save tokens tensor to our binary format"""
    # Create header (256 int32s)
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240520  # magic number
    header[1] = 1  # version
    header[2] = len(tokens)  # number of tokens
    
    # Convert tokens to uint16 if needed
    if tokens.dtype != torch.uint16:
        tokens = tokens.to(torch.uint16)
    
    # Write to file
    with open(filename, 'wb') as f:
        f.write(header.numpy().tobytes())
        f.write(tokens.numpy().tobytes())
    
    print(f"Saved {len(tokens)} tokens to {filename}")

def create_synthetic_data(vocab_size=50257, num_tokens=10000):
    """Create synthetic data for testing when no real data is available"""
    print("Creating synthetic data for testing...")
    
    # Create directories for synthetic data
    os.makedirs("data/cache", exist_ok=True)
    
    # Generate random tokens
    tokens = torch.randint(0, vocab_size, (num_tokens,), dtype=torch.uint16)
    
    # Write training file
    save_tokens_to_bin(tokens, "data/cache/synthetic_train_000000.bin")
    
    # Write validation file (use different random tokens)
    val_tokens = torch.randint(0, vocab_size, (num_tokens // 5,), dtype=torch.uint16)
    save_tokens_to_bin(val_tokens, "data/cache/synthetic_val_000000.bin")
    
    return "data/cache/synthetic_train_*.bin", "data/cache/synthetic_val_*.bin"

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    """
    Distributed data generator that handles various error cases and data sources.
    
    Args:
        filename_pattern: Glob pattern for data files
        batch_size: Batch size across all GPUs
        rank: Current process rank
        world_size: Total number of processes
    """
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {filename_pattern}")
    
    assert batch_size % world_size == 0, f"Batch size ({batch_size}) must be divisible by world size ({world_size})"
    local_batch_size = batch_size // world_size
    
    # Create an infinite iterator over the files
    def file_iterator():
        while True:
            for file in files:
                yield file
    
    file_iter = file_iterator()
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        # If we need more tokens than what's left in the current shard, get next shard
        if pos + batch_size + 1 >= len(tokens):
            try:
                tokens, pos = _load_data_shard(next(file_iter)), 0
            except Exception as e:
                print(f"Error loading next data shard: {e}. Restarting from first file.")
                file_iter = file_iterator()
                tokens, pos = _load_data_shard(next(file_iter)), 0
        
        # Get the tokens for this rank
        try:
            buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
            inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
            pos += batch_size
            yield inputs, targets
        except Exception as e:
            print(f"Error generating batch: {e}. Skipping to next shard.")
            tokens, pos = _load_data_shard(next(file_iter)), 0

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    dataset = "synthetic"  # 'synthetic', 'binary', or 'huggingface'
    train_files = "data/cache/*_train_*.bin"  # For binary dataset
    val_files = "data/cache/*_val_*.bin"  # For binary dataset
    huggingface_dataset = "rotten_tomatoes"  # Dataset name for Hugging Face
    huggingface_text_column = "text"  # Text column for Hugging Face datasets
    hf_dataset_config = None  # Config name for Hugging Face dataset
    val_tokens = 10240  # How many tokens of validation data
    train_seq_len = 1024  # Sequence length for training
    val_seq_len = 1024  # Sequence length for validation
    
    # Model configuration
    vocab_size = 50257  # GPT-2 vocabulary size
    num_layers = None  # Auto-determined based on GPU memory if None
    num_heads = None  # Auto-determined based on GPU memory if None
    model_dim = None  # Auto-determined based on GPU memory if None
    
    # optimization
    num_iterations = 1000  # Number of iterations to run
    cooldown_frac = 0.4  # Fraction of training spent cooling down the learning rate
    
    # evaluation and logging
    val_loss_every = 100  # Every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = True  # Whether to save checkpoints
args = Hyperparameters()

# Command line argument parsing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a GPT model on any dataset")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "binary", "huggingface"],
                        help="Dataset type: synthetic, binary, or huggingface")
    parser.add_argument("--hf_dataset", type=str, default="rotten_tomatoes",
                       help="Hugging Face dataset name (if using huggingface dataset type)")
    parser.add_argument("--hf_text_column", type=str, default="text",
                       help="Text column name in the Hugging Face dataset")
    parser.add_argument("--hf_dataset_config", type=str, default=None,
                       help="Config name for Hugging Face dataset (e.g., '20220301.en' for Wikipedia)")
    parser.add_argument("--train_files", type=str, default="data/cache/*_train_*.bin",
                       help="Glob pattern for training data files (if using binary dataset type)")
    parser.add_argument("--val_files", type=str, default="data/cache/*_val_*.bin",
                       help="Glob pattern for validation data files (if using binary dataset type)")
    parser.add_argument("--train_seq_len", type=int, default=1024,
                       help="Sequence length for training")
    parser.add_argument("--val_seq_len", type=int, default=1024,
                       help="Sequence length for validation")
    parser.add_argument("--num_iterations", type=int, default=1000,
                       help="Number of training iterations")
    parser.add_argument("--val_every", type=int, default=100,
                       help="Run validation every N iterations")
    parser.add_argument("--save_checkpoint", action="store_true",
                       help="Save model checkpoint after training")
    parser.add_argument("--model_dim", type=int, default=None,
                       help="Model dimension (if None, auto-determined based on GPU)")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of transformer layers (if None, auto-determined based on GPU)")
    parser.add_argument("--num_heads", type=int, default=None,
                       help="Number of attention heads (if None, auto-determined based on GPU)")
    
    cmd_args = parser.parse_args()
    
    # Update hyperparameters based on command line args
    args.dataset = cmd_args.dataset
    args.huggingface_dataset = cmd_args.hf_dataset
    args.huggingface_text_column = cmd_args.hf_text_column
    args.hf_dataset_config = cmd_args.hf_dataset_config
    args.train_files = cmd_args.train_files
    args.val_files = cmd_args.val_files
    args.train_seq_len = cmd_args.train_seq_len
    args.val_seq_len = cmd_args.val_seq_len
    args.num_iterations = cmd_args.num_iterations
    args.val_loss_every = cmd_args.val_every
    args.save_checkpoint = cmd_args.save_checkpoint
    args.num_layers = cmd_args.num_layers
    args.num_heads = cmd_args.num_heads
    args.model_dim = cmd_args.model_dim

# Set up distributed training if available, otherwise use single GPU
try:
    # torchrun sets these env variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Initialize distributed process group if multiple GPUs
    if world_size > 1:
        assert torch.cuda.is_available()
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl")
        dist.barrier()
    else:
        # Single GPU setup
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"Error initializing distributed environment: {e}")
    # Fallback to single GPU or CPU
    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

master_process = (rank == 0)  # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# Begin by printing basic info
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Device: {device}, Rank: {rank}/{world_size}")
if torch.cuda.is_available():
    print0(f"GPU: {torch.cuda.get_device_name(device)}")
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi())
print0("="*100)

# Set up the dataset
if args.dataset == 'synthetic':
    print0("Using synthetic data for training", console=True)
    args.train_files, args.val_files = create_synthetic_data(args.vocab_size)
elif args.dataset == 'huggingface':
    print0(f"Using Hugging Face dataset: {args.huggingface_dataset}", console=True)
    try:
        args.train_files, args.val_files = prepare_huggingface_dataset(
            args.huggingface_dataset, 
            text_column=args.huggingface_text_column,
            max_length=args.train_seq_len,
            config=args.hf_dataset_config
        )
    except ImportError:
        print0("Failed to load Hugging Face dataset. Falling back to synthetic data.", console=True)
        args.train_files, args.val_files = create_synthetic_data(args.vocab_size)
elif args.dataset == 'binary':
    print0(f"Using binary data files: {args.train_files}", console=True)
    # Verify files exist
    train_files = sorted(glob.glob(args.train_files))
    val_files = sorted(glob.glob(args.val_files))
    
    if not train_files:
        print0(f"No training files found matching {args.train_files}", console=True)
        print0("Falling back to synthetic data", console=True)
        args.train_files, args.val_files = create_synthetic_data(args.vocab_size)
    elif not val_files:
        print0(f"No validation files found matching {args.val_files}", console=True)
        print0("Using training files for validation", console=True)
        args.val_files = args.train_files
    else:
        print0(f"Found {len(train_files)} training files and {len(val_files)} validation files", console=True)
else:
    print0(f"Unknown dataset type: {args.dataset}. Using synthetic data.", console=True)
    args.train_files, args.val_files = create_synthetic_data(args.vocab_size)

########################################
#    Configure model based on GPU      #
########################################

# Set model size based on available GPU memory
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # in GB
    print0(f"GPU Memory: {gpu_memory:.2f} GB", console=True)
    
    # Adjust model size based on available memory
    if args.num_layers is None or args.num_heads is None or args.model_dim is None:
        if gpu_memory > 32:  
            num_layers = 12
            num_heads = 12
            model_dim = 768
        elif gpu_memory > 16:  
            num_layers = 8
            num_heads = 8
            model_dim = 512
        elif gpu_memory > 8:  
            num_layers = 6
            num_heads = 6
            model_dim = 384
        else:  # Lower-end GPU
            num_layers = 4
            num_heads = 4
            model_dim = 256
        
        # Override with user settings if provided
        args.num_layers = args.num_layers or num_layers
        args.num_heads = args.num_heads or num_heads
        args.model_dim = args.model_dim or model_dim
    
    # Try to estimate memory requirements and adjust sequence length if needed
    try:
        gpu_mem = torch.cuda.get_device_properties(device).total_memory
        free_mem = gpu_mem - torch.cuda.memory_allocated()
        
        # Rough estimate of memory needed per token
        mem_per_token = args.model_dim * 4 * 4  # 4 bytes per float * 4x overhead for activations
        
        max_tokens = free_mem // (mem_per_token * 2)  # Leave half of memory free
        max_seq_len = min(args.train_seq_len, max_tokens // 2)  # Safe margin
        
        if max_seq_len < args.train_seq_len:
            print0(f"Adjusting sequence length from {args.train_seq_len} to {max_seq_len} to fit GPU memory", console=True)
            args.train_seq_len = max_seq_len
            args.val_seq_len = max_seq_len
    except Exception as e:
        print0(f"Error estimating memory: {e}. Using default sequence lengths.", console=True)
else:
    # CPU-only mode (very small model)
    args.num_layers = args.num_layers or 4
    args.num_heads = args.num_heads or 4
    args.model_dim = args.model_dim or 256

print0(f"Model config: layers={args.num_layers}, heads={args.num_heads}, dim={args.model_dim}", console=True)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, num_heads=args.num_heads, model_dim=args.model_dim,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).to(device)

# Convert embeddings to bfloat16 if supported
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.bfloat16()

# Broadcast parameters in distributed setting
if world_size > 1:
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s)
adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
# Use fused Adam if on CUDA
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=torch.cuda.is_available())
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay
def get_lr(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule: linearly increase
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
def get_window_size_blocks(step: int):
    x = step / args.num_iterations # progress in training
    assert 0 <= x <= 1
    # Linearly increase the block-wise sliding window size over training 128 -> 1792
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

########################################
#      Overlap Communication Setup     #
########################################

# Create parameter buckets for better overlap
def create_buckets(params, bucket_size_mb=25):
    """Group parameters into buckets of approximately bucket_size_mb MB each"""
    buckets = []
    current_bucket = []
    current_size = 0

    # Sort parameters by size (largest first) for better bucketing
    sorted_params = sorted(params, key=lambda p: p.numel(), reverse=True)

    for param in sorted_params:
        param_size_mb = param.numel() * param.element_size() / (1024 * 1024)

        if current_size + param_size_mb > bucket_size_mb and current_bucket:
            buckets.append(current_bucket)
            current_bucket = [param]
            current_size = param_size_mb
        else:
            current_bucket.append(param)
            current_size += param_size_mb

    if current_bucket:
        buckets.append(current_bucket)

    return buckets

# Create buckets for all parameters
all_params = [p for p in model.parameters() if p.requires_grad]

# Only set up bucketed gradient communication for multi-GPU training
if world_size > 1:
    param_buckets = create_buckets(all_params)
    
    print0(f"Created {len(param_buckets)} gradient buckets")
    for i, bucket in enumerate(param_buckets):
        total_size = sum(p.numel() * p.element_size() for p in bucket) / (1024 * 1024)
        print0(f"Bucket {i}: {len(bucket)} params, {total_size:.1f} MB")
    
    # Bucket state tracking
    bucket_ready_count = [0] * len(param_buckets)
    bucket_handles = [None] * len(param_buckets)
    param_to_bucket = {}
    
    # Map each parameter to its bucket index
    for bucket_idx, bucket in enumerate(param_buckets):
        for param in bucket:
            param_to_bucket[param] = bucket_idx
    
    def _gradient_hook(param: Tensor):
        """Called when a parameter's gradient is ready"""
        if param.grad is None:
            return
    
        bucket_idx = param_to_bucket[param]
        bucket_ready_count[bucket_idx] += 1
    
        # Check if all parameters in this bucket are ready
        if bucket_ready_count[bucket_idx] == len(param_buckets[bucket_idx]):
            # All-reduce this bucket
            bucket_grads = [p.grad for p in param_buckets[bucket_idx]]
    
            # For multi-tensor operations, we can reduce them together
            if len(bucket_grads) == 1:
                handle = dist.all_reduce(bucket_grads[0], op=dist.ReduceOp.AVG, async_op=True)
            else:
                # Use multi-tensor all-reduce for efficiency
                handle = dist.all_reduce_coalesced(bucket_grads, op=dist.ReduceOp.AVG, async_op=True)
    
            bucket_handles[bucket_idx] = handle
    
    # Register hooks for all parameters
    print0("Registering bucketed gradient hooks...")
    for param in all_params:
        param.register_post_accumulate_grad_hook(_gradient_hook)
    
    def wait_for_gradients():
        """Wait for all gradient reductions to complete and reset bucket state"""
        for handle in bucket_handles:
            if handle is not None:
                handle.wait()
    
        # Reset state for next iteration
        for i in range(len(bucket_ready_count)):
            bucket_ready_count[i] = 0
            bucket_handles[i] = None
else:
    # For single GPU, no need to wait for gradients
    def wait_for_gradients():
        pass

def debug_print_tokens(tokens, vocab_size=50257, max_tokens=20):
    """Print some information about the tokens to verify dataset"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        # Show token distribution
        token_counts = {}
        for t in tokens[:1000]:
            token_counts[int(t)] = token_counts.get(int(t), 0) + 1
        # Get top 5 tokens
        top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 tokens in sample: {top_tokens}")
        
        # Try to decode some tokens if tiktoken is available
        decoded = enc.decode(tokens[:max_tokens].tolist())
        print(f"Sample text: '{decoded}'")
    except Exception as e:
        print(f"Error decoding tokens: {e}")
        # Fallback to just printing token IDs
        print(f"First {max_tokens} tokens: {tokens[:max_tokens].tolist()}")

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = world_size * args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        
        # Only reduce validation loss in multi-GPU mode
        if world_size > 1:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    
    if step == 0 and rank == 0:
        print0("======= DATASET DEBUG INFO =======", console=True)
        print0(f"Using dataset type: {args.dataset}", console=True)
        print0(f"Files pattern: {args.train_files}", console=True)
        debug_print_tokens(inputs.cpu())
        print0("==================================", console=True)
    
    model(inputs, targets, get_window_size_blocks(step)).backward()
    wait_for_gradients()

    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) # momentum warmup for muon
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    # step the optimizers
    for opt in optimizers:
        opt.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

# Only destroy process group in multi-GPU mode
if world_size > 1:
    dist.destroy_process_group()


