from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import torch
from transformers import AutoModel

from gaussianProbe import _find_encoder_layers, compute_and_save_statistics, print_layer_moments, summarize_layer_moments


LayerSpec = Union[str, int]
ArrayTable = Dict[str, np.ndarray]



def ordered_layer_names(
    names_or_activations: Union[Iterable[str], Mapping[str, Any]],
    layers: Optional[Sequence[LayerSpec]] = None,
    include_input: bool = True,
    include_final: bool = False,
) -> List[str]:
    """
    Return a stable ordered list such as ["input", "layer_0", ..., "layer_L"].

    Parameters
    ----------
    names_or_activations:
        Either an iterable of layer names or a dict keyed by layer name.
    layers:
        Optional subset. Ints are interpreted as layer indices; strings may be
        "input", "final", or explicit names like "layer_7".
    include_input:
        Add "input" when available and layers is None.
    include_final:
        Add "final" when available and layers is None.
    """
    names = set(names_or_activations.keys() if isinstance(names_or_activations, Mapping) else names_or_activations)
    layer_indices = sorted(int(n.split("_")[1]) for n in names if n.startswith("layer_"))

    if layers is None:
        ordered = []
        if include_input and "input" in names:
            ordered.append("input")
        ordered.extend(f"layer_{i}" for i in layer_indices)
        if include_final and "final" in names:
            ordered.append("final")
        return ordered

    ordered: List[str] = []
    for layer in layers:
        if isinstance(layer, str):
            name = layer if layer in {"input", "final"} or layer.startswith("layer_") else f"layer_{layer}"
        else:
            name = f"layer_{int(layer)}"
        if name not in names:
            raise ValueError(f"Layer {name!r} was requested but is not available.")
        ordered.append(name)

    return ordered


QKVSpec = Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Mapping[str, torch.Tensor],
    Callable[[int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]
QKVByLayer = Mapping[int, QKVSpec]


def _as_btd_or_project(
    t: torch.Tensor,
    h: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """
    Convert a prescribed Q/K/V object into a tensor shaped like h: (B, T, D).

    Supported forms:
      - (D, D): interpreted as a right projection matrix, h @ t
      - (T, D): same tokens for every batch element
      - (1, T, D): broadcast over batch
      - (B, T, D): already materialized
    """
    b, seq_len, hidden_size = h.shape
    t = t.to(device=h.device, dtype=h.dtype)

    if t.ndim == 2:
        if t.shape == (hidden_size, hidden_size):
            return h @ t

        if t.shape == (seq_len, hidden_size):
            return t.unsqueeze(0).expand(b, -1, -1)

        raise ValueError(
            f"{name} has shape {tuple(t.shape)}. Expected (D,D) or (T,D), "
            f"with D={hidden_size}, T={seq_len}."
        )

    if t.ndim == 3:
        if t.shape == h.shape:
            return t

        if t.shape == (1, seq_len, hidden_size):
            return t.expand(b, -1, -1)

        raise ValueError(
            f"{name} has shape {tuple(t.shape)}. Expected (B,T,D) or (1,T,D), "
            f"with B={b}, T={seq_len}, D={hidden_size}."
        )

    raise ValueError(f"{name} must have rank 2 or 3, got rank {t.ndim}.")


def _materialize_qkv(
    layer_idx: int,
    h: torch.Tensor,
    qkv_by_layer: QKVByLayer,
    require_all_layers: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return Q, K, V for one layer as tensors of shape (B, T, D).

    If require_all_layers=False and a layer is missing, Q=K=V=h.
    """
    if layer_idx not in qkv_by_layer:
        if require_all_layers:
            raise KeyError(f"Missing prescribed Q/K/V for layer {layer_idx}.")
        return h, h, h

    spec = qkv_by_layer[layer_idx]

    if callable(spec):
        q, k, v = spec(layer_idx, h)
    elif isinstance(spec, Mapping):
        q, k, v = spec["q"], spec["k"], spec["v"]
    else:
        q, k, v = spec

    q = _as_btd_or_project(q, h, f"Q[layer_{layer_idx}]")
    k = _as_btd_or_project(k, h, f"K[layer_{layer_idx}]")
    v = _as_btd_or_project(v, h, f"V[layer_{layer_idx}]")

    return q, k, v


@torch.no_grad()
def prescribed_single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Single-head bidirectional attention.

    q, k, v: (B, T, D)
    returns: (B, T, D)

    chunk_size avoids materializing the full T x T attention matrix at once.
    For seq_len=8192, full attention is usually OK on a large GPU, but chunking
    is safer for fp32 or smaller GPUs.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"Q/K/V must have identical shape. Got Q={q.shape}, K={k.shape}, V={v.shape}."
        )

    _, seq_len, d = q.shape
    scale = scale if scale is not None else d ** -0.5

    kt = k.transpose(-1, -2)

    if chunk_size is None or chunk_size >= seq_len:
        scores = torch.matmul(q, kt) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(probs, v)

    outs = []
    for start in range(0, seq_len, chunk_size):
        stop = min(start + chunk_size, seq_len)
        scores = torch.matmul(q[:, start:stop, :], kt) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        outs.append(torch.matmul(probs, v))

    return torch.cat(outs, dim=1)



@torch.no_grad()
def _deit_feed_forward_no_norm(layer: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    """
    Apply the pretrained DeiT feed-forward block without applying LayerNorm.

    Supports both recent HF DeiT blocks:
        layer.mlp(h)

    and older HF DeiT blocks:
        layer.output(layer.intermediate(h), h)
    where the old output module includes the second residual connection.
    """
    if hasattr(layer, "mlp"):
        return layer.mlp(h)

    if hasattr(layer, "intermediate") and hasattr(layer, "output"):
        # Older HF DeiT API. DeiTOutput.forward(hidden_states, input_tensor)
        # already adds the residual input_tensor, so we return the delta only.
        old_out_with_residual = layer.output(layer.intermediate(h), h)
        return old_out_with_residual - h

    raise AttributeError(
        "Could not find DeiT feed-forward block. Expected either layer.mlp "
        "or layer.intermediate + layer.output."
    )


@torch.no_grad()
def run_deit_layers_prescribed_single_head(
    model: torch.nn.Module,
    x: torch.Tensor,
    qkv_by_layer: QKVByLayer,
    activations: MutableMapping[str, torch.Tensor],
    prefix: str = "layer",
    require_all_layers: bool = True,
    attention_chunk_size: Optional[int] = None,
    apply_original_attention_output_projection: bool = False,
) -> torch.Tensor:
    """
    Run DeiT encoder layers on Gaussian hidden states with:

        h <- h + SingleHeadAttention(Q_l, K_l, V_l)
        h <- h + pretrained_DeiT_FFN_l(h)

    No layernorm_before, no layernorm_after, no final layernorm.

    x must have shape (B, T, D), where for tiny DeiT:
        D = 192

    T can be arbitrary for this direct hidden-state experiment, because we are
    not using patch embeddings or positional embeddings here.
    """
    h = x
    activations["input"] = h.detach()

    layers = _find_encoder_layers(model)

    for layer_idx, layer in enumerate(layers):
        q, k, v = _materialize_qkv(
            layer_idx=layer_idx,
            h=h,
            qkv_by_layer=qkv_by_layer,
            require_all_layers=require_all_layers,
        )

        attn_out = prescribed_single_head_attention(
            q=q,
            k=k,
            v=v,
            chunk_size=attention_chunk_size,
        )

        if apply_original_attention_output_projection:
            # Current HF DeiT attention output projection lives here:
            # layer.attention.output.dense
            if (
                hasattr(layer, "attention")
                and hasattr(layer.attention, "output")
                and hasattr(layer.attention.output, "dense")
            ):
                attn_out = layer.attention.output.dense(attn_out)

            # Some older variants may expose output.dense directly under attention.
            elif hasattr(layer, "attention") and hasattr(layer.attention, "dense"):
                attn_out = layer.attention.dense(attn_out)

            else:
                raise AttributeError(
                    "apply_original_attention_output_projection=True requires "
                    "layer.attention.output.dense or layer.attention.dense."
                )

        h = h + attn_out

        ffn_out = _deit_feed_forward_no_norm(layer, h)
        h = h + ffn_out

        activations[f"{prefix}_{layer_idx}"] = h.detach()

    return h


@torch.no_grad()
def probe_deit_gaussian_prescribed_single_head(
    model_name: str = "facebook/deit-tiny-patch16-224",
    qkv_by_layer: Optional[QKVByLayer] = None,
    batch_size: int = 1,
    seq_len: int = 197,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
    print_summary: bool = True,
    require_all_layers: bool = True,
    attention_chunk_size: Optional[int] = None,
    apply_original_attention_output_projection: bool = False,
) -> Tuple[ArrayTable, Dict[str, torch.Tensor]]:
    """
    Probe tiny DeiT with Gaussian hidden states, replacing multi-head attention
    by prescribed residual + single-head attention.

    For facebook/deit-tiny-patch16-224:
        hidden_size = 192
        num_hidden_layers = 12

    The default seq_len=197 corresponds to 196 image patches + CLS token.
    If you want the distillation-token variant, use seq_len=198 with the
    distilled checkpoint.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()

    model_type = getattr(model.config, "model_type", "")
    if model_type not in {"deit", "vit"}:
        raise ValueError(
            f"This probe is for DeiT/ViT-style models, but got model_type={model_type!r}."
        )

    hidden_size = model.config.hidden_size
    input_dtype = dtype or next(model.parameters()).dtype

    x = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=input_dtype,
    )

    if qkv_by_layer is None:
        raise ValueError(
            "qkv_by_layer is required. For example, call "
            "make_identity_q_positive_k_negative_v_specs(...)."
        )

    activations: Dict[str, torch.Tensor] = {}

    final = run_deit_layers_prescribed_single_head(
        model=model,
        x=x,
        qkv_by_layer=qkv_by_layer,
        activations=activations,
        require_all_layers=require_all_layers,
        attention_chunk_size=attention_chunk_size,
        apply_original_attention_output_projection=apply_original_attention_output_projection,
    )

    activations["final"] = final.detach()

    layers = ordered_layer_names(
        activations,
        include_input=True,
        include_final=False,
    )
    table = summarize_layer_moments(activations, layers)

    if print_summary:
        print_layer_moments(table)

    return table, activations

def make_spd_matrix(
    hidden_size: int,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    seed: int,
    eps: float = 1e-2,
    normalize_spectrum: bool = True,
) -> torch.Tensor:
    """
    Make a symmetric positive definite matrix.

    Returns W = A A^T + eps I.
    If normalize_spectrum=True, rescales W so its largest eigenvalue is about 1.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    A = torch.randn(
        hidden_size,
        hidden_size,
        generator=g,
        device=device,
        dtype=dtype,
    )

    W = A @ A.T
    I = torch.eye(hidden_size, device=device, dtype=dtype)
    W = W + eps * I

    if normalize_spectrum:
        # eigvalsh is safer in float32. Cast back afterward.
        eigvals = torch.linalg.eigvalsh(W.float())
        W = W / eigvals.max().to(dtype=W.dtype).clamp_min(1e-12)

    return W


def make_identity_q_positive_k_negative_v_specs(
    num_layers: int,
    hidden_size: int,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    seed: int = 0,
    eps: float = 1e-2,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Layer-wise prescription:

      W_Q = I
      W_K positive definite
      W_V negative definite

    The runner interprets these as:
      Q = h @ W_Q
      K = h @ W_K
      V = h @ W_V
    """
    I = torch.eye(hidden_size, device=device, dtype=dtype)

    specs: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer_idx in range(num_layers):
        W_K = make_spd_matrix(
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            seed=seed + 10_000 * layer_idx + 1,
            eps=eps,
            normalize_spectrum=True,
        )

        W_V_pos = make_spd_matrix(
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            seed=seed + 10_000 * layer_idx + 2,
            eps=eps,
            normalize_spectrum=True,
        )

        W_V = -W_V_pos

        specs[layer_idx] = {
            "q": I,
            "k": W_K,
            "v": W_V,
        }

    return specs

def make_shared_qkv_definite_specs(
    num_layers: int,
    hidden_size: int,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    seed: int = 0,
    eps: float = 1e-2,
    q_mode: str = "identity",
    sign_mode: str = "opposite",
    normalize_spectrum: bool = True,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Make one shared set of Q/K/V projection matrices and reuse it at all layers.

    Parameters
    ----------
    num_layers:
        Number of transformer layers.

    hidden_size:
        Hidden dimension D.

    q_mode:
        "identity":
            W_Q = I

        "pd":
            W_Q is symmetric positive definite.

    sign_mode:
        "opposite":
            W_K is positive definite and W_V is negative definite.

        "same":
            W_K and W_V are both positive definite.

    normalize_spectrum:
        If True, rescales each generated definite matrix so its largest absolute
        eigenvalue is approximately 1.

    Returns
    -------
    specs:
        Dictionary mapping every layer index to the same shared matrices:

            Q = h @ W_Q
            K = h @ W_K
            V = h @ W_V
    """
    if q_mode not in {"identity", "pd"}:
        raise ValueError(f"q_mode must be 'identity' or 'pd', got {q_mode!r}.")

    if sign_mode not in {"same", "opposite"}:
        raise ValueError(f"sign_mode must be 'same' or 'opposite', got {sign_mode!r}.")

    if q_mode == "identity":
        W_Q = torch.eye(hidden_size, device=device, dtype=dtype)
    else:
        W_Q = make_spd_matrix(
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
            seed=seed + 1,
            eps=eps,
            normalize_spectrum=normalize_spectrum,
        )

    W_K = make_spd_matrix(
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
        seed=seed + 2,
        eps=eps,
        normalize_spectrum=normalize_spectrum,
    )

    W_V_pos = make_spd_matrix(
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
        seed=seed + 3,
        eps=eps,
        normalize_spectrum=normalize_spectrum,
    )

    if sign_mode == "same":
        W_V = W_V_pos
    else:
        W_V = -W_V_pos

    specs: Dict[int, Dict[str, torch.Tensor]] = {
        layer_idx: {
            "q": W_Q,
            "k": W_K,
            "v": W_V,
        }
        for layer_idx in range(num_layers)
    }

    return specs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_name = "facebook/deit-tiny-patch16-224"

    tmp_model = AutoModel.from_pretrained(
        model_name,
        use_safetensors=True,
    )
    num_layers = tmp_model.config.num_hidden_layers
    hidden_size = tmp_model.config.hidden_size
    del tmp_model

    sign = "opposite"  # "same" or "opposite"

    qkv_specs = make_shared_qkv_definite_specs(
        num_layers=num_layers,
        hidden_size=hidden_size,
        device=device,
        dtype=dtype,
        seed=0,
        eps=1e-1,
        q_mode="identity",
        sign_mode= sign,
    )

    seed_list = range(20)

    for seed in seed_list:
        stats, acts = probe_deit_gaussian_prescribed_single_head(
            model_name=model_name,
            qkv_by_layer=qkv_specs,
            batch_size=1,
            seq_len=8192,
            device=device,
            dtype=dtype,
            seed=seed,
            attention_chunk_size=None,
            require_all_layers=True,
            apply_original_attention_output_projection=False,
        )

        compute_and_save_statistics(
            acts,
            out_dir="./DeiTmod/stats/" + sign + "sign",
            layers=None,
            compute_distances=True,
            seed=seed,
        )