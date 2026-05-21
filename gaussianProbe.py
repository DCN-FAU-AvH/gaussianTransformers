"""
Core utilities for probing a transformer with Gaussian inputs and computing statistics.

It computes activations/statistics,
writes statistics to plain .txt files, and loads those .txt files later so plots can be
adjusted without rerunning the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel

LayerSpec = Union[str, int]
ArrayTable = Dict[str, np.ndarray]


# -----------------------------------------------------------------------------
# Generic layer / tensor helpers
# -----------------------------------------------------------------------------

def flatten_activations(act: torch.Tensor) -> torch.Tensor:
    """Return activations as an (N, d) matrix, preserving the hidden dimension."""
    return act.reshape(-1, act.size(-1))


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


def layer_x_positions(layer_names: Sequence[str]) -> np.ndarray:
    """Map input -> 0, layer_k -> k + 1, final -> len(layer_names)-1."""
    xs = []
    for i, name in enumerate(layer_names):
        if name == "input":
            xs.append(0)
        elif name.startswith("layer_"):
            xs.append(int(name.split("_")[1]) + 1)
        else:
            xs.append(i)
    return np.asarray(xs, dtype=float)


@torch.no_grad()
def run_encoder_layers_direct(
    model: torch.nn.Module,
    x: torch.Tensor,
    activations: MutableMapping[str, torch.Tensor],
    prefix: str = "layer",
) -> torch.Tensor:
    """
    Run Gaussian hidden states directly through encoder blocks.
    Needed for DeiT/ViT, whose top-level forward expects pixel_values,
    not inputs_embeds.
    """
    h = x
    activations["input"] = h.detach()

    for i, block in enumerate(_find_encoder_layers(model)):
        try:
            y = block(h, output_attentions=False)
        except TypeError:
            try:
                y = block(h)
            except TypeError:
                y = block(hidden_states=h, output_attentions=False)

        h = y[0] if isinstance(y, tuple) else y
        activations[f"{prefix}_{i}"] = h.detach()

    return h


# -----------------------------------------------------------------------------
# .txt table I/O
# -----------------------------------------------------------------------------

def save_table_txt(path: Union[str, os.PathLike], table: Mapping[str, Sequence[Any]], fmt: str = "%.10g") -> None:
    """
    Save a column-oriented table to a plain text file using a commented header.

    String columns are supported. They are written using fmt="%s" while numeric
    columns use the provided numeric format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(table.keys())
    if not keys:
        raise ValueError("Cannot save an empty table.")

    n = len(next(iter(table.values())))
    if any(len(v) != n for v in table.values()):
        raise ValueError("All table columns must have the same length.")

    rows = list(zip(*(table[k] for k in keys)))
    formats = []
    for k in keys:
        arr = np.asarray(table[k])
        formats.append("%s" if arr.dtype.kind in {"U", "S", "O"} else fmt)

    np.savetxt(path, np.asarray(rows, dtype=object), header=" ".join(keys), comments="# ", fmt=formats)


def load_table_txt(path: Union[str, os.PathLike]) -> ArrayTable:
    """
    Load a table saved by save_table_txt.

    Returns a dict of numpy arrays. The "layer" column is returned as strings;
    all other columns are returned as floats when possible.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        header = None
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                header = stripped.lstrip("#").strip().split()
                break
    if header is None:
        raise ValueError(f"No header found in {path}")

    raw = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if raw.shape == ():
        raw = np.asarray([raw])

    out: ArrayTable = {}
    for name in raw.dtype.names:
        arr = raw[name]
        if name == "layer":
            out[name] = arr.astype(str)
        else:
            try:
                out[name] = arr.astype(float)
            except (TypeError, ValueError):
                out[name] = arr
    return out


# -----------------------------------------------------------------------------
# Moment and covariance summaries
# -----------------------------------------------------------------------------

@torch.no_grad()
def marginal_moments(x: torch.Tensor, eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """
    Compute per-coordinate mean, variance, skewness, and kurtosis for an (N, d) matrix.
    """
    x = flatten_activations(x) if x.ndim != 2 else x
    mean = x.mean(dim=0)
    xc = x - mean
    var = xc.pow(2).mean(dim=0)
    std = torch.sqrt(var + eps)
    skew = xc.pow(3).mean(dim=0) / (std.pow(3) + eps)
    kurt = xc.pow(4).mean(dim=0) / (std.pow(4) + eps)
    return {"mean": mean, "var": var, "skew": skew, "kurt": kurt}


@torch.no_grad()
def summarize_layer_moments(
    activations: Mapping[str, torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
) -> ArrayTable:
    """
    Compact one-row-per-layer summary.

    Saved columns:
      layer, x,
      mean_avg,
      std_min
      std_max
      std_avg,
      trace_cov,
      skew_avg, skew_std,
      kurt_avg, kurt_std
    """
    layer_names = list(layer_names or ordered_layer_names(activations))

    rows: Dict[str, List[Any]] = {
        "layer": [],
        "x": [],
        "mean_avg": [],
        "std_min": [],
        "std_max": [],
        "std_avg": [],
        "trace_cov": [],
        "skew_avg": [],
        "skew_std": [],
        "kurt_avg": [],
        "kurt_std": [],
    }

    xs = layer_x_positions(layer_names)

    for xpos, name in zip(xs, layer_names):
        mm = marginal_moments(activations[name])

        std = torch.sqrt(mm["var"].clamp_min(0.0))
        trace_cov = mm["var"].sum()

        rows["layer"].append(name)
        rows["x"].append(xpos)
        rows["mean_avg"].append(mm["mean"].mean().item())
        rows["std_avg"].append(std.mean().item())
        rows["std_min"].append(std.min().item())
        rows["std_max"].append(std.max().item())
        rows["trace_cov"].append(trace_cov.item())
        rows["skew_avg"].append(mm["skew"].mean().item())
        rows["skew_std"].append(mm["skew"].std(unbiased=False).item())
        rows["kurt_avg"].append(mm["kurt"].mean().item())
        rows["kurt_std"].append(mm["kurt"].std(unbiased=False).item())

    return {k: np.asarray(v) for k, v in rows.items()}


def print_layer_moments(table: Mapping[str, np.ndarray]) -> None:
    """Pretty-print the compact layer moment table."""
    print(
        f"{'layer':<10} {'mean':>9} {'std':>9} {'trace':>11} | "
        f"{'skew':>9} {'kurt':>9}"
    )
    print("-" * 68)

    for i, layer in enumerate(table["layer"]):
        print(
            f"{str(layer):<10} "
            f"{table['mean_avg'][i]:9.4g} "
            f"{table['std_avg'][i]:9.4g} "
            f"{table['trace_cov'][i]:11.4g} | "
            f"{table['skew_avg'][i]:9.4g} "
            f"{table['kurt_avg'][i]:9.4g}"
        )


# Backward-compatible alias for your old function name.
def compute_moments(tensor: torch.Tensor) -> Dict[str, float]:
    table = summarize_layer_moments({"layer": tensor}, ["layer"])
    return {k: float(v[0]) for k, v in table.items() if k not in {"layer", "x"}}


@torch.no_grad()
def representative_coordinate_indices(
    act: torch.Tensor,
    criteria: Sequence[str] = (
        "median_variance",
        "max_variance",
        "median_kurtosis",
        "max_kurtosis",
        "min_kurtosis",
    ),
) -> Dict[str, int]:
    """
    Select representative hidden coordinates from one activation tensor.

    Criteria:
      median_variance  : coordinate closest to median variance
      max_variance     : coordinate with largest variance
      median_kurtosis  : coordinate closest to median kurtosis
      max_kurtosis     : coordinate with largest kurtosis
      min_kurtosis     : coordinate with smallest kurtosis
    """
    x = flatten_activations(act)
    mm = marginal_moments(x)

    var = mm["var"]
    kurt = mm["kurt"]

    out: Dict[str, int] = {}

    for criterion in criteria:
        if criterion == "median_variance":
            target = var.median()
            out[criterion] = torch.argmin(torch.abs(var - target)).item()

        elif criterion == "max_variance":
            out[criterion] = torch.argmax(var).item()

        elif criterion == "median_kurtosis":
            target = kurt.median()
            out[criterion] = torch.argmin(torch.abs(kurt - target)).item()

        elif criterion == "max_kurtosis":
            out[criterion] = torch.argmax(kurt).item()

        elif criterion == "min_kurtosis":
            out[criterion] = torch.argmin(kurt).item()

        else:
            raise ValueError(f"Unknown coordinate criterion: {criterion}")

    return out

# -----------------------------------------------------------------------------
# Model probing
# -----------------------------------------------------------------------------

def _find_encoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        if hasattr(encoder, "layer"):
            return encoder.layer
        if hasattr(encoder, "layers"):
            return encoder.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError(
        "Could not locate encoder layers. Inspect the model architecture and "
        "adapt _find_encoder_layers for this model."
    )


def register_encoder_layer_hooks(
    model: torch.nn.Module,
    activations: MutableMapping[str, torch.Tensor],
    prefix: str = "layer",
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register hooks on encoder blocks and return their removable handles."""
    handles = []
    for idx, layer in enumerate(_find_encoder_layers(model)):
        name = f"{prefix}_{idx}"

        def hook_fn(module, inputs, output, name=name):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[name] = hidden.detach()

        handles.append(layer.register_forward_hook(hook_fn))
    return handles


@torch.no_grad()
def probe_model_with_gaussian(
    model_name: str = "answerdotai/ModernBERT-base",
    batch_size: int = 1,
    seq_len: int = 2048,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None,
    print_summary: bool = True,
) -> Tuple[ArrayTable, Dict[str, torch.Tensor]]:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()

    hidden_size = model.config.hidden_size
    input_dtype = dtype or next(model.parameters()).dtype
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=input_dtype)

    activations: Dict[str, torch.Tensor] = {}

    model_type = getattr(model.config, "model_type", "")
    if model_type in {"deit", "vit"}:
        final = run_encoder_layers_direct(model, x, activations)
        activations["final"] = final.detach()
    else:
        handles = register_encoder_layer_hooks(model, activations)
        try:
            output = model(inputs_embeds=x)
        finally:
            for handle in handles:
                handle.remove()

        activations["input"] = x.detach()
        activations["final"] = output.last_hidden_state.detach()

    layers = ordered_layer_names(activations, include_input=True, include_final=False)
    table = summarize_layer_moments(activations, layers)
    if print_summary:
        print_layer_moments(table)
    return table, activations

# -----------------------------------------------------------------------------
# Gaussian distances: implementation for mean/cov/sampling/SW2/MMD
# -----------------------------------------------------------------------------

@torch.no_grad()
def mean_and_cov(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return empirical mean and covariance for an (N, d) sample matrix."""
    x = flatten_activations(x) if x.ndim != 2 else x
    mu = x.mean(dim=0)
    xc = x - mu
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    return mu, cov


@torch.no_grad()
def sample_gaussian(mu: torch.Tensor, cov: torch.Tensor, n: int) -> torch.Tensor:
    """Sample n observations from N(mu, cov), with a small Cholesky fallback."""
    # cov = 0.5 * (cov + cov.T)
    # eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    # jitter = 0.0
    # for _ in range(5):
    #     try:
    #         L = torch.linalg.cholesky(cov + jitter * eye)
    #         break
    #     except RuntimeError:
    #         jitter = 1e-6 if jitter == 0 else jitter * 10
    # else:
    #     raise RuntimeError("Cholesky decomposition failed even after adding jitter.")
    L = torch.linalg.cholesky(cov)
    return mu + torch.randn(n, mu.numel(), device=mu.device, dtype=mu.dtype) @ L.T


@torch.no_grad()
def sliced_wasserstein_2_to_gaussian(
    x: torch.Tensor,
    mu: Optional[torch.Tensor] = None,
    cov: Optional[torch.Tensor] = None,
    n_proj: int = 128,
) -> float:
    """
    Sliced W2 between empirical x and a Gaussian.

    If mu/cov are omitted, the target is the moment-matched Gaussian of x.
    """
    x = flatten_activations(x) if x.ndim != 2 else x
    n, d = x.shape
    if mu is None or cov is None:
        mu, cov = mean_and_cov(x)

    z = sample_gaussian(mu, cov, n)
    theta = torch.randn(n_proj, d, device=x.device, dtype=x.dtype)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

    px = theta @ x.T
    pz = theta @ z.T
    px_sorted, _ = px.sort(dim=1)
    pz_sorted, _ = pz.sort(dim=1)
    return torch.sqrt((px_sorted - pz_sorted).pow(2).mean(dim=1).mean()).item()


@torch.no_grad()
def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * a).sum(dim=1, keepdim=True) + (b * b).sum(dim=1, keepdim=True).T - 2.0 * (a @ b.T)


@torch.no_grad()
def _median_heuristic_sigma(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    z = torch.cat([x, y], dim=0)
    z_shuf = z[torch.randperm(z.shape[0], device=z.device)]
    d2 = ((z - z_shuf) ** 2).sum(dim=1)
    return torch.sqrt(torch.median(d2).clamp_min(eps) / 2.0)


@torch.no_grad()
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: Optional[torch.Tensor] = None, unbiased: bool = True) -> float:
    """RBF-kernel MMD, returned as sqrt(MMD^2)."""
    x = flatten_activations(x) if x.ndim != 2 else x
    y = flatten_activations(y) if y.ndim != 2 else y
    n, m = x.shape[0], y.shape[0]
    if unbiased and (n < 2 or m < 2):
        raise ValueError("Unbiased MMD requires at least two samples in each set.")

    sigma = _median_heuristic_sigma(x, y) if sigma is None else sigma
    sigma2 = sigma * sigma

    kxx = torch.exp(-_pairwise_sq_dists(x, x) / (2.0 * sigma2))
    kyy = torch.exp(-_pairwise_sq_dists(y, y) / (2.0 * sigma2))
    kxy = torch.exp(-_pairwise_sq_dists(x, y) / (2.0 * sigma2))

    if unbiased:
        kxx_mean = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
        kyy_mean = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    else:
        kxx_mean = kxx.mean()
        kyy_mean = kyy.mean()

    mmd2 = torch.clamp(kxx_mean + kyy_mean - 2.0 * kxy.mean(), min=0.0)
    return torch.sqrt(mmd2).item()


@torch.no_grad()
def summarize_gaussian_distances(
    activations: Mapping[str, torch.Tensor],
    layer_names: Optional[Sequence[str]] = None,
    n_proj: int = 128,
    compute_mmd: bool = True,
) -> ArrayTable:
    """Compute SW2 and optional MMD to each layer's moment-matched Gaussian."""
    layer_names = list(layer_names or ordered_layer_names(activations))
    rows: Dict[str, List[Any]] = {"layer": [], "x": [], "sw2_matched": []}
    if compute_mmd:
        rows["mmd_matched"] = []

    for xpos, name in zip(layer_x_positions(layer_names), layer_names):
        x = flatten_activations(activations[name])
        mu, cov = mean_and_cov(x)
        rows["layer"].append(name)
        rows["x"].append(xpos)
        rows["sw2_matched"].append(sliced_wasserstein_2_to_gaussian(x, mu, cov, n_proj=n_proj))
        if compute_mmd:
            rows["mmd_matched"].append(mmd_rbf(x, sample_gaussian(mu, cov, x.shape[0])))

    return {k: np.asarray(v) for k, v in rows.items()}


# -----------------------------------------------------------------------------
# One-call compute/save convenience
# -----------------------------------------------------------------------------
@torch.no_grad()
def save_representative_coordinates_txt(
    activations: Mapping[str, torch.Tensor],
    path: Union[str, os.PathLike],
    layer_names: Optional[Sequence[str]] = None,
    criteria: Sequence[str] = (
        "median_variance",
        "max_variance",
        "median_kurtosis",
        "max_kurtosis",
        "min_kurtosis",
    ),
    max_points_per_coordinate: Optional[int] = None,
) -> Path:
    """
    Save selected activation coordinates as a compact text table.

    Columns:
      layer, x, criterion, coord_idx, sample_idx, value

    This is enough to replot representative histograms/densities without
    saving the full activation tensors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    layer_names = list(layer_names or ordered_layer_names(activations))

    rows: Dict[str, List[Any]] = {
        "layer": [],
        "x": [],
        "criterion": [],
        "coord_idx": [],
        "sample_idx": [],
        "value": [],
    }

    for xpos, lname in zip(layer_x_positions(layer_names), layer_names):
        x = flatten_activations(activations[lname])
        selected = representative_coordinate_indices(x, criteria=criteria)

        n = x.shape[0]

        if max_points_per_coordinate is None or max_points_per_coordinate >= n:
            sample_idx = torch.arange(n, device=x.device)
        else:
            sample_idx = torch.linspace(
                0,
                n - 1,
                max_points_per_coordinate,
                device=x.device,
            ).long()

        sample_idx_cpu = sample_idx.detach().cpu().numpy()

        for criterion, coord_idx in selected.items():
            values = x[sample_idx, coord_idx].detach().cpu().numpy()

            rows["layer"].extend([lname] * len(values))
            rows["x"].extend([xpos] * len(values))
            rows["criterion"].extend([criterion] * len(values))
            rows["coord_idx"].extend([coord_idx] * len(values))
            rows["sample_idx"].extend(sample_idx_cpu.tolist())
            rows["value"].extend(values.tolist())

    save_table_txt(path, rows)
    return path

def compute_and_save_statistics(
    activations: Mapping[str, torch.Tensor],
    out_dir: Union[str, os.PathLike],
    layers: Optional[Sequence[LayerSpec]] = None,
    compute_distances: bool = False,
    n_proj: int = 128,
    save_representative_coords: bool = True,
    representative_criteria: Sequence[str] = (
        "median_variance",
        "median_kurtosis",
    ),
    max_points_per_coordinate: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Compute text-based statistics and save them as .txt files.

    Writes seed-dependent files with short names:

      - moments_s{seed}.txt      layer-wise mean/std/skew/kurt/trace summaries
      - dist_s{seed}.txt         SW2/MMD distances to moment-matched Gaussian
      - coords_s{seed}.txt       representative coordinate samples

    If seed is None, writes:

      - moments.txt
      - dist.txt
      - coords.txt
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_names = (
        ordered_layer_names(activations, layers=layers)
        if layers is not None
        else ordered_layer_names(activations)
    )

    suffix = f"_s{seed}" if seed is not None else ""

    filenames = {
        "moments": f"moments{suffix}.txt",
        "distances": f"dist{suffix}.txt",
        "coordinates": f"coords{suffix}.txt",
    }

    paths: Dict[str, Path] = {}

    moments = summarize_layer_moments(activations, layer_names)
    paths["moments"] = out_dir / filenames["moments"]
    save_table_txt(paths["moments"], moments)

    if compute_distances:
        distances = summarize_gaussian_distances(
            activations,
            layer_names,
            n_proj=n_proj,
            compute_mmd=True,
        )
        paths["distances"] = out_dir / filenames["distances"]
        save_table_txt(paths["distances"], distances)

    if save_representative_coords:
        paths["coordinates"] = out_dir / filenames["coordinates"]
        save_representative_coordinates_txt(
            activations=activations,
            path=paths["coordinates"],
            layer_names=layer_names,
            criteria=representative_criteria,
            max_points_per_coordinate=max_points_per_coordinate,
        )

    return paths


if __name__ == "__main__":
    
    model_name = "facebook/deit-tiny-patch16-224" #"answerdotai/ModernBERT-base" #"facebook/deit-tiny-patch16-224",
    batch_size = 1
    seq_len = 8192
    seed_list = range(20)
    compute_distances = True

    if model_name == "facebook/deit-tiny-patch16-224":
        out_dir = f"./DeiT/stats"
    elif model_name == "answerdotai/ModernBERT-base":
        out_dir = f"./modernBERT/stats"

    for seed in seed_list:
        stats, acts = probe_model_with_gaussian(
            model_name = model_name, 
            batch_size=batch_size,
            seq_len = seq_len,
            seed=seed,
        )
        compute_and_save_statistics(
            acts,
            out_dir= out_dir,
            layers=None,
            compute_distances=True,
            seed=seed,
        )
