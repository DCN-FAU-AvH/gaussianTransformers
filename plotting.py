"""
Plotting utilities for Gaussian-input transformer statistics.

All summary plots can be generated either directly from activations or from the
.txt files produced by gaussianInputs.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gaussianProbe import load_table_txt


NICEBLUE = "#2f5dd2"
NICEORANGE = "#d98c19"

LayerSpec = Union[str, int]


def set_pub_style() -> None:
    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 3.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


def _savefig(fig: plt.Figure, savepath: Optional[Union[str, os.PathLike]]) -> None:
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=300)


# -----------------------------------------------------------------------------
# Summary plots from saved .txt tables
# -----------------------------------------------------------------------------

def _stack_seed_tables(
    stats_dir: Union[str, os.PathLike],
    stem: str,
    seeds: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Load seed-dependent tables and stack numeric columns.

    Example files:
      moments_s0.txt, moments_s1.txt, ...
      dist_s0.txt, dist_s1.txt, ...

    Returns
    -------
    seeds_loaded:
        Array of seeds actually found.
    stacked:
        Dict where numeric columns have shape (n_seeds, n_layers).
        Non-numeric columns such as 'layer' are copied from the first file.
    """
    stats_dir = Path(stats_dir)

    if seeds is None:
        files = sorted(stats_dir.glob(f"{stem}_s*.txt"))
        seeds_loaded = []
        for f in files:
            seed_str = f.stem.split("_s")[-1]
            try:
                seeds_loaded.append(int(seed_str))
            except ValueError:
                continue
    else:
        seeds_loaded = list(seeds)

    tables = []
    valid_seeds = []

    for seed in seeds_loaded:
        path = stats_dir / f"{stem}_s{seed}.txt"
        if path.exists():
            tables.append(load_table_txt(path))
            valid_seeds.append(seed)

    if not tables:
        raise FileNotFoundError(
            f"No files found for stem={stem!r} in {stats_dir}. "
            f"Expected files like {stem}_s0.txt."
        )

    ref_len = len(tables[0]["x"])
    for seed, table in zip(valid_seeds, tables):
        if len(table["x"]) != ref_len:
            raise ValueError(
                f"Seed {seed} has incompatible length: "
                f"{len(table['x'])}, expected {ref_len}."
            )

    stacked: Dict[str, np.ndarray] = {}

    for key in tables[0].keys():
        first = tables[0][key]

        if key == "layer":
            stacked[key] = first
            continue

        try:
            stacked[key] = np.stack(
                [np.asarray(t[key], dtype=float) for t in tables],
                axis=0,
            )
        except (ValueError, TypeError):
            stacked[key] = first

    return np.asarray(valid_seeds, dtype=int), stacked

def plot_seed_mean_quantile_curve(
    xs: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    label: str,
    color: str,
    ax: Optional[plt.Axes] = None,
    q_low: float = 0.1,
    q_high: float = 0.9,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> plt.Axes:
    """
    Plot mean over seeds with a quantile band.

    Parameters
    ----------
    xs:
        Shape (n_layers,).
    values:
        Shape (n_seeds, n_layers).
    q_low, q_high:
        Quantiles for the band. For example, 0.1 and 0.9 give an 80% band.
    """
    if values.ndim != 2:
        raise ValueError(f"values must have shape (n_seeds, n_layers), got {values.shape}")

    if ax is None:
        _, ax = plt.subplots(figsize=(3.25, 2))

    mean = np.nanmean(values, axis=0)
    lo = np.nanquantile(values, q_low, axis=0)
    hi = np.nanquantile(values, q_high, axis=0)

    ax.plot(xs, mean, color=color, lw=linewidth, ls=linestyle, label=label)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)

    return ax


def plot_from_saved_statistics(
    stats_dir: Union[str, os.PathLike],
    plot_dir: Union[str, os.PathLike],
    seeds: Optional[Sequence[int]] = None,
    show: bool = True,
    q_low: float = 0.1,
    q_high: float = 0.9,
    model_label: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Load seed-dependent .txt statistics and regenerate standard plots.

    Expected files
    --------------
    For each seed s:

      moments_s{s}.txt
      dist_s{s}.txt

    Produces
    --------
    mean_trace_covariance.pdf:
        Mean over seeds with quantile bands for mean_avg and trace_cov.

    sw2_mmd_matched.pdf:
        Mean over seeds with quantile bands for sw2_matched and mmd_matched.
    """
    stats_dir = Path(stats_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    figs: Dict[str, plt.Figure] = {}

    colors = {
        "mean": NICEBLUE,
        "trace": NICEORANGE,
        "sw2": NICEBLUE,
        "mmd": NICEORANGE,
    }

    # ------------------------------------------------------------------
    # Layer moments: mean_avg and trace_cov
    # ------------------------------------------------------------------
    try:
        seeds_loaded, moments = _stack_seed_tables(
            stats_dir=stats_dir,
            stem="moments",
            seeds=seeds,
        )

        xs = np.asarray(moments["x"][0], dtype=float)

        fig, ax1 = plt.subplots(figsize=(3.25, 2), constrained_layout=True)
        ax2 = ax1.twinx()

        ax2.spines["left"].set_visible(False)
        ax2.patch.set_visible(False)

        plot_seed_mean_quantile_curve(
            xs=xs,
            values=np.asarray(moments["mean_avg"], dtype=float),
            ylabel=r"$\mu_\ell$",
            label=r"$\mathbb{E}[h_\ell]$",
            color=colors["mean"],
            ax=ax1,
            q_low=q_low,
            q_high=q_high,
        )

        plot_seed_mean_quantile_curve(
            xs=xs,
            values=np.asarray(moments["trace_cov"], dtype=float),
            ylabel=r"$\mathrm{tr}(\Sigma^{(\ell)})$",
            label=r"$\mathrm{tr}(\Sigma^{(\ell)})$",
            color=colors["trace"],
            ax=ax2,
            q_low=q_low,
            q_high=q_high,
        )

        ax1.set_xlabel(r"Layer $\ell$")
        ax1.set_ylabel(r"$\mu^{(\ell)}$")
        ax2.set_ylabel(r"$\mathrm{tr}(\Sigma^{(\ell)})$")

        xtick_step = max(1, len(xs) // 4)
        ax1.set_xticks(xs[::xtick_step].astype(int))

        ax1.tick_params(axis="x", which="major", labelsize=9, colors="black")

        ax1.spines["left"].set_visible(True)
        ax1.spines["left"].set_color(colors["mean"])
        ax1.yaxis.label.set_color(colors["mean"])
        ax1.tick_params(axis="y", which="both", labelsize=9, colors=colors["mean"])

        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(colors["trace"])
        ax2.yaxis.label.set_color(colors["trace"])
        ax2.tick_params(axis="y", which="both", labelsize=9, colors=colors["trace"])

        if len(xs) > 1:
            ax1.set_xlim(xs.min() - 0.3, xs.max() + 0.3)

        _savefig(fig, plot_dir / f"{model_label}_mean_trace_covariance.pdf" if model_label else plot_dir / "mean_trace_covariance.pdf")

        if show:
            plt.show()

        figs["mean_trace"] = fig

    except FileNotFoundError:
        pass

    # ------------------------------------------------------------------
    # Gaussian distances: SW2 and MMD
    # ------------------------------------------------------------------
    try:
        seeds_loaded, distances = _stack_seed_tables(
            stats_dir=stats_dir,
            stem="dist",
            seeds=seeds,
        )

        xs = np.asarray(distances["x"][0], dtype=float)

        fig, ax1 = plt.subplots(figsize=(3.25, 2), constrained_layout=True)
        ax2 = ax1.twinx()

        ax2.spines["left"].set_visible(False)
        ax2.patch.set_visible(False)

        plot_seed_mean_quantile_curve(
            xs=xs,
            values=np.asarray(distances["sw2_matched"], dtype=float),
            ylabel=r"SW-2",
            label=r"SW$_2$",
            color=colors["sw2"],
            ax=ax1,
            q_low=q_low,
            q_high=q_high,
        )

        plot_seed_mean_quantile_curve(
            xs=xs,
            values=np.asarray(distances["mmd_matched"], dtype=float),
            ylabel=r"MMD",
            label="MMD",
            color=colors["mmd"],
            ax=ax2,
            q_low=q_low,
            q_high=q_high,
        )

        ax1.set_xlabel(r"Layer $\ell$")
        ax1.set_ylabel(r"SW-2")
        ax2.set_ylabel(r"MMD")

        xtick_step = max(1, len(xs) // 4)
        ax1.set_xticks(xs[::xtick_step].astype(int))

        ax1.tick_params(axis="x", which="major", labelsize=9, colors="black")

        ax1.spines["left"].set_visible(True)
        ax1.spines["left"].set_color(colors["sw2"])
        ax1.yaxis.label.set_color(colors["sw2"])
        ax1.tick_params(axis="y", which="both", labelsize=9, colors=colors["sw2"])

        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(colors["mmd"])
        ax2.yaxis.label.set_color(colors["mmd"])
        ax2.tick_params(axis="y", which="both", labelsize=9, colors=colors["mmd"])

        if len(xs) > 1:
            ax1.set_xlim(xs.min() - 0.3, xs.max() + 0.3)

        _savefig(fig, plot_dir / f"{model_label}_sw2_mmd_matched.pdf" if model_label else plot_dir / "sw2_mmd_matched.pdf")

        if show:
            plt.show()

        figs["sw2_mmd"] = fig

    except FileNotFoundError:
        pass

    return figs


def plot_representative_coordinate_distributions_from_table(
    table: Mapping[str, np.ndarray],
    criterion: str = "median_variance",
    layers: Optional[Sequence[LayerSpec]] = None,
    max_panels: int = 6,
    bins: int = 120,
    xlim: Optional[tuple[float, float]] = None,
    density: bool = True,
    savepath: Optional[Union[str, os.PathLike]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot one representative coordinate distribution across selected layers.

    Input should be representative_coordinates.txt loaded with load_table_txt().

    Expected columns:
      layer, x, criterion, coord_idx, sample_idx, value

    Parameters
    ----------
    criterion:
        Which representative coordinate to plot, e.g.
        "median_variance", "max_variance", "median_kurtosis",
        "max_kurtosis", or "min_kurtosis".
    layers:
        Optional layer subset. Examples:
          ["input", 0, 5, 10, 15, 21]
          ["input", "layer_0", "layer_5", "layer_21"]
        If None, evenly spaced layers are selected.
    max_panels:
        Maximum number of panels when layers is None.
    bins:
        Number of histogram bins.
    xlim:
        Fixed x-axis range. If None, a robust global range is inferred from
        the 0.5 and 99.5 percentiles of all plotted values.
    density:
        Use density-normalized histograms.
    """
    required = {"layer", "x", "criterion", "coord_idx", "value"}
    missing = required.difference(table.keys())
    if missing:
        raise KeyError(f"Missing columns in representative-coordinate table: {sorted(missing)}")

    layer_col = np.asarray(table["layer"], dtype=str)
    crit_col = np.asarray(table["criterion"], dtype=str)
    x_col = np.asarray(table["x"], dtype=float)
    coord_col = np.asarray(table["coord_idx"], dtype=int)
    value_col = np.asarray(table["value"], dtype=float)

    available_layers = []
    for lname in layer_col:
        if lname not in available_layers:
            available_layers.append(lname)

    # Restrict to requested criterion.
    crit_mask = crit_col == criterion
    if not np.any(crit_mask):
        available_criteria = sorted(set(crit_col.tolist()))
        raise ValueError(
            f"Criterion {criterion!r} not found. "
            f"Available criteria: {available_criteria}"
        )

    # Resolve layer names.
    if layers is None:
        crit_layers = []
        for lname in available_layers:
            if np.any(crit_mask & (layer_col == lname)):
                crit_layers.append(lname)

        if len(crit_layers) > max_panels:
            idx = np.linspace(0, len(crit_layers) - 1, max_panels, dtype=int)
            selected_layers = [crit_layers[i] for i in idx]
        else:
            selected_layers = crit_layers
    else:
        names_set = set(available_layers)
        selected_layers = []
        for layer in layers:
            if isinstance(layer, str):
                lname = layer if layer in {"input", "final"} or layer.startswith("layer_") else f"layer_{layer}"
            else:
                lname = f"layer_{int(layer)}"

            if lname not in names_set:
                raise ValueError(f"Requested layer {lname!r} not found.")
            selected_layers.append(lname)

    if not selected_layers:
        raise ValueError("No layers selected for plotting.")

    # Collect arrays and metadata.
    arrays = []
    coords = []
    xpos = []

    for lname in selected_layers:
        mask = crit_mask & (layer_col == lname)
        vals = value_col[mask]
        if vals.size == 0:
            continue

        arrays.append(vals)
        coords.append(int(coord_col[mask][0]))
        xpos.append(float(x_col[mask][0]))

    if not arrays:
        raise ValueError(f"No data found for criterion {criterion!r} in selected layers.")

    # Fixed x-axis range.
    all_values = np.concatenate(arrays)

    if xlim is None:
        lo, hi = np.percentile(all_values[np.isfinite(all_values)], [0.5, 99.5])
        if np.isclose(lo, hi):
            pad = 0.05 * abs(hi) if hi != 0 else 1e-3
            lo, hi = lo - pad, hi + pad
        else:
            pad = 0.05 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        xlim = (float(lo), float(hi))

    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)

    ncols = len(arrays) 
    nrows = 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0, 1.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    hist_color = NICEBLUE

    for i, (ax, vals, lname, coord_idx, layer_x) in enumerate(
        zip(axes_flat, arrays, selected_layers, coords, xpos)
    ):
        ax.hist(
            vals,
            bins=bin_edges,
            density=density,
            histtype="stepfilled",
            alpha=0.18,
            color=hist_color,
            edgecolor=hist_color,
            linewidth=1,
        )

        ax.hist(
            vals,
            bins=bin_edges,
            density=density,
            histtype="step",
            color=hist_color,
            linewidth=1,
        )

        ax.set_xlim(*xlim)

        ax.set_xticks([xlim[0] // 2, 0, xlim[1] // 2])
        ax.set_xticklabels([rf"$ {xlim[0] // 2:.0f} $", r"$0$", rf"$ {xlim[1] // 2:.0f} $"])


        ax.set_xticks([xlim[0] // 2, 0, xlim[1] // 2])
        ax.set_xticklabels([rf"$ {xlim[0] // 2:.0f} $", r"$0$", rf"$ {xlim[1] // 2:.0f} $"])

        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels([r"$0.0$", r"$1.0$"])

        if lname == "input":
            label = r"$\ell=0$"
        elif lname.startswith("layer_"):
            label = rf"$\ell={int(lname.split('_')[1]) + 1}$"
        else:
            label = lname

        ax.set_xlabel(label, fontsize=12)


    _savefig(fig, savepath)
    if show:
        plt.show()
    return fig




if __name__ == "__main__":
    set_pub_style()

    model = "modernBERT"  # "modernBERT" or "DeiT"

    plot_from_saved_statistics(
        stats_dir="./" + model + "/stats",
        plot_dir="./" + model + "/plots",
        show=False,
        model_label=model,
    )

    criterion = "median_variance" # "median_kurtosis"

    if model == "modernBERT":
        layers = ["input", 4, 8, 12, 16, 21]
        xlim = (-10, 10)
    elif model == "DeiT":
        layers = ["input", 2, 4, 6, 8, 11]
        xlim = (-4, 4)

    rep = load_table_txt("./" + model + "/stats/representative_coordinates.txt")

    plot_representative_coordinate_distributions_from_table(
        rep,
        criterion= criterion,
        layers = layers, 
        bins=200,
        xlim=xlim,
        savepath="./" + model + "/plots/" + model + "_repCoord_" + criterion + ".pdf",
        show=False,
    )

    sign = "opposite" # "same" or "opposite"
    model = "DeiTmod" 

    figs = plot_from_saved_statistics(    
    stats_dir="./" + model + "/stats/" + sign + "sign",
    plot_dir="./" + model + "/plots/",
    show=False,
    model_label=model + "_" + sign + "sign",
    )