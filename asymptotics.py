import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

from datetime import date
from scipy.integrate import solve_ivp


def simulate(
    A,                     # (d,d) array or callable A(t)->(d,d)
    B,                     # (d,d) array or callable B(t)->(d,d); assumed symmetric
    V,                     # (d,d) array or callable V(t)->(d,d);
    c,                     # (d,) array or callable c(t)->(d,)
    mu0: np.ndarray,       # (d,)
    Sigma0: np.ndarray,    # (d,d)
    t_span: tuple[float, float],
    t_eval: np.ndarray | None = None,
    TF: bool = True,  # True: Gaussian transformer ODE, False: Kalman filter PDE
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str = "RK45",
):
    """
    Simulates:
        d/dt mu    = (A(t) + V(t)) mu + V(t) Σ B(t) mu + c(t)
        d/dt Σ     = A(t) Σ + Σ A(t)^T + gamma * ( V(t) Σ B(t) Σ + Σ B^T(t) Σ V^T(t) )
    with mu(0)=mu0, Σ(0)=Sigma0, where
        gamma = 1.0 if TF=True (Gaussian transformer case)
        gamma = 0.5 if TF=False (Kalman filter case)

    A, B, V: either fixed (d,d) arrays or callables A(t)->(d,d), B(t)->(d,d), V(t)->(d,d).
    c: either fixed (d,) array or callable c(t)->(d,).
    """
    mu0 = np.asarray(mu0, dtype=float).reshape(-1)
    Sigma0 = np.asarray(Sigma0, dtype=float)
    d = mu0.size

    # Helpers to get A(t), B(t)
    if callable(A):
        def A_of_t(t):
            At = np.asarray(A(t), dtype=float)
            assert At.shape == (d, d), "A(t) must be (d,d)"
            return At
    else:
        A = np.asarray(A, dtype=float)
        assert A.shape == (d, d), "A must be (d,d)"
        def A_of_t(t): return A

    if callable(B):
        def B_of_t(t):
            Bt = np.asarray(B(t), dtype=float)
            assert Bt.shape == (d, d), "B(t) must be (d,d)"
            return Bt
    else:
        B = np.asarray(B, dtype=float)
        assert B.shape == (d, d), "B must be (d,d)"
        def B_of_t(t): return B

    if callable(V):
        def V_of_t(t):
            Vt = np.asarray(V(t), dtype=float)
            assert Vt.shape == (d, d), "V(t) must be (d,d)"
            return Vt
    else:
        V = np.asarray(V, dtype=float)
        assert V.shape == (d, d), "V must be (d,d)"
        def V_of_t(t): return V

    if callable(c):
        def c_of_t(t):
            ct = np.asarray(c(t), dtype=float).reshape(-1)
            assert ct.shape == (d,), "c(t) must be (d,)"
            return ct
    else:
        c = np.asarray(c, dtype=float).reshape(-1)
        assert c.shape == (d,), "c must be (d,)"
        def c_of_t(t): return c

    # Symmetrize initial Sigma
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)

    def pack(mu, Sigma): return np.concatenate([mu, Sigma.reshape(-1)])
    def unpack(y):
        mu = y[:d]
        Sigma = y[d:].reshape(d, d)
        # keep Sigma on the symmetric manifold
        Sigma = 0.5 * (Sigma + Sigma.T)
        return mu, Sigma

    def rhs(t, y):
        mu, Sigma = unpack(y)
        At = A_of_t(t)
        Bt = B_of_t(t)
        Vt = V_of_t(t)
        ct = c_of_t(t)
        gamma = 1.0 if TF else 0.5

        dmu = (At + Vt) @ mu + Vt @ Sigma @ Bt @ mu + ct
        dSigma = At @ Sigma + Sigma @ At.T + gamma * Vt @ Sigma @ Bt @ Sigma + gamma * Sigma @ Bt.T @ Sigma @ Vt.T
        dSigma = 0.5 * (dSigma + dSigma.T)  # enforce symmetry in the vector field
        return pack(dmu, dSigma)


    y0 = pack(mu0, Sigma0)

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol, method=method)

    # Unpack solution
    t = sol.t
    mu_traj = np.zeros((t.size, d))
    Sigma_traj = np.zeros((t.size, d, d))
    for i in range(t.size):
        mu_traj[i], Sigma_traj[i] = unpack(sol.y[:, i])

    return {"t": t, "mu": mu_traj, "Sigma": Sigma_traj, "solver": sol}



def set_pub_style() -> None:
    mpl.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.55,
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
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _savefig(fig: plt.Figure, path: Optional[Union[str, os.PathLike]]) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)


def _cov_ellipse(
    mean: np.ndarray,
    cov: np.ndarray,
    level: float = 1.0,
    edgecolor: str = "0.2",
    facecolor: str = "none",
    alpha: float = 1.0,
    linewidth: float = 1.1,
    linestyle: str = "-",
) -> Ellipse:
    """
    Return an ellipse corresponding to mean + level * covariance contour.
    For visual comparison, level is interpreted as the number of covariance
    standard deviations along principal axes.
    """
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * level * np.sqrt(vals)

    return Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=linestyle,
    )


def _set_equal_padded_limits(
    ax: plt.Axes,
    points: np.ndarray,
    pad_frac: float = 0.12,
    fixed_xlim: Optional[Tuple[float, float]] = None,
    fixed_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if fixed_xlim is not None:
        ax.set_xlim(*fixed_xlim)
    if fixed_ylim is not None:
        ax.set_ylim(*fixed_ylim)

    # if fixed_xlim is not None and fixed_ylim is not None:
    #     ax.set_aspect("equal", adjustable="box")
    #     return

    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.size == 0:
        return

    xmin, ymin = finite.min(axis=0)
    xmax, ymax = finite.max(axis=0)

    dx = xmax - xmin
    dy = ymax - ymin
    span = max(dx, dy)

    if span <= 0:
        span = 1.0

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1.0 + pad_frac)

    if fixed_xlim is None:
        ax.set_xlim(xmid - half, xmid + half)
    if fixed_ylim is None:
        ax.set_ylim(ymid - half, ymid + half)


def make_gaussian_plot_single_pdf_pub(
    t: np.ndarray,
    mu_traj1: np.ndarray,
    Sigma_traj: np.ndarray,
    mu_target: Optional[np.ndarray] = None,
    Sigma_target: Optional[np.ndarray] = None,
    proj: Tuple[int, int] = (0, 1),
    pdf_out: Union[str, os.PathLike] = "gaussian_trajectory.pdf",
    levels: Sequence[float] = (1.0, 2.0),
    n_ellipses: int = 7,
    figsize: Tuple[float, float] = (2, 1.5),
    fixed_xlim: Optional[Tuple[float, float]] = None,
    fixed_ylim: Optional[Tuple[float, float]] = None,
    show_start_end_markers: bool = True,
    show: bool = False,
) -> Union[str, os.PathLike]:
    """
    Gaussian mean/covariance trajectory plot.

    """
    set_pub_style()

    i, j = proj
    mu = np.asarray(mu_traj1)[:, [i, j]]
    Sigma = np.asarray(Sigma_traj)[:, [i, j]][:, :, [i, j]]

    if mu_target is not None:
        mu_target_2d = np.asarray(mu_target)[[i, j]]
    else:
        mu_target_2d = None

    if Sigma_target is not None:
        Sigma_target_2d = np.asarray(Sigma_target)[[i, j]][:, [i, j]]
    else:
        Sigma_target_2d = None

    traj_color = "#2f5dd2"       # blue, consistent with SW2
    start_color = "0.2"
    target_color = "#d98c19"     # orange, consistent with MMD
    ellipse_color = "#2f5dd2"
    target_ellipse_color = "#d98c19"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(
        mu[:, 0],
        mu[:, 1],
        color=traj_color,
        linewidth=1.55,
        zorder=3,
    )

    if show_start_end_markers:
        ax.plot(
            mu[0, 0],
            mu[0, 1],
            marker="o",
            markersize=3.6,
            color=start_color,
            linestyle="None",
            zorder=5,
        )
        ax.plot(
            mu[-1, 0],
            mu[-1, 1],
            marker="o",
            markersize=3.6,
            color=traj_color,
            linestyle="None",
            zorder=5,
        )

    if mu_target_2d is not None:
        ax.plot(
            mu_target_2d[0],
            mu_target_2d[1],
            marker="s",
            markersize=3.5,
            color=target_color,
            linestyle="None",
            zorder=6,
        )

    # Draw covariance ellipses at evenly spaced times.
    ellipse_ids = np.linspace(0, len(mu) - 1, min(n_ellipses, len(mu)), dtype=int)

    for idx in ellipse_ids:
        alpha = 0.20 + 0.45 * idx / max(1, len(mu) - 1)
        for level in levels:
            ell = _cov_ellipse(
                mu[idx],
                Sigma[idx],
                level=level,
                edgecolor=ellipse_color,
                facecolor="none",
                alpha=alpha,
                linewidth=0.9,
            )
            ax.add_patch(ell)

    if mu_target_2d is not None and Sigma_target_2d is not None:
        for level in levels:
            ell = _cov_ellipse(
                mu_target_2d,
                Sigma_target_2d,
                level=level,
                edgecolor=target_ellipse_color,
                facecolor="none",
                alpha=0.75,
                linewidth=1.0,
                linestyle="--",
            )
            ax.add_patch(ell)

    # Include trajectory, ellipses, target in limit calculation.
    limit_points = [mu]

    for idx in ellipse_ids:
        # Conservative extent estimate from covariance eigenvalues.
        vals = np.linalg.eigvalsh(0.5 * (Sigma[idx] + Sigma[idx].T))
        radius = max(levels) * np.sqrt(np.clip(vals.max(), 0.0, None))
        corners = np.array([
            mu[idx] + [-radius, -radius],
            mu[idx] + [-radius, radius],
            mu[idx] + [radius, -radius],
            mu[idx] + [radius, radius],
        ])
        limit_points.append(corners)

    if mu_target_2d is not None:
        limit_points.append(mu_target_2d[None, :])

    all_points = np.vstack(limit_points)
    _set_equal_padded_limits(
        ax,
        all_points,
        fixed_xlim=fixed_xlim,
        fixed_ylim=fixed_ylim,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.tick_params(axis="both", which="major", labelsize=9)

    _savefig(fig, pdf_out)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_out

def make_gaussian_shared_legend_pdf(
    pdf_out: Union[str, os.PathLike] = "gaussian_shared_legend.pdf",
    ncol: int = 4,
    figsize: Tuple[float, float] = (5.8, 0.42),
    include_target_covariance: bool = False,
    show: bool = False,
) -> Union[str, os.PathLike]:
    """
    Create a standalone horizontal legend for several Gaussian trajectory figures.
    """
    set_pub_style()

    traj_color = "#2f5dd2"
    start_color = "0.15"
    target_color = "#d98c19"

    handles = [
        Line2D(
            [0],
            [0],
            color=traj_color,
            linewidth=1.55,
            label=r"$\mu_t$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=start_color,
            linestyle="None",
            markersize=4.0,
            label=r"$\mu_0$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=traj_color,
            linestyle="None",
            markersize=4.0,
            label=r"$\mu_T$",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=target_color,
            linestyle="None",
            markersize=4.0,
            label=r"$\mu_\infty$",
        ),

        Line2D(
            [0],
            [0],
            color=traj_color,
            linewidth=1.0,
            label=r"$\Sigma_t$",
        ),

    ]

    if include_target_covariance:
        handles.append(
            Line2D(
                [0],
                [0],
                color=target_color,
                linestyle="--",
                linewidth=1.0,
                label=r"$\Sigma_\infty$ contours",
            )
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.25,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    _savefig(fig, pdf_out)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_out



# -----------------------------
# Demo run
# -----------------------------
d = 2

np.random.seed(0)

signA = "indef" # "stable" or "unstable" or "indef"

if signA == "unstable":
    A = np.array([[1.0, -1.0], [0.0, 0.5]])
elif signA == "stable":
    A = - np.array([[1.0, -1.0], [0.0, 0.5]])
elif signA == "indef":
    A = np.random.rand(d, d) * 2 - 1  

print("Eigenvalues of A:", np.linalg.eigvals(A))

signB = "PD" 

if signB == "PD":
    B = np.array([[2.0, 1.0],[1.0, 2.0]])
elif signB == "ND":
    B0 = np.random.rand(d, d)
    B = - B0 @ B0.T  # ensure symmetric negative definite

print("Eigenvalues of B:", np.linalg.eigvals(B))

typeV = "Id" #"ND"

if typeV == "Id":
    eta = -1
    V = eta * np.eye(d) 
elif typeV == "PD":
    V0 = np.random.rand(d, d)
    V = V0 @ V0.T  # ensure symmetric positive definite
elif typeV == "ND":
    V = np.array([[-1.0, 0.5],[0.5, -1.0]])

print("Eigenvalues of V:", np.linalg.eigvals(V))


mu0 = np.array([-1.0, -1.0])
Sigma0 = np.array([[0.4, 0.2],[0.2, 0.2]])
Sigma0 = 0.5 * (Sigma0 + Sigma0.T)  # ensure symmetry

print("Sigma0: ", Sigma0)
muinf = np.array([1,2])

tsteps = 500
tf = 100
t_eval = np.linspace(0, tf, tsteps)

TF = True  # True for Gaussian transformer, False for Kalman-Bucy filter

today = date.today().strftime("%Y%m%d")
filename = f"A{signA}_B{signB}_V{typeV}_{today}"
pdf_out = "plots/meanMatching_" + filename + ".pdf"

out1 = simulate(A=A, B=B, V=V, c = np.zeros(d), mu0=mu0, Sigma0=Sigma0, 
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            t_eval=t_eval, TF = TF, method="RK45")

print(np.shape(out1["mu"]), np.shape(out1["Sigma"]))

SigmaInf = out1["Sigma"][-1]
c = - (A + V + V @ SigmaInf @ B) @ muinf

out_TF = simulate(A=A, B=B, V=V, c = c, mu0=mu0, Sigma0=Sigma0, 
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            t_eval=t_eval, TF = TF, method="RK45")

make_gaussian_plot_single_pdf_pub(
    t=out_TF["t"],
    mu_traj1=out_TF["mu"],
    Sigma_traj=out_TF["Sigma"],
    mu_target=muinf,
    Sigma_target=None,
    proj=(0, 1),
    pdf_out="plots/asymptoticMean_A-" + signA + ".pdf",
)

generate_legend = False
if generate_legend:
    make_gaussian_shared_legend_pdf(
        pdf_out="plots/gaussian_shared_legend.pdf",
        ncol=5,
    )