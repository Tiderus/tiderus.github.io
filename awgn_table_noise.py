"""
AWGN + Non-Euclidean Riemannian Spaces  ×  Table-Driven Noise

Noise table  T = (1, 3, 4, −1, −4, 5, −2, 0, 2, 0, −3, −5)
──────────────────────────────────────────────────────────────
Properties of T
  mean  =  0.0        (perfectly balanced: positive and negative sum to zero)
  std   ≈  3.03       (same as a N(0, 3.03²) Gaussian for fair comparison)
  range = [−5, 5]     (bounded, unlike unbounded Gaussian tails)
  N     = 12 entries  (period-12 deterministic cycling)

Two table-noise modes
  Cyclic   — T[i % 12] in order, fully deterministic, zero randomness
  Sampled  — draw entries uniformly from T with replacement (seeded RNG)

Both modes normalise by TABLE_STD so σ has the same meaning as in the
Gaussian AWGN case.  Noise is applied via the Riemannian pipeline:
    v_raw  →  project_tangent(p, v_raw)  →  Exp_p(σ · v)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from awgn_non_euclidean import (
    EuclideanManifold, Sphere, HyperbolicSpace,
    _make_euclidean_signal, _make_sphere_signal, _make_hyperbolic_signal,
)


# ═══════════════════════════════════════════════════════════════════════════
# The noise table
# ═══════════════════════════════════════════════════════════════════════════

TABLE: np.ndarray = np.array(
    [1, 3, 4, -1, -4, 5, -2, 0, 2, 0, -3, -5], dtype=float
)
TABLE_MEAN: float = float(TABLE.mean())   # exactly 0.0
TABLE_STD:  float = float(TABLE.std())    # ≈ 3.028


# ═══════════════════════════════════════════════════════════════════════════
# Noise sources
# ═══════════════════════════════════════════════════════════════════════════

class CyclicTableNoise:
    """
    Deterministic noise by cycling through T[0], T[1], …, T[11], T[0], …

    Each call to .sample() advances an internal pointer so that successive
    calls consume the table without overlap.  Reset with offset=0.
    """

    def __init__(self, table: np.ndarray = TABLE, offset: int = 0):
        self.table = table
        self._pos  = offset

    def sample(self, shape: tuple, _rng=None) -> np.ndarray:
        n   = int(np.prod(shape))
        idx = np.arange(self._pos, self._pos + n) % len(self.table)
        self._pos += n
        return (self.table[idx] / TABLE_STD).reshape(shape)

    def reset(self, offset: int = 0) -> None:
        self._pos = offset


class SampledTableNoise:
    """
    Stochastic noise drawn uniformly at random from T (with replacement).

    After normalisation samples have unit variance, making σ comparable
    to the Gaussian AWGN case.
    """

    def __init__(self, table: np.ndarray = TABLE):
        self.table = table

    def sample(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        idx = rng.integers(0, len(self.table), size=shape)
        return (self.table[idx] / TABLE_STD).reshape(shape)


class GaussianNoise:
    """Wrapper around standard_normal for uniform call signature."""

    def sample(self, shape: tuple, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Riemannian AWGN with arbitrary noise source
# ═══════════════════════════════════════════════════════════════════════════

def riemannian_awgn(manifold, points: np.ndarray, sigma: float,
                    noise_source, rng=None) -> np.ndarray:
    """
    Apply AWGN on any Riemannian manifold using a pluggable noise source.

    For each point p:
        v_raw = noise_source.sample(p.shape, rng)
        v     = manifold.project_tangent(p, v_raw)
        p̃    = manifold.exp_map(p, σ · v)
    """
    noisy = np.empty_like(points)
    for i, p in enumerate(points):
        v_raw = noise_source.sample(p.shape, rng)
        v     = manifold.project_tangent(p, v_raw)
        noisy[i] = manifold.exp_map(p, sigma * v)
    return noisy


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_sphere_wireframe(ax, alpha: float = 0.07) -> None:
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:25j]
    ax.plot_surface(
        np.cos(u) * np.sin(v),
        np.sin(u) * np.sin(v),
        np.cos(v),
        alpha=alpha, color="gray", linewidth=0,
    )


def _draw_poincare_boundary(ax) -> None:
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray",
                         linewidth=1.5, linestyle="--")
    ax.add_patch(circle)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Table analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_table_analysis(seed: int = 42) -> None:
    """
    Four-panel deep-dive into the noise table:
        (a) Values as a bar chart
        (b) Discrete distribution vs matching Gaussian PDF
        (c) 10 000 random draws (empirical histogram)
        (d) Cyclic sequence repeated 3× (shows periodicity)
    """
    rng = np.random.default_rng(seed)
    sampler = SampledTableNoise()
    large   = sampler.sample((10_000,), rng) * TABLE_STD  # un-normalised for display

    x = np.linspace(-7, 7, 300)
    gauss_pdf = (
        np.exp(-x ** 2 / (2 * TABLE_STD ** 2))
        / (TABLE_STD * np.sqrt(2 * np.pi))
    )

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle(
        r"Noise Table  T = (1, 3, 4, −1, −4, 5, −2, 0, 2, 0, −3, −5)"
        f"\n  mean = {TABLE_MEAN:.1f}   std = {TABLE_STD:.3f}   N = {len(TABLE)}",
        fontsize=11,
    )

    # (a) bar chart ─────────────────────────────────────────────────────────
    ax = axes[0]
    colors = ["tomato" if t >= 0 else "steelblue" for t in TABLE]
    ax.bar(range(len(TABLE)), TABLE, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(TABLE)))
    ax.set_xticklabels([f"T[{i}]" for i in range(len(TABLE))], rotation=45, fontsize=7)
    ax.set_title("(a) Table values")
    ax.set_ylabel("Value")

    # (b) population distribution vs Gaussian ───────────────────────────────
    ax = axes[1]
    unique, counts = np.unique(TABLE, return_counts=True)
    ax.bar(unique, counts / len(TABLE) / 1, width=0.5,   # probability mass
           alpha=0.75, color="goldenrod", label="Table PMF")
    ax.plot(x, gauss_pdf, "k-", linewidth=2,
            label=f"Gaussian N(0, {TABLE_STD:.1f}²)")
    ax.set_title("(b) PMF vs Gaussian PDF")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density / Probability")
    ax.legend(fontsize=8)

    # (c) empirical histogram of 10 k random draws ──────────────────────────
    ax = axes[2]
    bins = np.arange(-6.25, 6.5, 0.5)
    ax.hist(large, bins=bins, density=True, alpha=0.75,
            color="mediumorchid", label="10 000 random draws")
    ax.plot(x, gauss_pdf, "k-", linewidth=2, label="Gaussian")
    ax.set_title("(c) 10 000 sampled draws")
    ax.set_xlabel("Value")
    ax.legend(fontsize=8)

    # (d) cyclic sequence (3 full periods) ──────────────────────────────────
    ax = axes[3]
    cyc = CyclicTableNoise()
    seq = cyc.sample((3 * len(TABLE),)) * TABLE_STD  # un-normalised
    t   = np.arange(len(seq))
    ax.stem(t, seq, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    for period in range(1, 4):
        ax.axvline(period * len(TABLE) - 0.5, color="gray",
                   linestyle=":", linewidth=1)
    ax.set_title("(d) Cyclic sequence (3 periods)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.set_xticks(np.arange(0, 3 * len(TABLE), len(TABLE)))
    ax.set_xticklabels([f"Period {i+1}" for i in range(3)])

    plt.tight_layout()
    plt.savefig("awgn_table_analysis.png", dpi=150)
    plt.show()
    print("Saved awgn_table_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Manifold × noise-source comparison  (3 × 3 grid)
# ═══════════════════════════════════════════════════════════════════════════

def plot_manifold_comparison(snr_db: float = 10.0,
                             n_points: int = 300,
                             seed: int = 42) -> None:
    """
    3 rows (Euclidean | Sphere S² | Hyperbolic H²)  ×
    3 cols (Gaussian  | Cyclic    | Sampled)
    """
    rng = np.random.default_rng(seed)

    euc = EuclideanManifold(n=2)
    sph = Sphere(n=3)
    hyp = HyperbolicSpace(n=2)

    sig_e = _make_euclidean_signal(n_points)
    sig_s = _make_sphere_signal(n_points)
    sig_h = _make_hyperbolic_signal(n_points)

    noise_specs = [
        ("Gaussian",      GaussianNoise()),
        ("Cyclic table",  CyclicTableNoise()),
        ("Sampled table", SampledTableNoise()),
    ]

    BLUE, RED = "steelblue", "tomato"
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        f"AWGN noise sources on Riemannian manifolds  (target SNR = {snr_db} dB)\n"
        r"Table  T = (1, 3, 4, −1, −4, 5, −2, 0, 2, 0, −3, −5)",
        fontsize=12,
    )

    for col, (noise_label, noise_src) in enumerate(noise_specs):
        # ── Euclidean  R² ───────────────────────────────────────────────────
        sigma_e = euc.sigma_for_snr(sig_e, snr_db)
        if isinstance(noise_src, CyclicTableNoise):
            noise_src.reset()
        noisy_e = riemannian_awgn(euc, sig_e, sigma_e, noise_src,
                                   np.random.default_rng(seed + col))
        snr_e = euc.snr_db(sig_e, noisy_e)

        ax = fig.add_subplot(3, 3, col + 1)
        ax.scatter(*noisy_e.T, s=5, alpha=0.45, color=RED)
        ax.scatter(*sig_e.T,   s=5, alpha=0.90, color=BLUE)
        ax.set_title(f"Euclidean R²\n{noise_label}  SNR={snr_e:.1f} dB", fontsize=9)
        ax.set_aspect("equal")
        if col == 0:
            ax.set_ylabel("Euclidean R²", fontsize=10)

        # ── Sphere  S² ──────────────────────────────────────────────────────
        sigma_s = sph.sigma_for_snr(sig_s, snr_db)
        if isinstance(noise_src, CyclicTableNoise):
            noise_src.reset()
        noisy_s = riemannian_awgn(sph, sig_s, sigma_s, noise_src,
                                   np.random.default_rng(seed + col + 10))
        snr_s = sph.snr_db(sig_s, noisy_s)

        ax = fig.add_subplot(3, 3, 3 + col + 1, projection="3d")
        _draw_sphere_wireframe(ax)
        ax.scatter(*noisy_s.T, s=5, alpha=0.55, color=RED)
        ax.scatter(*sig_s.T,   s=5, alpha=0.90, color=BLUE)
        ax.set_title(f"Sphere S²\n{noise_label}  SNR={snr_s:.1f} dB", fontsize=9)
        ax.tick_params(labelsize=6)

        # ── Hyperbolic  H² (Poincaré disk) ──────────────────────────────────
        sigma_h = hyp.sigma_for_snr(sig_h, snr_db)
        if isinstance(noise_src, CyclicTableNoise):
            noise_src.reset()
        noisy_h = riemannian_awgn(hyp, sig_h, sigma_h, noise_src,
                                   np.random.default_rng(seed + col + 20))
        snr_h = hyp.snr_db(sig_h, noisy_h)

        noisy_d = np.array([hyp.to_poincare(p) for p in noisy_h])
        sig_d   = np.array([hyp.to_poincare(p) for p in sig_h])

        ax = fig.add_subplot(3, 3, 6 + col + 1)
        _draw_poincare_boundary(ax)
        ax.scatter(*noisy_d.T, s=5, alpha=0.45, color=RED)
        ax.scatter(*sig_d.T,   s=5, alpha=0.90, color=BLUE)
        ax.set_title(
            f"Hyperbolic H² (Poincaré)\n{noise_label}  SNR={snr_h:.1f} dB",
            fontsize=9,
        )
        if col == 0:
            ax.set_ylabel("Hyperbolic H²", fontsize=10)

    # shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=7, label="signal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,  markersize=7, label="noisy"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("awgn_table_manifolds.png", dpi=150)
    plt.show()
    print("Saved awgn_table_manifolds.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — SNR accuracy sweep across noise sources and manifolds
# ═══════════════════════════════════════════════════════════════════════════

def plot_snr_sweep(seed: int = 42) -> None:
    """
    Actual Riemannian SNR vs target SNR for all 9 combinations
    (3 manifolds × 3 noise sources).
    """
    euc = EuclideanManifold(n=2)
    sph = Sphere(n=3)
    hyp = HyperbolicSpace(n=2)

    sig_e = _make_euclidean_signal(200)
    sig_s = _make_sphere_signal(200)
    sig_h = _make_hyperbolic_signal(200)

    targets = np.arange(-5, 26, 2.5)

    manifolds   = [("Euclidean R²",    euc, sig_e),
                   ("Sphere S²",       sph, sig_s),
                   ("Hyperbolic H²",   hyp, sig_h)]
    noise_specs = [("Gaussian",      GaussianNoise()),
                   ("Cyclic table",  CyclicTableNoise()),
                   ("Sampled table", SampledTableNoise())]

    linestyles = ["-", "--", ":"]
    colors_m   = ["steelblue", "goldenrod", "mediumorchid"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(targets, targets, "k-", linewidth=1, label="ideal", zorder=0)

    for (m_label, manifold, sig), color in zip(manifolds, colors_m):
        for (n_label, noise_src), ls in zip(noise_specs, linestyles):
            actuals = []
            for t in targets:
                sigma = manifold.sigma_for_snr(sig, t)
                if isinstance(noise_src, CyclicTableNoise):
                    noise_src.reset()
                noisy = riemannian_awgn(manifold, sig, sigma, noise_src,
                                        np.random.default_rng(seed))
                actuals.append(manifold.snr_db(sig, noisy))
            ax.plot(targets, actuals, linestyle=ls, color=color,
                    linewidth=1.6, label=f"{m_label} / {n_label}")

    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel("Actual Riemannian SNR (dB)")
    ax.set_title(
        "SNR accuracy: Gaussian vs Table noise on Riemannian manifolds\n"
        r"Table  T = (1, 3, 4, −1, −4, 5, −2, 0, 2, 0, −3, −5)"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("awgn_table_snr_sweep.png", dpi=150)
    plt.show()
    print("Saved awgn_table_snr_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    np.random.seed(0)

    print("=" * 60)
    print("  AWGN + Non-Euclidean Spaces  ×  Table-Driven Noise")
    print("=" * 60)

    # ── Table statistics ───────────────────────────────────────────────────
    print(f"\nTable  T = {list(map(int, TABLE))}")
    print(f"  mean  = {TABLE_MEAN:.4f}   (zero-mean ✓)")
    print(f"  std   = {TABLE_STD:.4f}")
    print(f"  range = [{int(TABLE.min())}, {int(TABLE.max())}]")
    print(f"  N     = {len(TABLE)}")

    # ── Noise source comparison on a single point ──────────────────────────
    print("\n── Single-point noise demo (p = north pole on S²) ──")
    sph = Sphere(n=3)
    p   = sph.origin()
    rng = np.random.default_rng(7)
    sigma = 0.3

    gauss_src  = GaussianNoise()
    cyclic_src = CyclicTableNoise()
    sample_src = SampledTableNoise()

    for label, src in [("Gaussian",      gauss_src),
                       ("Cyclic table",  cyclic_src),
                       ("Sampled table", sample_src)]:
        pts = np.tile(p, (500, 1))
        noisy = riemannian_awgn(sph, pts, sigma, src, rng)
        dists = sph._batch_dist(pts, noisy)
        print(f"  {label:<15}  mean geodesic dist = {dists.mean():.4f}  "
              f"std = {dists.std():.4f}")

    # ── SNR sweep printout ─────────────────────────────────────────────────
    print("\n── SNR sweep  (Sphere S², all noise sources) ──")
    sig_s = _make_sphere_signal(200)
    print(f"{'Target':>8}  {'Gaussian':>12}  {'Cyclic':>12}  {'Sampled':>12}")
    print("-" * 50)
    for t in [20, 15, 10, 5, 0]:
        row = []
        for src in [GaussianNoise(), CyclicTableNoise(), SampledTableNoise()]:
            sigma = sph.sigma_for_snr(sig_s, t)
            noisy = riemannian_awgn(sph, sig_s, sigma, src,
                                    np.random.default_rng(42))
            row.append(sph.snr_db(sig_s, noisy))
        print(f"{t:>8}  {row[0]:>12.2f}  {row[1]:>12.2f}  {row[2]:>12.2f}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nPlot 1/3: table analysis …")
    plot_table_analysis()

    print("Plot 2/3: manifold × noise-source comparison …")
    plot_manifold_comparison(snr_db=10.0)

    print("Plot 3/3: SNR sweep …")
    plot_snr_sweep()


if __name__ == "__main__":
    main()
