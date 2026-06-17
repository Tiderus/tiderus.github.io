"""
Additive White Gaussian Noise (AWGN) in Non-Euclidean Riemannian Spaces

Standard AWGN (Euclidean):
    x_noisy = x + n,   n ~ N(0, σ²I)

Riemannian AWGN on manifold M:
    1. Generate Gaussian noise v in the tangent space T_p(M)  (locally flat).
    2. Map back to the manifold:  x_noisy = Exp_p(σ · v)

The key operators for each manifold:
    exp_map(p, v)  — shoot a geodesic from p with initial velocity v
    log_map(p, q)  — tangent vector at p that points toward q
    distance(p, q) — geodesic (intrinsic) distance

Implemented manifolds
─────────────────────
    EuclideanManifold  : R^n          (flat,  κ =  0)
    Sphere             : S^{n-1}      (round, κ = +1)
    HyperbolicSpace    : H^n          (saddle,κ = -1, hyperboloid model)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════
# Abstract manifold
# ═══════════════════════════════════════════════════════════════════════════

class Manifold(ABC):
    """Abstract Riemannian manifold with built-in Riemannian AWGN."""

    # ── required interface ──────────────────────────────────────────────────

    @abstractmethod
    def exp_map(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map Exp_p(v): geodesic from p with tangent v."""

    @abstractmethod
    def log_map(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Log map Log_p(q): tangent vector at p pointing toward q."""

    @abstractmethod
    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Geodesic distance d(p, q)."""

    @abstractmethod
    def project_tangent(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project ambient vector v to the tangent space T_p(M)."""

    @abstractmethod
    def origin(self) -> np.ndarray:
        """Canonical reference point (north pole, hyperboloid tip, …)."""

    # ── Riemannian AWGN ─────────────────────────────────────────────────────

    def awgn(
        self,
        points: np.ndarray,          # (N, ambient_dim)
        sigma: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Add isotropic Gaussian noise to every point.

        For each point p:
          v_raw ~ N(0, I)  in the ambient space
          v     = project_tangent(p, v_raw)   →  lives in T_p(M)
          x_noisy = Exp_p(σ · v)
        """
        noisy = np.empty_like(points)
        for i, p in enumerate(points):
            v_raw = rng.standard_normal(p.shape)
            v = self.project_tangent(p, v_raw)
            noisy[i] = self.exp_map(p, sigma * v)
        return noisy

    # ── SNR utilities ────────────────────────────────────────────────────────

    def _batch_dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.array([self.distance(a, b) for a, b in zip(A, B)])

    def snr_db(self, signal: np.ndarray, noisy: np.ndarray) -> float:
        """
        Riemannian SNR in dB:
            SNR = 10 log10( E[d(o, p)²] / E[d(p, p̃)²] )

        where o = origin(), p = clean point, p̃ = noisy point.
        """
        o = self.origin()
        origins = np.tile(o, (len(signal), 1))
        sig_power = float(np.mean(self._batch_dist(origins, signal) ** 2))
        noise_power = float(np.mean(self._batch_dist(signal, noisy) ** 2))
        if noise_power < 1e-30:
            return np.inf
        return 10.0 * np.log10(sig_power / noise_power)

    def sigma_for_snr(self, signal: np.ndarray, snr_db: float) -> float:
        """
        Noise sigma that approximately achieves the target SNR.

        Uses the approximation: noise_power ≈ σ² · tangent_dim
        so σ ≈ sqrt( signal_power / (tangent_dim · 10^(SNR/10)) ).
        """
        o = self.origin()
        origins = np.tile(o, (len(signal), 1))
        sig_power = float(np.mean(self._batch_dist(origins, signal) ** 2))
        noise_power_target = sig_power / (10.0 ** (snr_db / 10.0))
        return float(np.sqrt(noise_power_target))


# ═══════════════════════════════════════════════════════════════════════════
# Euclidean space  R^n   (flat baseline)
# ═══════════════════════════════════════════════════════════════════════════

class EuclideanManifold(Manifold):
    """Flat Euclidean space R^n.  Riemannian AWGN reduces to standard AWGN."""

    def __init__(self, n: int):
        self.n = n

    def exp_map(self, p, v):          return p + v
    def log_map(self, p, q):          return q - p
    def distance(self, p, q):         return float(np.linalg.norm(q - p))
    def project_tangent(self, p, v):  return v          # T_p(R^n) = R^n
    def origin(self):                 return np.zeros(self.n)


# ═══════════════════════════════════════════════════════════════════════════
# Sphere  S^{n-1} ⊂ R^n   (positive curvature κ = +1)
# ═══════════════════════════════════════════════════════════════════════════

class Sphere(Manifold):
    """
    Unit sphere S^{n-1} embedded in R^n.

    Points   : unit vectors  ‖p‖ = 1
    Tangent  : T_p(S^{n-1}) = { v : p·v = 0 }

    Formulae:
        exp_map(p, v)  = cos(‖v‖)·p + sin(‖v‖)·v̂
        log_map(p, q)  = θ·(q − cos θ · p)/sin θ,    θ = arccos(p·q)
        distance(p, q) = arccos(p·q)
    """

    def __init__(self, n: int):
        self.n = n          # ambient dimension;  sphere dimension = n − 1

    def project_tangent(self, p, v):
        return v - np.dot(v, p) * p

    def exp_map(self, p, v):
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            return p.copy()
        return np.cos(nv) * p + np.sin(nv) * (v / nv)

    def log_map(self, p, q):
        cos_t = float(np.clip(np.dot(p, q), -1.0, 1.0))
        theta = np.arccos(cos_t)
        if theta < 1e-10:
            return np.zeros_like(p)
        return theta * (q - cos_t * p) / np.sin(theta)

    def distance(self, p, q):
        return float(np.arccos(np.clip(float(np.dot(p, q)), -1.0, 1.0)))

    def origin(self):
        o = np.zeros(self.n)
        o[-1] = 1.0          # north pole
        return o

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        v = rng.standard_normal(self.n)
        return v / np.linalg.norm(v)


# ═══════════════════════════════════════════════════════════════════════════
# Hyperbolic space  H^n   (negative curvature κ = −1, hyperboloid model)
# ═══════════════════════════════════════════════════════════════════════════

class HyperbolicSpace(Manifold):
    """
    Hyperbolic space H^n — hyperboloid model in Minkowski space R^{n,1}.

    Minkowski inner product:
        ⟨x, y⟩_M = −x₀y₀ + x₁y₁ + ⋯ + xₙyₙ

    Manifold:
        { x ∈ R^{n+1} : ⟨x, x⟩_M = −1,  x₀ > 0 }

    Points  : (cosh r, sinh r · û)  for unit û ∈ R^n, r ≥ 0
    Tangent : T_p(H^n) = { v : ⟨p, v⟩_M = 0 }

    Formulae:
        exp_map(p, v)  = cosh(‖v‖_M)·p + sinh(‖v‖_M)·v/‖v‖_M
        log_map(p, q)  = d·(q − cosh(d)·p)/sinh(d),  d = arccosh(−⟨p,q⟩_M)
        distance(p, q) = arccosh(−⟨p,q⟩_M)
    """

    def __init__(self, n: int):
        self.n = n          # intrinsic dimension;  ambient = n + 1

    # ── Minkowski helpers ───────────────────────────────────────────────────

    @staticmethod
    def _mink(x: np.ndarray, y: np.ndarray) -> float:
        return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))

    @staticmethod
    def _mink_norm(v: np.ndarray) -> float:
        """Norm of a spacelike tangent vector in the Minkowski metric."""
        return float(np.sqrt(max(0.0, -v[0] ** 2 + np.dot(v[1:], v[1:]))))

    # ── manifold interface ──────────────────────────────────────────────────

    def project_tangent(self, p, v):
        # <p,p>_M = -1, so T-projection is: v - (<p,v>_M / <p,p>_M)·p = v + <p,v>_M·p
        return v - self._mink(p, v) * p

    def exp_map(self, p, v):
        nv = self._mink_norm(v)
        if nv < 1e-12:
            return p.copy()
        return np.cosh(nv) * p + np.sinh(nv) * (v / nv)

    def log_map(self, p, q):
        mpq = self._mink(p, q)          # = −cosh d
        d = float(np.arccosh(np.clip(-mpq, 1.0, None)))
        if d < 1e-10:
            return np.zeros_like(p)
        # mpq = <p,q>_M = -cosh(d), so q - cosh(d)·p = q + mpq·p
        return d * (q + mpq * p) / np.sinh(d)

    def distance(self, p, q):
        return float(np.arccosh(np.clip(-self._mink(p, q), 1.0, None)))

    def origin(self):
        o = np.zeros(self.n + 1)
        o[0] = 1.0           # tip of the hyperboloid
        return o

    def random_point(self, rng: np.random.Generator, scale: float = 0.8) -> np.ndarray:
        """Point at a random distance (Exp-distributed with mean=scale) from origin."""
        direction = rng.standard_normal(self.n)
        direction /= np.linalg.norm(direction)
        r = rng.exponential(scale)
        p = np.zeros(self.n + 1)
        p[0] = np.cosh(r)
        p[1:] = np.sinh(r) * direction
        return p

    def to_poincare(self, p: np.ndarray) -> np.ndarray:
        """Project hyperboloid → Poincaré disk: (x₁,…,xₙ)/(1 + x₀)."""
        return p[1:] / (1.0 + p[0])


# ═══════════════════════════════════════════════════════════════════════════
# Demo builders
# ═══════════════════════════════════════════════════════════════════════════

def _make_euclidean_signal(n_points: int) -> np.ndarray:
    """Ring of points in R²."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([2.0 * np.cos(theta), 2.0 * np.sin(theta)], axis=1)


def _make_sphere_signal(n_points: int) -> np.ndarray:
    """Equatorial great circle on S²."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta), np.zeros(n_points)], axis=1)


def _make_hyperbolic_signal(n_points: int) -> np.ndarray:
    """Geodesic arc from origin in H²."""
    t = np.linspace(0.05, 1.5, n_points)
    return np.stack([np.cosh(t), np.sinh(t), np.zeros(n_points)], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(snr_db: float = 10.0, n_points: int = 300, seed: int = 42) -> None:
    """
    Three-panel comparison: Euclidean · Sphere S² · Hyperbolic H²
    plus a fourth panel with an SNR-sweep across all three manifolds.
    """
    rng = np.random.default_rng(seed)

    euc  = EuclideanManifold(n=2)
    sph  = Sphere(n=3)
    hyp  = HyperbolicSpace(n=2)

    sig_e = _make_euclidean_signal(n_points)
    sig_s = _make_sphere_signal(n_points)
    sig_h = _make_hyperbolic_signal(n_points)

    # ── add noise ──────────────────────────────────────────────────────────
    rng_e, rng_s, rng_h = (np.random.default_rng(seed + i) for i in range(3))

    sigma_e = euc.sigma_for_snr(sig_e, snr_db)
    sigma_s = sph.sigma_for_snr(sig_s, snr_db)
    sigma_h = hyp.sigma_for_snr(sig_h, snr_db)

    noisy_e = euc.awgn(sig_e, sigma_e, rng_e)
    noisy_s = sph.awgn(sig_s, sigma_s, rng_s)
    noisy_h = hyp.awgn(sig_h, sigma_h, rng_h)

    snr_e = euc.snr_db(sig_e, noisy_e)
    snr_s = sph.snr_db(sig_s, noisy_s)
    snr_h = hyp.snr_db(sig_h, noisy_h)

    print(f"Target SNR : {snr_db:.1f} dB")
    print(f"  Euclidean  : {snr_e:.2f} dB  (σ = {sigma_e:.4f})")
    print(f"  Sphere S²  : {snr_s:.2f} dB  (σ = {sigma_s:.4f})")
    print(f"  Hyperbolic : {snr_h:.2f} dB  (σ = {sigma_h:.4f})")

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f"AWGN in Non-Euclidean Spaces  (target SNR = {snr_db} dB)",
        fontsize=13,
    )

    BLUE, RED = "steelblue", "tomato"

    # ── panel 1 : Euclidean ────────────────────────────────────────────────
    ax1 = fig.add_subplot(141)
    ax1.scatter(*noisy_e.T, s=6, alpha=0.45, color=RED,  label="noisy")
    ax1.scatter(*sig_e.T,   s=6, alpha=0.90, color=BLUE, label="signal")
    ax1.set_title(f"Euclidean R²\n(κ = 0)  SNR = {snr_e:.1f} dB")
    ax1.set_aspect("equal")
    ax1.legend(markerscale=2, fontsize=8)

    # ── panel 2 : Sphere ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(142, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:25j]
    ax2.plot_surface(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        alpha=0.08, color="gray", linewidth=0,
    )
    ax2.scatter(*noisy_s.T, s=6, alpha=0.55, color=RED)
    ax2.scatter(*sig_s.T,   s=6, alpha=0.90, color=BLUE)
    ax2.set_title(f"Sphere S²\n(κ = +1)  SNR = {snr_s:.1f} dB")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.tick_params(labelsize=7)

    # ── panel 3 : Hyperbolic (Poincaré disk) ──────────────────────────────
    ax3 = fig.add_subplot(143)
    disk = plt.Circle((0, 0), 1.0, fill=False, color="gray",
                       linewidth=1.5, linestyle="--")
    ax3.add_patch(disk)
    noisy_d = np.array([hyp.to_poincare(p) for p in noisy_h])
    sig_d   = np.array([hyp.to_poincare(p) for p in sig_h])
    ax3.scatter(*noisy_d.T, s=6, alpha=0.45, color=RED,  label="noisy")
    ax3.scatter(*sig_d.T,   s=6, alpha=0.90, color=BLUE, label="signal")
    ax3.set_title(f"Hyperbolic H²\n(κ = −1)  SNR = {snr_h:.1f} dB\n(Poincaré disk)")
    ax3.set_xlim(-1.15, 1.15); ax3.set_ylim(-1.15, 1.15)
    ax3.set_aspect("equal")
    ax3.legend(markerscale=2, fontsize=8)

    # ── panel 4 : SNR sweep comparison ────────────────────────────────────
    ax4 = fig.add_subplot(144)
    snr_targets = np.arange(-5, 26, 2.5)
    results: dict[str, list[float]] = {"Euclidean": [], "Sphere S²": [], "Hyperbolic H²": []}

    for target in snr_targets:
        for label, manifold, sig in [
            ("Euclidean",    euc, sig_e),
            ("Sphere S²",    sph, sig_s),
            ("Hyperbolic H²",hyp, sig_h),
        ]:
            sigma = manifold.sigma_for_snr(sig, target)
            noisy_tmp = manifold.awgn(sig, sigma, np.random.default_rng(seed))
            results[label].append(manifold.snr_db(sig, noisy_tmp))

    colors_sweep = {"Euclidean": BLUE, "Sphere S²": "goldenrod", "Hyperbolic H²": "mediumorchid"}
    for label, actuals in results.items():
        ax4.plot(snr_targets, actuals, "o-", ms=4, label=label,
                 color=colors_sweep[label])
    ax4.plot(snr_targets, snr_targets, "k--", linewidth=1, label="ideal")
    ax4.set_xlabel("Target SNR (dB)")
    ax4.set_ylabel("Actual Riemannian SNR (dB)")
    ax4.set_title("SNR Accuracy Across Manifolds")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("awgn_non_euclidean.png", dpi=150)
    plt.show()
    print("Plot saved to awgn_non_euclidean.png")


def plot_curvature_effect(seed: int = 42) -> None:
    """
    Show how curvature distorts noise at different sigma levels.

    A single point is corrupted many times; we plot the cloud of noisy
    samples on each manifold to make the curvature effect visible.
    """
    rng = np.random.default_rng(seed)
    n_samples = 500
    sigmas = [0.1, 0.3, 0.6, 1.0]

    sph = Sphere(n=3)
    hyp = HyperbolicSpace(n=2)

    # anchor points: slightly off-origin so distances are well-defined
    anchor_s = np.array([0.0, np.sin(0.5), np.cos(0.5)])  # 0.5 rad from north pole
    anchor_h = np.array([np.cosh(0.5), np.sinh(0.5), 0.0])

    fig, axes = plt.subplots(2, len(sigmas), figsize=(14, 6))
    fig.suptitle("Noise cloud on Sphere (top) and Hyperbolic plane (bottom)\n"
                 "as σ increases  —  note how curvature wraps / fans out the cloud",
                 fontsize=11)

    for col, sigma in enumerate(sigmas):
        # ── Sphere ─────────────────────────────────────────────────────────
        anchors_s = np.tile(anchor_s, (n_samples, 1))
        cloud_s   = sph.awgn(anchors_s, sigma, np.random.default_rng(seed + col))

        ax_s = axes[0, col]
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        ax_s = fig.add_subplot(2, len(sigmas), col + 1, projection="3d")
        ax_s.plot_surface(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                          alpha=0.07, color="gray", linewidth=0)
        ax_s.scatter(*cloud_s.T, s=4, alpha=0.5, color="goldenrod")
        ax_s.scatter(*anchor_s, s=60, color="steelblue", zorder=5)
        ax_s.set_title(f"S²  σ = {sigma}", fontsize=9)
        ax_s.tick_params(labelsize=6)
        ax_s.set_xlabel("x", fontsize=7); ax_s.set_ylabel("y", fontsize=7)

        # ── Hyperbolic (Poincaré) ───────────────────────────────────────────
        anchors_h = np.tile(anchor_h, (n_samples, 1))
        cloud_h   = hyp.awgn(anchors_h, sigma, np.random.default_rng(seed + col + 100))
        cloud_d   = np.array([hyp.to_poincare(p) for p in cloud_h])
        anchor_d  = hyp.to_poincare(anchor_h)

        ax_h = fig.add_subplot(2, len(sigmas), len(sigmas) + col + 1)
        circle = plt.Circle((0, 0), 1.0, fill=False, color="gray",
                             linewidth=1.2, linestyle="--")
        ax_h.add_patch(circle)
        ax_h.scatter(*cloud_d.T, s=4, alpha=0.5, color="mediumorchid")
        ax_h.scatter(*anchor_d, s=60, color="steelblue", zorder=5)
        ax_h.set_xlim(-1.1, 1.1); ax_h.set_ylim(-1.1, 1.1)
        ax_h.set_aspect("equal")
        ax_h.set_title(f"H²  σ = {sigma}", fontsize=9)

    plt.tight_layout()
    plt.savefig("awgn_curvature_effect.png", dpi=150)
    plt.show()
    print("Plot saved to awgn_curvature_effect.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    np.random.seed(0)

    print("=" * 55)
    print("  AWGN in Non-Euclidean Riemannian Spaces")
    print("=" * 55)

    # ── quick sanity check ─────────────────────────────────────────────────
    print("\n── Manifold geometry sanity check ──")
    sph = Sphere(n=3)
    hyp = HyperbolicSpace(n=2)

    p = np.array([1.0, 0.0, 0.0])   # equatorial point on S²
    q = np.array([0.0, 1.0, 0.0])
    print(f"Sphere  d(p,q)            = {sph.distance(p,q):.6f}  (expect π/2 ≈ {np.pi/2:.6f})")
    v = sph.log_map(p, q)
    print(f"Sphere  ‖Exp_p(Log_p(q)) − q‖ = {np.linalg.norm(sph.exp_map(p,v) - q):.2e}")

    o = hyp.origin()
    r = np.array([np.cosh(1.0), np.sinh(1.0), 0.0])
    print(f"Hyperbolic d(o, r=1.0)    = {hyp.distance(o, r):.6f}  (expect 1.000000)")
    v = hyp.log_map(o, r)
    print(f"Hyperbolic ‖Exp_o(Log_o(r)) − r‖ = {np.linalg.norm(hyp.exp_map(o,v) - r):.2e}")

    # ── SNR sweep printout ─────────────────────────────────────────────────
    print("\n── SNR sweep (target vs actual) ──")
    rng = np.random.default_rng(7)
    euc = EuclideanManifold(n=2)
    sig_e = _make_euclidean_signal(200)
    sig_s = _make_sphere_signal(200)
    sig_h = _make_hyperbolic_signal(200)

    print(f"{'Target':>8}  {'Euclidean':>12}  {'Sphere S²':>12}  {'Hyperbolic H²':>14}")
    print("-" * 55)
    for target in [20, 15, 10, 5, 0]:
        rows = []
        for manifold, sig in [(euc, sig_e), (sph, sig_s), (hyp, sig_h)]:
            s = manifold.sigma_for_snr(sig, target)
            noisy = manifold.awgn(sig, s, np.random.default_rng(42))
            rows.append(manifold.snr_db(sig, noisy))
        print(f"{target:>8}  {rows[0]:>12.2f}  {rows[1]:>12.2f}  {rows[2]:>14.2f}")

    # ── plots ──────────────────────────────────────────────────────────────
    print("\nGenerating comparison plot …")
    plot_all(snr_db=10.0)

    print("Generating curvature-effect plot …")
    plot_curvature_effect()


if __name__ == "__main__":
    main()
