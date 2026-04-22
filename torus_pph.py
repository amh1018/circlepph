"""
Persistent Path Homology on T²
==============================

Clean, consolidated implementation for analyzing PPH on torus samples.

Usage:
    python torus_pph.py                    # Run full analysis
    python torus_pph.py --quick            # Quick test
    python torus_pph.py --n 6             # Single analysis (6×6 grid)
    python torus_pph.py --n 6 --height sin_plus_sin
"""

import os
import hashlib
import pickle
import dataclasses
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

# Limit the Rust/LoPHAT Rayon thread-pool to one thread.  Must be set
# before grpphati is imported.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

# PPH computation
from grpphati.filtrations import Filtration, ProperGroundedFiltration
from grpphati.homologies import RegularPathHomology
from grpphati.pipelines.standard import make_standard_pipeline
from grpphati.backends import LoPHATBackend
from grpphati.optimisations import component_appendage_empty


# =============================================================================
# RESULT CACHE
# =============================================================================

_CACHE_DIR = Path(".pph_cache") / "torus"


def _cfg_key(cfg) -> str:
    """Stable MD5 key derived from all fields of a Config dataclass."""
    d = dataclasses.asdict(cfg)
    return hashlib.md5(repr(sorted(d.items())).encode()).hexdigest()


def _load_cached(key: str):
    """Return the cached Result for *key*, or None if absent / unreadable."""
    path = _CACHE_DIR / f"{key}.pkl"
    if path.exists():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _save_cached(key: str, result) -> None:
    """Persist *result* to disk under *key*."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_DIR / f"{key}.pkl", "wb") as fh:
        pickle.dump(result, fh)


# =============================================================================
# CONFIGURATION
# =============================================================================

HeightVariant = Literal[
    # --- single-variable (suffixed) ---
    'sin_theta', 'sin_phi', 'cos_theta', 'cos_phi',
    'sin2_theta', 'sin2_phi', 'cos2_theta', 'cos2_phi',
    'abs_sin_theta', 'abs_sin_phi',
    'sawtooth_theta', 'sawtooth_phi',
    'double_max_theta', 'double_max_phi',
    # --- ported from circle_pph_2 (all applied to θ) ---
    'double_max_mod',     # double_max with flattened peak at π/2
    'squeeze',            # compressed sine on [0,π), original on [π,2π)
    'plateau',            # sine with flat zero plateau near π
    'asymmetric',         # asymmetrically time-stretched sine
    'valley',             # sine with a dip carved into the upper half
    'shallow',            # amplitude-modulated shallow sine
    'quarter_triangle',   # piecewise-affine tent: 0→1 on [0,π/2], 1→0 on [π/2,2π]
    'damped',             # exponentially damped sine: exp(-0.5θ)·sin(4θ)
    'sin3', 'cos3', 'sin4', 'sin5', 'cos5',
    'triangle',           # symmetric triangle wave
    'square',             # square wave (±1)
    'saw2', 'tri2', 'tri3',
    'sin_sq',             # sin²(θ)
    'abs_cos',            # |cos(θ)|
    'sin_plus_sin2',      # sin + ½ sin(2θ)
    'sin_plus_sin3',      # sin + 0.3 sin(3θ)
    'sin_plus_cos2',      # sin + cos(2θ)
    'sin_cos',            # sin + cos
    'asymmetric2',        # sin + ½ cos(2θ)
    'bump',               # double Gaussian bump
    'sin_cubed',          # sin³(θ)
    'cos_cubed',          # cos³(θ)
    'double_min',         # two minima via cos modulation
    'triple_max',         # sin²·cos
    'skewed',             # Fourier series skew
    'sin_abs_cos',        # sin·|cos|
    'cos_abs_sin',        # cos·|sin|
    'sigmoid_periodic',   # tanh(3 sin θ)
    'clipped_sin',        # sin clipped to [−0.5, 0.5]
    'rectified_sin',      # max(0, sin θ)
    'sin_shifted',        # sin(θ + π/3)
    'cos_shifted_third',  # cos(θ + 2π/3)
    'devil',              # sin·cos(3θ) + ½ sin(5θ)·cos
    'heartbeat',          # exp-damped high-freq sine
    'fractal3',           # partial Fourier square-wave
    'wobble',             # self-modulated sine
    'ekg',                # EKG-like pulse train
    'teeth',              # sin·sin(2θ)·sin(3θ)
    'chaos',              # iterated-phase sine
    'nested_sin',         # sin(θ + sin(θ + sin θ))
    'mod_wave',           # amplitude-modulated by cos(3θ)
    'phase_mod',          # sin(θ + 2 sin θ)
    'ripple',             # sin with high-freq ripple
    'dragon',             # sin³ + cos²(3θ)·sin(2θ)
    'staircase',          # alternating-sign Fourier series
    'interference',       # two-frequency interference
    'tangled',            # doubly self-modulated trig
    'bouncy',             # high-freq damped oscillation
    'lopsided',           # sin⁵ + cos²(2θ) − ½ sin(3θ)
    'saw_smooth',         # Fourier sawtooth (9 terms)
    'alien',              # nested-trig alien wave
    'epileptic',          # product of four sines
    'saw_down', 'saw_up', 'saw_steep', 'saw_gentle',
    'saw_offset', 'double_saw', 'triple_saw',
    'saw_flipped', 'saw_abs', 'zigzag',
    'saw_sin', 'saw_squared', 'ramp_drop',
    'step2', 'step3', 'step4',
    'half_half', 'random_step', 'cantor_like',
    # --- two-variable sums / products ---
    'sin_plus_sin', 'sin_minus_sin', 'sin_times_sin',
    'sin_theta_cos_phi', 'cos_theta_sin_phi',
    'diagonal', 'antidiagonal',
    'max_sin', 'min_sin',
    'saddle',
    # --- 3D embedding ---
    'z_coord', 'x_coord', 'y_coord',
    # --- misc ---
    'radial', 'constant',
    'sawtooth_sum', 'sawtooth_product', 'sawtooth_diag'
]

SamplingVariant = Literal[
    'grid',        # uniform n×m grid
    'jittered',    # jittered grid
    'random',      # random uniform on T²
    'clustered',   # two clusters at (0,0) and (π,π)
    'chebyshev',   # Chebyshev nodes in each direction
]


@dataclass
class Config:
    """Analysis configuration."""
    n: int                          # grid side length in θ direction
    m: int | None = None            # grid side length in φ direction; defaults to n
    height_variant: HeightVariant = 'sin_theta'
    double_edges: bool = True       # add both u→v and v→u for height-tied pairs
    R: float = 2.0                  # torus major radius (for 3D-embedding variants)
    r: float = 1.0                  # torus minor radius
    sampling: SamplingVariant = 'grid'
    jitter_std: float = 0.3
    random_seed: int | None = None


# =============================================================================
# SAMPLING
# =============================================================================

def sample_torus(cfg: Config) -> list[tuple[float, float]]:
    """Generate (theta, phi) pairs on T² = [0,2π)²."""
    n = cfg.n
    m = cfg.m if cfg.m is not None else n
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.sampling == 'grid':
        return [
            (i * 2 * np.pi / n, j * 2 * np.pi / m)
            for i in range(n) for j in range(m)
        ]

    elif cfg.sampling == 'jittered':
        dtheta = 2 * np.pi / n
        dphi = 2 * np.pi / m
        pts = []
        for i in range(n):
            for j in range(m):
                theta = (i * dtheta + rng.normal(0, cfg.jitter_std * dtheta)) % (2 * np.pi)
                phi = (j * dphi + rng.normal(0, cfg.jitter_std * dphi)) % (2 * np.pi)
                pts.append((float(theta), float(phi)))
        return pts

    elif cfg.sampling == 'random':
        total = n * m
        thetas = rng.uniform(0, 2 * np.pi, size=total)
        phis = rng.uniform(0, 2 * np.pi, size=total)
        return list(zip(thetas.tolist(), phis.tolist()))

    elif cfg.sampling == 'clustered':
        total = n * m
        half = total // 2
        rest = total - half
        std = cfg.jitter_std
        c1_t = rng.normal(0, std, size=half) % (2 * np.pi)
        c1_p = rng.normal(0, std, size=half) % (2 * np.pi)
        c2_t = rng.normal(np.pi, std, size=rest) % (2 * np.pi)
        c2_p = rng.normal(np.pi, std, size=rest) % (2 * np.pi)
        thetas = np.concatenate([c1_t, c2_t])
        phis = np.concatenate([c1_p, c2_p])
        return list(zip(thetas.tolist(), phis.tolist()))

    elif cfg.sampling == 'chebyshev':
        k_n = np.arange(1, n + 1)
        k_m = np.arange(1, m + 1)
        theta_nodes = sorted((np.cos((2 * k_n - 1) * np.pi / (2 * n)) + 1) * np.pi)
        phi_nodes = sorted((np.cos((2 * k_m - 1) * np.pi / (2 * m)) + 1) * np.pi)
        return [(float(t), float(p)) for t in theta_nodes for p in phi_nodes]

    else:
        raise ValueError(f"Unknown sampling variant: {cfg.sampling}")


# =============================================================================
# HEIGHT FUNCTIONS
# =============================================================================

def height(theta: float, phi: float, variant: HeightVariant = 'sin_theta',
           R: float = 2.0, r: float = 1.0) -> float:
    """Height function on T²."""
    theta = theta % (2 * np.pi)
    phi = phi % (2 * np.pi)

    if variant == 'sin_theta':
        return np.sin(theta)
    if variant == 'sin_phi':
        return np.sin(phi)
    if variant == 'cos_theta':
        return np.cos(theta)
    if variant == 'cos_phi':
        return np.cos(phi)

    if variant == 'sin2_theta':
        return np.sin(2 * theta)
    if variant == 'sin2_phi':
        return np.sin(2 * phi)
    if variant == 'cos2_theta':
        return np.cos(2 * theta)
    if variant == 'cos2_phi':
        return np.cos(2 * phi)

    if variant == 'abs_sin_theta':
        return abs(np.sin(theta))
    if variant == 'abs_sin_phi':
        return abs(np.sin(phi))

    if variant == 'sawtooth_theta':
        return 1 - theta / np.pi
    if variant == 'sawtooth_phi':
        return 1 - phi / np.pi

    if variant == 'double_max_theta':
        if theta < np.pi / 2:
            return 1 - np.sin(2 * theta)
        return np.sin(theta)
    if variant == 'double_max_phi':
        if phi < np.pi / 2:
            return 1 - np.sin(2 * phi)
        return np.sin(phi)

    if variant == 'double_max_mod':
        if theta == np.pi / 2:
            return 1 - 0.05
        elif theta < np.pi / 2:
            return 1 - np.sin(2 * theta)
        else:
            return np.sin(theta)

    if variant == 'squeeze':
        if theta < np.pi:
            return np.sin(2 * theta - np.pi / 2)
        else:
            return np.sin(theta)

    if variant == 'plateau':
        if 0.75 * np.pi < theta < 1.25 * np.pi:
            return 0.0
        else:
            return np.sin(theta)

    if variant == 'asymmetric':
        if theta <= np.pi / 2:
            return np.sin(theta)
        else:
            mid = 5 * np.pi / 4
            if theta <= mid:
                t = np.pi/2 + (theta - np.pi/2) * np.pi / (mid - np.pi/2)
            else:
                t = 3*np.pi/2 + (theta - mid) * (np.pi/2) / (2*np.pi - mid)
            return np.sin(t)

    if variant == 'valley':
        base = np.sin(theta)
        if np.pi / 2 <= theta <= 3 * np.pi / 2:
            return base - 0.3 * np.sin(2 * (theta - np.pi / 2))
        return base

    if variant == 'shallow':
        return np.sin(theta) * (0.75 + 0.25 * np.cos(theta))

    if variant == 'quarter_triangle':
        if theta <= np.pi / 2:
            return theta / (np.pi / 2)
        else:
            return 1.0 - (theta - np.pi / 2) / (3 * np.pi / 2)

    if variant == 'damped':
        return float(np.exp(-0.5 * theta) * np.sin(4 * theta))

    if variant == 'sin3':
        return np.sin(3 * theta)
    if variant == 'cos3':
        return np.cos(3 * theta)
    if variant == 'sin4':
        return np.sin(4 * theta)
    if variant == 'sin5':
        return np.sin(5 * theta)
    if variant == 'cos5':
        return np.cos(5 * theta)

    if variant == 'triangle':
        if theta < np.pi:
            return 1 - 2 * theta / np.pi
        else:
            return -3 + 2 * theta / np.pi

    if variant == 'square':
        return 1.0 if theta < np.pi else -1.0

    if variant == 'saw2':
        t = theta % np.pi
        return 1 - 2 * t / np.pi

    if variant == 'tri2':
        t = theta % np.pi
        return 1 - 4 * abs(t / np.pi - 0.5)

    if variant == 'tri3':
        t = theta % (2 * np.pi / 3)
        return 1 - 6 * abs(t / (2 * np.pi / 3) - 0.5)

    if variant == 'sin_sq':
        return np.sin(theta) ** 2

    if variant == 'abs_cos':
        return abs(np.cos(theta))

    if variant == 'sin_plus_sin2':
        return np.sin(theta) + 0.5 * np.sin(2 * theta)

    if variant == 'sin_plus_sin3':
        return np.sin(theta) + 0.3 * np.sin(3 * theta)

    if variant == 'sin_plus_cos2':
        return np.sin(theta) + np.cos(2 * theta)

    if variant == 'sin_cos':
        return np.sin(theta) + np.cos(theta)

    if variant == 'asymmetric2':
        return np.sin(theta) + 0.5 * np.cos(2 * theta)

    if variant == 'bump':
        return (np.exp(-8 * (theta - np.pi/2)**2)
                - np.exp(-8 * (theta - 3*np.pi/2)**2))

    if variant == 'sin_cubed':
        return np.sin(theta) ** 3

    if variant == 'cos_cubed':
        return np.cos(theta) ** 3

    if variant == 'double_min':
        if theta >= np.pi:
            return -1 + np.cos(2 * theta)
        else:
            return np.sin(theta)

    if variant == 'triple_max':
        return np.sin(theta) ** 2 * np.cos(theta)

    if variant == 'skewed':
        return (np.sin(theta) + 0.5 * np.sin(2*theta)
                + 0.25 * np.sin(3*theta))

    if variant == 'sin_abs_cos':
        return np.sin(theta) * abs(np.cos(theta))

    if variant == 'cos_abs_sin':
        return np.cos(theta) * abs(np.sin(theta))

    if variant == 'sigmoid_periodic':
        return np.tanh(3 * np.sin(theta))

    if variant == 'clipped_sin':
        return float(np.clip(np.sin(theta), -0.5, 0.5))

    if variant == 'rectified_sin':
        return max(0.0, float(np.sin(theta)))

    if variant == 'sin_shifted':
        return np.sin(theta + np.pi / 3)

    if variant == 'cos_shifted_third':
        return np.cos(theta + 2 * np.pi / 3)

    if variant == 'devil':
        return (np.sin(theta) * np.cos(3*theta)
                + 0.5 * np.sin(5*theta) * np.cos(theta))

    if variant == 'heartbeat':
        return np.exp(-10 * np.sin(theta/2)**2) * np.sin(3*theta)

    if variant == 'fractal3':
        return (np.sin(theta) + np.sin(3*theta)/3 + np.sin(5*theta)/5
                + np.sin(7*theta)/7 + np.sin(9*theta)/9)

    if variant == 'wobble':
        return np.sin(theta + np.sin(theta)) * np.cos(np.cos(2*theta))

    if variant == 'ekg':
        return (np.sin(5*theta) * np.exp(-3 * np.sin(theta/2)**2)
                + 0.3 * np.sin(2*theta))

    if variant == 'teeth':
        return np.sin(theta) * np.sin(2*theta) * np.sin(3*theta)

    if variant == 'chaos':
        return np.sin(theta + np.sin(2*theta + np.sin(3*theta)))

    if variant == 'nested_sin':
        return np.sin(theta + np.sin(theta + np.sin(theta)))

    if variant == 'mod_wave':
        return (1 + 0.5*np.cos(3*theta)) * np.sin(theta)

    if variant == 'phase_mod':
        return np.sin(theta + 2*np.sin(theta))

    if variant == 'ripple':
        return (np.sin(theta) + 0.2*np.sin(7*theta)
                + 0.1*np.sin(13*theta))

    if variant == 'dragon':
        return (np.sin(theta)**3
                + np.cos(3*theta)**2 * np.sin(2*theta))

    if variant == 'staircase':
        return (np.sin(theta) + np.sin(3*theta)/3 - np.sin(5*theta)/5
                + np.sin(7*theta)/7 - np.sin(9*theta)/9)

    if variant == 'interference':
        return (np.sin(theta) * np.cos(0.5*theta)
                + np.cos(theta) * np.sin(1.5*theta))

    if variant == 'tangled':
        return (np.sin(2*theta + np.cos(3*theta))
                * np.cos(theta - np.sin(2*theta)))

    if variant == 'bouncy':
        return np.sin(4*theta) * np.exp(-abs(np.sin(theta/2)))

    if variant == 'lopsided':
        return (np.sin(theta)**5 + np.cos(2*theta)**3
                - 0.5*np.sin(3*theta))

    if variant == 'saw_smooth':
        return float(sum((-1)**(k+1) * np.sin(k*theta) / k
                         for k in range(1, 10)))

    if variant == 'alien':
        return (np.sin(np.cos(theta) * np.pi)
                * np.cos(np.sin(2*theta) * np.pi))

    if variant == 'epileptic':
        return (np.sin(theta) * np.sin(2*theta)
                * np.sin(3*theta) * np.sin(4*theta))

    if variant == 'saw_down':
        return 1 - theta / np.pi

    if variant == 'saw_up':
        return theta / np.pi - 1

    if variant == 'saw_steep':
        if theta < np.pi / 2:
            return 1 - 4 * theta / (2 * np.pi)
        else:
            return 0.0

    if variant == 'saw_gentle':
        return 0.5 * theta / np.pi

    if variant == 'saw_offset':
        t = (theta + np.pi) % (2 * np.pi)
        return 1 - t / np.pi

    if variant == 'double_saw':
        t = theta % np.pi
        return 1 - 2 * t / np.pi

    if variant == 'triple_saw':
        t = theta % (2 * np.pi / 3)
        return 1 - 3 * t / np.pi

    if variant == 'saw_flipped':
        t = theta % np.pi
        tooth = 2 * t / np.pi - 1
        k = int(theta / np.pi)
        return tooth if k % 2 == 0 else -tooth

    if variant == 'saw_abs':
        t = theta % np.pi
        return abs(1 - 2 * t / np.pi)

    if variant == 'zigzag':
        t1 = theta % (2 * np.pi)
        t2 = theta % np.pi
        return (1 - t1 / np.pi) + 0.5 * (1 - 2 * t2 / np.pi)

    if variant == 'saw_sin':
        t = theta % (2 * np.pi)
        saw = 1 - t / np.pi
        return saw * abs(np.sin(theta))

    if variant == 'saw_squared':
        t = theta % (2 * np.pi)
        return 1 - (t / (2 * np.pi)) ** 2

    if variant == 'ramp_drop':
        if theta < 3 * np.pi / 2:
            return theta / (3 * np.pi / 2) - 0.5
        else:
            return -0.5

    if variant == 'step2':
        return -1.0 if theta < np.pi else 1.0

    if variant == 'step3':
        if theta < 2 * np.pi / 3:
            return -1.0
        elif theta < 4 * np.pi / 3:
            return 0.0
        else:
            return 1.0

    if variant == 'step4':
        levels = [-1.0, -1/3, 1/3, 1.0]
        idx = int(theta / (np.pi / 2))
        return levels[min(idx, 3)]

    if variant == 'half_half':
        return np.sin(theta) if theta < np.pi else -1.0

    if variant == 'random_step':
        k = int(theta / (np.pi / 4))
        vals = [0.3, -0.7, 1.0, -0.2, 0.8, -1.0, 0.5, -0.4]
        return vals[k % 8]

    if variant == 'cantor_like':
        t = theta / (2 * np.pi)
        for _ in range(4):
            frac = (t * 3) % 1
            if 1/3 <= (t * 3) % 3 < 2/3:
                return 0.0
            t = frac
        return 1.0

    if variant == 'sin_plus_sin':
        return np.sin(theta) + np.sin(phi)
    if variant == 'sin_minus_sin':
        return np.sin(theta) - np.sin(phi)
    if variant == 'sin_times_sin':
        return np.sin(theta) * np.sin(phi)

    if variant == 'sin_theta_cos_phi':
        return np.sin(theta) * np.cos(phi)
    if variant == 'cos_theta_sin_phi':
        return np.cos(theta) * np.sin(phi)

    if variant == 'diagonal':
        return np.sin(theta + phi)
    if variant == 'antidiagonal':
        return np.sin(theta - phi)

    if variant == 'max_sin':
        return max(np.sin(theta), np.sin(phi))
    if variant == 'min_sin':
        return min(np.sin(theta), np.sin(phi))

    if variant == 'saddle':
        return np.cos(theta) - np.cos(phi)

    if variant == 'z_coord':
        return r * np.sin(phi)
    if variant == 'x_coord':
        return (R + r * np.cos(phi)) * np.cos(theta)
    if variant == 'y_coord':
        return (R + r * np.cos(phi)) * np.sin(theta)

    if variant == 'radial':
        dtheta = min(abs(theta - np.pi), 2 * np.pi - abs(theta - np.pi))
        dphi = min(abs(phi - np.pi), 2 * np.pi - abs(phi - np.pi))
        return -np.sqrt(dtheta**2 + dphi**2)

    if variant == 'constant':
        return 0.0

    if variant == 'linear_combo':
        return theta + np.sqrt(2) * phi

    if variant == 'linear_combo2':
        return theta + np.e * phi

    if variant == 'linear_combo3':
        return np.sqrt(3) * theta + np.sqrt(7) * phi

    if variant == 'sin_irr':
        return np.sin(theta + np.sqrt(2) * phi)

    if variant == 'cos_irr':
        return np.cos(theta + np.pi * phi)

    if variant == 'spiral':
        return theta + np.sqrt(2) * phi + 0.1 * np.sin(3 * theta)

    if variant == 'tilt_sin':
        return np.sin(theta) + 1e-4 * np.sqrt(2) * phi

    if variant == 'tilt_cos':
        return np.cos(theta) + 1e-4 * np.sqrt(3) * phi

    if variant == 'tilt_saddle':
        return np.cos(theta) - np.cos(phi) + 1e-4 * np.sqrt(5) * (theta + phi)

    if variant == 'tilt_sin_plus_sin':
        return np.sin(theta) + np.sin(phi) + 1e-4 * np.sqrt(2) * theta

    if variant == 'hash':
        return np.sin(theta) + np.sin(np.sqrt(2) * theta) + np.sin(phi) + np.sin(np.sqrt(3) * phi)

    if variant == 'hash2':
        return (np.sin(theta + np.sqrt(2) * phi)
                + 0.5 * np.cos(np.sqrt(3) * theta - phi)
                + 0.25 * np.sin(np.sqrt(5) * theta + np.sqrt(7) * phi))

    if variant == 'golden_ramp':
        golden = (1 + np.sqrt(5)) / 2
        return theta + golden * phi

    if variant == 'log_spiral':
        return np.log1p(theta + np.sqrt(2) * phi)

    if variant == 'poly':
        return theta**2 + np.sqrt(2) * theta * phi + np.pi * phi**2

    if variant == 'rank':
        return 100 * theta + phi
    
    if variant == 'sawtooth_sum':
        # Sawtooth in both θ and φ simultaneously.
        # Each component is 1 − x/π on [0, 2π), dropping from +1 at x=0
        # to −1 just before x=2π, with a discontinuity at x=0.
        return (1 - theta / np.pi) + (1 - phi / np.pi)
    
    if variant == 'sawtooth_product':
        return (1 - theta / np.pi) * (1 - phi / np.pi)

    if variant == 'sawtooth_diag':
        s = ((theta + phi) / (2 * np.pi)) % 1.0   # ∈ [0, 1)
        return 1 - 2 * s                          # ∈ (−1, +1]

        


# =============================================================================
# FILTRATION
# =============================================================================

class EdgeWeightFiltration(Filtration):
    """Nodes at time 0, edges at their weight."""

    def __init__(self, G: nx.DiGraph):
        self.G = G

    def node_time(self, node) -> float:
        return 0.0

    def edge_time(self, edge) -> float:
        return self.G.edges[edge].get('weight', 1.0) if self.G.has_edge(*edge) else np.inf

    def node_iter(self):
        return [(v, 0) for v in self.G.nodes]

    def edge_iter(self):
        return [((u, v), self.G.edges[u, v].get('weight', 1.0)) for u, v in self.G.edges]

    def edge_dict(self):
        return {u: {v: self.G.edges[u, v].get('weight', 1.0)
                    for v in self.G.successors(u)} for u in self.G.nodes}

    def ground(self, grounding_G):
        return ProperGroundedFiltration(grounding_G, self)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
#
# Edge weights are the induced-metric arc lengths along straight segments in
# the (θ, φ) parameter domain, computed under the first fundamental form of
# the standard torus embedding in ℝ³:
#
#   ds² = (R + r cos φ)² dθ² + r² dφ²
#
# As a result, every edge weight genuinely depends on R and r:
#   - Points near φ = 0 (outer equator) have a larger effective θ-spacing
#     because the circumference (R + r)·Δθ is longer there.
#   - Points near φ = π (inner equator) have a smaller effective θ-spacing
#     because the circumference (R − r)·Δθ is shorter there.
#
# Consequence: the persistence diagram shifts as R/r changes even for height
# variants that don't explicitly depend on R and r (e.g. sin_theta, sin_phi).
# The graph topology (which edges exist, their directions) is determined solely
# by the height function, but all edge *weights* rescale non-uniformly with
# R and r, so the barcode birth/death values change.
#
# Note: this is NOT the geodesic distance on the torus (which would require
# solving a non-linear ODE). It is the arc length of the straight segment in
# parameter space, which is deterministic, fast to compute, and reduces to the
# flat metric √(Δθ² + Δφ²) in the limit R → ∞ (up to overall scale).
# =============================================================================

def torus_dist(theta1: float, phi1: float, theta2: float, phi2: float,
               R: float, r: float, n_steps: int = 64) -> float:
    """Arc length of the straight (θ, φ)-segment under the induced metric.

    Parametrizes the path as
        θ(s) = θ₁ + s·Δθ,  φ(s) = φ₁ + s·Δφ,  s ∈ [0, 1]
    where Δθ and Δφ are the *signed* shorter-wraparound differences, and
    integrates the line element
        ds = √( (R + r cos φ(s))² (Δθ)² + r² (Δφ)² ) ds

    Properties
    ----------
    - This is the induced-metric arc length along the straight (θ, φ) segment,
      NOT the intrinsic geodesic on the torus.
    - It reduces to the flat metric √(Δθ² + Δφ²) in the limit R → ∞, r fixed,
      because then (R + r cos φ) / R → 1 and the metric becomes approximately
      R·√(Δθ² + (r/R)² Δφ²) ≈ R·|Δθ| for pure-θ displacements.
    - More precisely, for pure-θ displacements at a fixed φ the distance is
      (R + r cos φ)·|Δθ|, which is the circumference arc length at that φ.
    - For pure-φ displacements the distance is r·|Δφ|, independent of θ.
    - Integration uses the composite Simpson rule with n_steps sub-intervals
      (must be even). n_steps=64 gives < 10⁻⁶ relative error for typical
      torus geometries.

    Parameters
    ----------
    theta1, phi1 : float  — start point in parameter space
    theta2, phi2 : float  — end point in parameter space
    R : float             — major radius of the torus (≥ r)
    r : float             — minor radius of the torus (> 0)
    n_steps : int         — number of Simpson sub-intervals (must be even)
    """
    # Signed shorter-wraparound differences in each coordinate
    dtheta = (theta2 - theta1) % (2 * np.pi)
    if dtheta > np.pi:
        dtheta -= 2 * np.pi

    dphi = (phi2 - phi1) % (2 * np.pi)
    if dphi > np.pi:
        dphi -= 2 * np.pi

    # Composite Simpson integration over s ∈ [0, 1]
    # Integrand: sqrt((R + r*cos(phi1 + s*dphi))^2 * dtheta^2 + r^2 * dphi^2)
    if n_steps % 2 != 0:
        n_steps += 1  # ensure even for Simpson's rule

    s = np.linspace(0.0, 1.0, n_steps + 1)
    phi_s = phi1 + s * dphi
    integrand = np.sqrt((R + r * np.cos(phi_s))**2 * dtheta**2 + r**2 * dphi**2)

    # Simpson weights: 1, 4, 2, 4, 2, ..., 4, 1  (scaled by h/3)
    h = 1.0 / n_steps
    weights = np.ones(n_steps + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0
    return float(h / 3.0 * np.dot(weights, integrand))


def _add_directed_edge(G: nx.DiGraph, u: int, v: int, w: float,
                        hu: float, hv: float, double_edges: bool) -> None:
    """Add directed edge(s) between u and v based on height comparison."""
    if abs(hu - hv) < 1e-10:
        if double_edges:
            G.add_edge(u, v, weight=w)
            G.add_edge(v, u, weight=w)
    elif hu < hv:
        G.add_edge(u, v, weight=w)
    else:
        G.add_edge(v, u, weight=w)


def build_graph(cfg: Config) -> tuple[nx.DiGraph, list[float], list[tuple[float, float]]]:
    """
    Build a directed graph on all sample points of T².

    Iterates over every unordered pair ``{u, v}`` and adds directed edges via
    ``_add_directed_edge``, which mirrors the construction used in
    ``circle_pph_2.py`` exactly — replacing ``arc_len`` with ``torus_dist``:

    * ``|f(u) - f(v)| < 1e-10`` (tie): add u→v **and** v→u if
      ``cfg.double_edges``, otherwise skip the pair entirely.
    * ``f(u) < f(v)``  →  add edge u→v with weight ``torus_dist(u, v)``.
    * ``f(u) > f(v)``  →  add edge v→u with weight ``torus_dist(u, v)``.

    This yields at most ``N(N-1)/2`` edges (no ties, or ties with
    ``double_edges=False``) and up to ``N(N-1)`` edges (all ties,
    ``double_edges=True``).

    Edge weights are induced-metric arc lengths (see torus_dist); they depend
    on cfg.R and cfg.r. The result cache key already hashes all Config fields
    including R and r, so results for different (R, r) values never collide.

    Returns: (graph, heights, points)  where points = list of (theta, phi).
    """
    points = sample_torus(cfg)
    N = len(points)
    heights = [height(t, p, cfg.height_variant, cfg.R, cfg.r) for t, p in points]

    G = nx.DiGraph()
    for i, (t, p) in enumerate(points):
        G.add_node(i, height=heights[i], theta=t, phi=p)

    for u in range(N):
        for v in range(u + 1, N):
            tu, pu = points[u]
            tv, pv = points[v]
            w = torus_dist(tu, pu, tv, pv, cfg.R, cfg.r)
            _add_directed_edge(G, u, v, w, heights[u], heights[v], cfg.double_edges)

    return G, heights, points


# =============================================================================
# PPH COMPUTATION
# =============================================================================

def compute_pph(G: nx.DiGraph) -> list[tuple[float, float]]:
    """Compute PPH barcode."""
    pipeline = make_standard_pipeline(
        EdgeWeightFiltration,
        RegularPathHomology,
        backend=LoPHATBackend(with_reps=False),
        optimisation_strat=component_appendage_empty,
    )
    return pipeline(G).barcode


@dataclass
class Result:
    """Analysis result."""
    n: int
    m: int
    height_variant: HeightVariant
    double_edges: bool
    n_nodes: int
    n_edges: int
    barcode: list[tuple[float, float]]

    @property
    def n_bars(self) -> int:
        return len(self.barcode)

    @property
    def max_death(self) -> float:
        if not self.barcode:
            return 0.0
        finite = [b[1] for b in self.barcode if np.isfinite(b[1])]
        return max(finite) if finite else 0.0

    def __repr__(self):
        return (f"Result(n={self.n}, m={self.m}, bars={self.n_bars}, "
                f"max_death={self.max_death/np.pi:.3f}π)")


def analyze(cfg: Config) -> Result:
    """Run full analysis for given configuration, using a disk cache."""
    key = _cfg_key(cfg)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    m = cfg.m if cfg.m is not None else cfg.n
    G, _, _ = build_graph(cfg)
    barcode = compute_pph(G)
    result = Result(
        n=cfg.n,
        m=m,
        height_variant=cfg.height_variant,
        double_edges=cfg.double_edges,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        barcode=barcode,
    )
    _save_cached(key, result)
    return result


def analyze_simple(n: int,
                   double_edges: bool = True,
                   height_variant: HeightVariant = 'sin_theta',
                   sampling: SamplingVariant = 'grid') -> Result:
    """Convenience wrapper."""
    return analyze(Config(
        n=n, height_variant=height_variant,
        double_edges=double_edges, sampling=sampling,
    ))


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def batch_analyze(n_range: range, **kwargs) -> list[Result]:
    """Analyze multiple n values."""
    return [analyze_simple(n, **kwargs) for n in n_range]


def compare(n_range: range, height_variant: HeightVariant = 'sin_theta') -> dict:
    """Compare with/without double edges across n_range."""
    return {
        'without': batch_analyze(n_range, double_edges=False, height_variant=height_variant),
        'with':    batch_analyze(n_range, double_edges=True,  height_variant=height_variant),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

MOD4_COLORS = {0: '#27ae60', 1: '#3498db', 2: '#e74c3c', 3: '#9b59b6'}


def plot_death_times(results_no: list[Result], results_yes: list[Result],
                     output: str | Path | None = None) -> plt.Figure:
    """Plot death times comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [(axes[0], results_no, 'WITHOUT Double Edges'),
                                (axes[1], results_yes, 'WITH Double Edges')]:
        for r in results:
            mod4 = r.n % 4
            for bar in r.barcode:
                death = bar[1] / np.pi if np.isfinite(bar[1]) else 1.05
                ax.scatter(r.n, death, c=MOD4_COLORS[mod4], s=40, alpha=0.7)

        ax.set_xlabel('n (grid side length)')
        ax.set_ylabel('Death time (units of π)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=MOD4_COLORS[i], markersize=10,
                   label=f'n ≡ {i} (mod 4)') for i in range(4)]
        ax.legend(handles=handles, loc='upper right')

    plt.suptitle('PPH Death Times on T²: Effect of Double Edges', fontweight='bold', y=1.02)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def plot_bar_counts(results_no: list[Result], results_yes: list[Result],
                    output: str | Path | None = None) -> plt.Figure:
    """Plot bar count comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ns = [r.n for r in results_no]
    ax.plot(ns, [r.n_bars for r in results_no], 'o-', label='Without double edges',
            color='#2980b9', markersize=5)
    ax.plot(ns, [r.n_bars for r in results_yes], 's--', label='With double edges',
            color='#e74c3c', markersize=5)

    ax.set_xlabel('n (grid side length)')
    ax.set_ylabel('Number of H₁ bars')
    ax.set_title('Bar Count on T²: Effect of Double Edges', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def plot_height_function(variant: HeightVariant = 'sin_theta',
                          R: float = 2.0, r: float = 1.0,
                          output: str | Path | None = None) -> plt.Figure:
    """Plot height function on T² as a 2D heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))

    theta = np.linspace(0, 2 * np.pi, 300)
    phi = np.linspace(0, 2 * np.pi, 300)
    TH, PH = np.meshgrid(theta, phi)
    Z = np.vectorize(lambda t, p: height(t, p, variant, R, r))(TH, PH)

    im = ax.contourf(TH / np.pi, PH / np.pi, Z, levels=40, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='f(θ, φ)')
    ax.set_xlabel('θ / π')
    ax.set_ylabel('φ / π')
    ax.set_title(f'Height function on T²: {variant}', fontweight='bold')

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def plot_persistence_diagram(
    result: Result,
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot the persistence diagram for a single torus PPH Result."""
    fig, ax = plt.subplots(figsize=(6, 6))

    finite = [(b, d) for b, d in result.barcode if np.isfinite(d)]
    infinite = [b for b, d in result.barcode if not np.isfinite(d)]

    all_vals = [b for b, d in finite] + [d for b, d in finite] + infinite
    lo = min(all_vals, default=0.0)
    hi = max(all_vals, default=1.0)
    pad = (hi - lo) * 0.12 if hi > lo else 0.5
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5, zorder=0)

    if finite:
        births, deaths = zip(*finite)
        ax.scatter(births, deaths, s=60, zorder=3,
                   facecolors='#3498db', edgecolors='#1a5276', linewidths=0.8,
                   label=f'finite ({len(finite)})')

    pin_y = hi - pad * 0.3
    if infinite:
        ax.scatter(infinite, [pin_y] * len(infinite),
                   s=80, marker='^', zorder=3,
                   facecolors='#e74c3c', edgecolors='#922b21', linewidths=0.8,
                   label=f'infinite ({len(infinite)})')
        ax.axhline(pin_y, color='#e74c3c', lw=0.6, ls=':', alpha=0.5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi + pad * 0.15)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    de_str = 'double_edges' if result.double_edges else 'single_edges'
    ax.set_title(
        f'Persistence diagram — T²\n'
        f'n={result.n}, m={result.m}, height={result.height_variant!r}, {de_str}',
        fontweight='bold',
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def plot_torus_graph(
    n: int,
    height_variant: HeightVariant = 'sin_theta',
    sampling: SamplingVariant = 'grid',
    double_edges: bool = True,
    arc_min: float = 0.0,
    arc_max: float = float('inf'),
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the directed torus graph in the flat (θ, φ) parameter square.
    Nodes are coloured by height using the RdBu_r diverging colormap.
    Edges whose arc-length weight falls within [arc_min, arc_max] are drawn
    as arrows coloured by weight using the plasma colormap; longer arrows
    are slightly bent. All other edges are suppressed.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cfg = Config(n=n, height_variant=height_variant,
                 sampling=sampling, double_edges=double_edges)
    G, heights, points = build_graph(cfg)

    thetas = [p[0] for p in points]
    phis   = [p[1] for p in points]

    edges_to_draw = [
        (u, v, G.edges[u, v]['weight'])
        for u, v in G.edges()
        if arc_min <= G.edges[u, v]['weight'] <= arc_max
    ]

    # --- figure setup ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # --- edge colormap ---
    if edges_to_draw:
        weights = [w for _, _, w in edges_to_draw]
        w_min, w_max = min(weights), max(weights)
        e_norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
    else:
        e_norm = mcolors.Normalize(vmin=0, vmax=1)
    e_cmap = cm.plasma

    # --- draw edges ---
    for u, v, w in edges_to_draw:
        x0, y0 = thetas[u] / np.pi, phis[u] / np.pi
        x1, y1 = thetas[v] / np.pi, phis[v] / np.pi
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        col = e_cmap(e_norm(w))

        if dx > 1.0 or dy > 1.0:
            # wrapping edge — draw as plain line, no arrow
            ax.plot([x0, x1], [y0, y1], color=col, lw=0.5, alpha=0.35, zorder=1)
        else:
            # non-wrapping edge — curve scales with arrow length
            length = np.hypot(x1 - x0, y1 - y0)
            rad = np.clip(length * 0.3, 0.0, 0.4)
            ax.annotate(
                '', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->',
                    color=col,
                    lw=0.9,
                    connectionstyle=f'arc3,rad={rad}',
                ),
                zorder=2,
            )

    # --- draw nodes ---
    h_arr = np.array(heights)
    h_min, h_max = float(h_arr.min()), float(h_arr.max())

    if h_max - h_min < 1e-12:
        n_norm = mcolors.Normalize(vmin=h_min - 1e-6, vmax=h_max + 1e-6)
    else:
        vc = float(np.median(h_arr))
        vc = max(h_min + 1e-9, min(vc, h_max - 1e-9))
        n_norm = mcolors.TwoSlopeNorm(vmin=h_min, vcenter=vc, vmax=h_max)

    sc = ax.scatter(
        [t / np.pi for t in thetas],
        [p / np.pi for p in phis],
        c=heights, cmap='RdBu_r', norm=n_norm,
        s=50, zorder=4, edgecolors='white', linewidths=0.4,
    )

    # --- colorbars ---
    plt.colorbar(sc, ax=ax, label='Height', fraction=0.046, pad=0.04)
    if edges_to_draw:
        sm = cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge weight', fraction=0.046, pad=0.08)

    # --- labels & styling ---
    ax.set_xlabel('θ / π')
    ax.set_ylabel('φ / π')
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 2.05)
    ax.set_title(
        f'Torus graph  n={n}, height={height_variant!r}, sampling={sampling!r}\n'
        f'double_edges={double_edges}, arc ∈ [{arc_min:.3f}, '
        f'{"∞" if not np.isfinite(arc_max) else f"{arc_max:.3f}"}]',
        fontweight='bold',
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    # --- optional save ---
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def plot_torus_3d(
    n: int,
    height_variant: HeightVariant = 'sin_theta',
    sampling: SamplingVariant = 'grid',
    double_edges: bool = True,
    arc_min: float = 0.0,
    arc_max: float = float('inf'),
    R: float = 4,
    r: float = 1.0,
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the directed torus graph in 3D, overlaid on the torus surface.

    Nodes are coloured by height using the RdBu_r diverging colormap.
    Edges whose arc-length weight falls within [arc_min, arc_max] are drawn
    coloured by weight using the plasma colormap. Wrap-around edges (crossing
    the periodic boundary) are skipped so no lines cut through the torus body.

    Parameters
    ----------
    n             : grid side length (n×n nodes for grid sampling)
    height_variant: height function to use
    sampling      : sampling strategy
    double_edges  : add both u→v and v→u for height-tied pairs
    arc_min       : lower bound on displayed edge weights (inclusive)
    arc_max       : upper bound on displayed edge weights (inclusive)
    R             : major radius of torus (centre to tube centre)
    r             : minor radius of torus (tube radius)
    output        : optional save path
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    def to_xyz(theta, phi):
        """Map torus parameters (θ, φ) → (x, y, z) in ℝ³."""
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        return x, y, z

    cfg = Config(
        n=n,
        height_variant=height_variant,
        sampling=sampling,
        double_edges=double_edges,
        R=R,
        r=r,
    )
    G, heights, points = build_graph(cfg)

    # --- torus surface mesh ---
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, 2 * np.pi, 60)
    U, V = np.meshgrid(u, v)
    Xs = (R + r * np.cos(V)) * np.cos(U)
    Ys = (R + r * np.cos(V)) * np.sin(U)
    Zs = r * np.sin(V)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=45)

    # Surface drawn very transparently so nodes are always visible
    ax.plot_surface(
        Xs, Ys, Zs,
        color='lightgrey',
        alpha=0.08,          # much more transparent than before
        linewidth=0,
        antialiased=True,
        zorder=0,
    )

    # --- precompute node 3-D positions ---
    node_xyz = [to_xyz(t, p) for t, p in points]

    # --- edges ---
    edges_to_draw = [
        (u_idx, v_idx, G.edges[u_idx, v_idx]['weight'])
        for u_idx, v_idx in G.edges()
        if arc_min <= G.edges[u_idx, v_idx]['weight'] <= arc_max
    ]

    if edges_to_draw:
        weights = [w for _, _, w in edges_to_draw]
        w_min, w_max = min(weights), max(weights)
        e_norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
    else:
        e_norm = mcolors.Normalize(vmin=0, vmax=1)
    e_cmap = cm.plasma

    for u_idx, v_idx, w in edges_to_draw:
        tu, pu = points[u_idx]
        tv, pv = points[v_idx]

        # Angular distances on the periodic domain
        d_theta = abs(tu - tv)
        d_theta = min(d_theta, 2 * np.pi - d_theta)
        d_phi   = abs(pu - pv)
        d_phi   = min(d_phi,   2 * np.pi - d_phi)
        is_wrap = d_theta > np.pi or d_phi > np.pi

        # Skip wrap-around edges — they would cut through the torus body
        if is_wrap:
            continue

        x0, y0, z0 = node_xyz[u_idx]
        x1, y1, z1 = node_xyz[v_idx]
        col = e_cmap(e_norm(w))

        ax.plot(
            [x0, x1], [y0, y1], [z0, z1],
            color=col, lw=0.7, alpha=0.55, zorder=2,
        )

    # --- nodes drawn last so they sit on top of edges and surface ---
    h_arr = np.array(heights)
    h_norm = mcolors.Normalize(vmin=h_arr.min(), vmax=h_arr.max())
    n_cmap = cm.RdBu_r
    node_colors = [n_cmap(h_norm(h)) for h in heights]

    xs, ys, zs = zip(*node_xyz)
    ax.scatter(
        xs, ys, zs,
        c=node_colors,
        s=60,                # larger than before
        zorder=6,
        depthshade=False,    # keep full brightness regardless of occlusion
        edgecolors='white',
        linewidths=0.5,
    )

    # --- colorbars ---
    sm_nodes = cm.ScalarMappable(cmap=n_cmap, norm=h_norm)
    sm_nodes.set_array([])
    cb_nodes = fig.colorbar(sm_nodes, ax=ax, shrink=0.5, pad=0.0, location='left')
    cb_nodes.set_label('Height', fontsize=9)

    if edges_to_draw:
        sm_edges = cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
        sm_edges.set_array([])
        cb_edges = fig.colorbar(sm_edges, ax=ax, shrink=0.5, pad=0.02, location='right')
        cb_edges.set_label('Arc-length weight', fontsize=9)

    ax.set_title(
        f'Torus graph — {height_variant}, {sampling}, double_edges={double_edges}',
        fontsize=11,
    )
    ax.set_axis_off()
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150, bbox_inches='tight')

    return fig

def print_table(results_no: list[Result], results_yes: list[Result]) -> None:
    """Print comparison table."""
    print(f"\n{'n':>3} {'n²':>5} {'mod4':>5} │ {'Without':>8} │ {'With':>8} │ {'Δ':>4}")
    print("─" * 42)

    for r_no, r_yes in zip(results_no, results_yes):
        mod4 = r_no.n % 4
        diff = r_yes.n_bars - r_no.n_bars
        marker = '←' if diff != 0 else ''
        print(f"{r_no.n:3d} {r_no.n_nodes:5d} {mod4:5d} │"
              f" {r_no.n_bars:8d} │ {r_yes.n_bars:8d} │ {diff:+3d} {marker}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPH analysis on T²")
    parser.add_argument('--n', type=int, help='Single n value to analyze (n×n grid)')
    parser.add_argument('--quick', action='store_true', help='Quick test (n=2..5)')
    parser.add_argument('--full',  action='store_true', help='Full sweep (n=2..15); slow')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Output directory')
    parser.add_argument('--height', default='sin_theta',
                        help='Height variant (e.g. sin_theta, quarter_triangle, damped)')
    parser.add_argument('--arc-min', type=float, default=0.0,
                        help='Minimum edge arc-length shown in plot_torus_graph (default 0)')
    parser.add_argument('--arc-max', type=float, default=float('inf'),
                        help='Maximum edge arc-length shown in plot_torus_graph (default ∞)')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.n:
        for de in [False, True]:
            r = analyze_simple(args.n, double_edges=de, height_variant=args.height)
            de_str = "with" if de else "without"
            print(f"n={args.n} ({args.n}×{args.n}={r.n_nodes} pts, {r.n_edges} edges) "
                  f"{de_str} double edges: {r.n_bars} bars")
            for b in r.barcode:
                death_str = f"{b[1]/np.pi:.3f}π" if np.isfinite(b[1]) else "∞"
                print(f"  [{b[0]/np.pi:.3f}π, {death_str}]")

        r_de = analyze_simple(args.n, double_edges=True, height_variant=args.height)
        plot_persistence_diagram(
            r_de,
            output=args.output_dir / f'torus_persistence_n{args.n}.png',
        )
        plot_torus_graph(
            n=args.n,
            height_variant=args.height,
            double_edges=True,
            arc_min=args.arc_min,
            arc_max=args.arc_max,
            output=args.output_dir / f'torus_graph_n{args.n}.png',
        )
        plot_torus_3d(
            n=args.n,
            height_variant=args.height,
            double_edges=True,
            arc_min=args.arc_min,
            arc_max=args.arc_max,
            output=args.output_dir / f'torus_graph_3d_n{args.n}.png',
        )
        return

    if args.full:
        n_range = range(2, 16)
    elif args.quick:
        n_range = range(2, 6)
    else:
        n_range = range(2, 10)

    print("=" * 60)
    print("PERSISTENT PATH HOMOLOGY ON T²")
    print("=" * 60)

    variants_to_run: list[HeightVariant] = [args.height]  # type: ignore[list-item]
    if args.height == 'sin_theta':
        variants_to_run = [          # type: ignore[list-item]
            'sin_theta', 'quarter_triangle', 'damped',
            'sin_plus_sin2', 'sigmoid_periodic', 'fractal3',
        ]

    for variant in variants_to_run:
        print(f"\n[Height variant: {variant}]")
        results = compare(n_range, variant)  # type: ignore[arg-type]
        print_table(results['without'], results['with'])

        stem = f'torus_{variant}'
        plot_death_times(results['without'], results['with'],
                         args.output_dir / f'{stem}_death_times.png')
        plot_bar_counts(results['without'], results['with'],
                        args.output_dir / f'{stem}_bar_counts.png')
        plot_height_function(variant,  # type: ignore[arg-type]
                             output=args.output_dir / f'{stem}_height_function.png')

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()