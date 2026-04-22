"""
Persistent Path Homology on S¹
==============================

Clean, consolidated implementation for analyzing PPH on circle samples.

Usage:
    python circle_pph.py                    # Run full analysis
    python circle_pph.py --quick            # Quick test
    python circle_pph.py --n 12             # Single analysis
"""

import os
import hashlib
import pickle
import dataclasses
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

# Limit the Rust/LoPHAT Rayon thread-pool to one thread so batch runs
# don't saturate all cores.  Set before importing grpphati.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

# Homology computation
from grpphati.filtrations import Filtration, ProperGroundedFiltration
from grpphati.homologies import RegularPathHomology, DirectedFlagComplexHomology
from grpphati.pipelines.standard import make_standard_pipeline
from grpphati.backends import LoPHATBackend
from grpphati.optimisations import component_appendage_empty

HomologyVariant = Literal['path', 'flag']

_HOMOLOGY_CLASSES = {
    'path': RegularPathHomology,
    'flag': DirectedFlagComplexHomology,
}


# =============================================================================
# RESULT CACHE
# =============================================================================

_CACHE_DIR = Path(".homology_cache") / "circle"


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
    'standard', 'double_max', 'double_max_mod',
    'squeeze', 'plateau', 'asymmetric', 'valley', 'shallow', 'constant',
    'sin2', 'cos', 'cos2', 'sin3', 'cos3', 'sin4', 'sin5', 'cos5',
    'sawtooth', 'triangle', 'square', 'saw2', 'tri2', 'tri3',
    'sin_sq', 'abs_sin', 'abs_cos',
    'sin_plus_sin2', 'sin_plus_sin3', 'sin_plus_cos2',
    'sin_cos', 'asymmetric2', 'bump',
    'sin_cubed', 'cos_cubed',
    'double_min', 'triple_max', 'skewed',
    'sin_abs_cos', 'cos_abs_sin',
    'sigmoid_periodic', 'clipped_sin', 'rectified_sin',
    'sin_shifted', 'cos_shifted_third',
    'devil', 'heartbeat', 'fractal3', 'wobble', 'ekg',
    'teeth', 'chaos', 'nested_sin', 'mod_wave', 'phase_mod',
    'ripple', 'dragon', 'staircase', 'interference', 'tangled',
    'bouncy', 'lopsided', 'saw_smooth', 'alien', 'epileptic',
    'saw_down', 'saw_up', 'saw_steep', 'saw_gentle',
    'saw_offset', 'double_saw', 'triple_saw',
    'saw_flipped', 'saw_abs', 'zigzag',
    'saw_sin', 'saw_squared', 'ramp_drop',
    'step2', 'step3', 'step4',
    'half_half', 'random_step', 'cantor_like',
    'quarter_triangle', 'damped',
]

SamplingVariant = Literal[
    'uniform',
    'jittered',
    'clustered',
    'two_clusters',
    'beta',
    'chebyshev',
    'random',
    'custom',
]


@dataclass
class Config:
    """Analysis configuration."""
    n: int
    height_variant: HeightVariant = 'standard'
    double_edges: bool = True
    dip_depth: float = 1.0
    epsilon0: float = 0.05
    perturb_index: int | None = None
    perturb_delta: float = 0.0
    # --- sampling ---
    sampling: SamplingVariant = 'uniform'
    jitter_std: float = 0.3
    cluster_std: float = 0.3
    beta_a: float = 0.5
    beta_b: float = 0.5
    random_seed: int | None = None
    custom_angles: list[float] | None = None
    # --- homology ---
    homology: HomologyVariant = 'path'


# =============================================================================
# SAMPLING
# =============================================================================

def sample_angles(cfg: Config) -> list[float]:
    """Generate n angles in [0, 2π) according to cfg.sampling."""
    n = cfg.n
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.sampling == 'uniform':
        angles = [i * 2 * np.pi / n for i in range(n)]

    elif cfg.sampling == 'jittered':
        spacing = 2 * np.pi / n
        base = np.array([i * spacing for i in range(n)])
        noise = rng.normal(0, cfg.jitter_std * spacing, size=n)
        angles = sorted((base + noise) % (2 * np.pi))

    elif cfg.sampling == 'clustered':
        raw = rng.normal(0, cfg.cluster_std, size=n)
        angles = sorted(a % (2 * np.pi) for a in raw)

    elif cfg.sampling == 'two_clusters':
        half = n // 2
        rest = n - half
        c1 = rng.normal(0, cfg.cluster_std, size=half)
        c2 = rng.normal(np.pi, cfg.cluster_std, size=rest)
        angles = sorted(a % (2 * np.pi) for a in np.concatenate([c1, c2]))

    elif cfg.sampling == 'beta':
        raw = rng.beta(cfg.beta_a, cfg.beta_b, size=n)
        angles = sorted(float(a) * 2 * np.pi for a in raw)

    elif cfg.sampling == 'chebyshev':
        k = np.arange(1, n + 1)
        nodes = np.cos((2*k - 1) * np.pi / (2*n))
        angles = sorted((node + 1) * np.pi for node in nodes)

    elif cfg.sampling == 'random':
        angles = sorted(rng.uniform(0, 2 * np.pi, size=n).tolist())

    elif cfg.sampling == 'custom':
        if cfg.custom_angles is None or len(cfg.custom_angles) != n:
            raise ValueError(f"custom_angles must be a list of length n={n}")
        angles = sorted(a % (2 * np.pi) for a in cfg.custom_angles)

    else:
        raise ValueError(f"Unknown sampling variant: {cfg.sampling}")

    return [float(a) for a in angles]


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
# HEIGHT FUNCTIONS
# =============================================================================

def height(theta: float, variant: HeightVariant = 'standard', dip_depth: float = 1.0,
           epsilon0: float = 0.05) -> float:
    """Height function on S¹."""
    theta = theta % (2 * np.pi)

    if variant == 'standard':
        return np.sin(theta)

    if variant == 'double_max':
        if theta < np.pi / 2:
            return 1 - dip_depth * np.sin(2 * theta)
        else:
            return np.sin(theta)

    if variant == 'double_max_mod':
        if theta == np.pi / 2:
            return 1 - epsilon0
        elif theta < np.pi / 2:
            return 1 - dip_depth * np.sin(2 * theta)
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

    if variant == 'constant':
        return 0.0

    if variant == 'sin2':
        return np.sin(2 * theta)

    if variant == 'cos':
        return np.cos(theta)

    if variant == 'cos2':
        return np.cos(2 * theta)

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

    if variant == 'sawtooth':
        return 1 - theta / np.pi
    if variant == 'sawtooth_continuous':
        return (2 / np.pi) * sum(np.sin(k * theta) / k for k in range(1, 20))

    if variant == 'triangle':
        if theta < np.pi:
            return 1 - 2 * theta / np.pi
        else:
            return -3 + 2 * theta / np.pi
    if variant == 'damped':
        return np.exp(-0.5 * theta) * np.sin(4 * theta)

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

    if variant == 'abs_sin':
        return abs(np.sin(theta))

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
        return np.exp(-8 * (theta - np.pi/2)**2) - np.exp(-8 * (theta - 3*np.pi/2)**2)

    if variant == 'sin_cubed':
        return np.sin(theta) ** 3

    if variant == 'cos_cubed':
        return np.cos(theta) ** 3

    if variant == 'double_min':
        if theta >= np.pi:
            return -1 + dip_depth * np.cos(2 * theta)
        else:
            return np.sin(theta)

    if variant == 'triple_max':
        return np.sin(theta) ** 2 * np.cos(theta)

    if variant == 'skewed':
        return np.sin(theta) + 0.5 * np.sin(2*theta) + 0.25 * np.sin(3*theta)

    if variant == 'sin_abs_cos':
        return np.sin(theta) * abs(np.cos(theta))

    if variant == 'cos_abs_sin':
        return np.cos(theta) * abs(np.sin(theta))

    if variant == 'sigmoid_periodic':
        return np.tanh(3 * np.sin(theta))

    if variant == 'clipped_sin':
        return float(np.clip(np.sin(theta), -0.5, 0.5))

    if variant == 'rectified_sin':
        return max(0.0, np.sin(theta))

    if variant == 'sin_shifted':
        return np.sin(theta + np.pi / 3)

    if variant == 'cos_shifted_third':
        return np.cos(theta + 2 * np.pi / 3)

    if variant == 'devil':
        return np.sin(theta) * np.cos(3*theta) + 0.5 * np.sin(5*theta) * np.cos(theta)

    if variant == 'heartbeat':
        return np.exp(-10 * (np.sin(theta/2))**2) * np.sin(3*theta)

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
        return np.sin(theta) + 0.2*np.sin(7*theta) + 0.1*np.sin(13*theta)

    if variant == 'dragon':
        return np.sin(theta)**3 + np.cos(3*theta)**2 * np.sin(2*theta)

    if variant == 'staircase':
        return (np.sin(theta) + np.sin(3*theta)/3 - np.sin(5*theta)/5
                + np.sin(7*theta)/7 - np.sin(9*theta)/9)

    if variant == 'interference':
        return np.sin(theta) * np.cos(0.5*theta) + np.cos(theta) * np.sin(1.5*theta)

    if variant == 'tangled':
        return np.sin(2*theta + np.cos(3*theta)) * np.cos(theta - np.sin(2*theta))

    if variant == 'bouncy':
        return np.sin(4*theta) * np.exp(-abs(np.sin(theta/2)))

    if variant == 'lopsided':
        return np.sin(theta)**5 + np.cos(2*theta)**3 - 0.5*np.sin(3*theta)

    if variant == 'saw_smooth':
        return sum((-1)**(k+1) * np.sin(k*theta) / k for k in range(1, 10))

    if variant == 'alien':
        return np.sin(np.cos(theta) * np.pi) * np.cos(np.sin(2*theta) * np.pi)

    if variant == 'epileptic':
        return np.sin(theta) * np.sin(2*theta) * np.sin(3*theta) * np.sin(4*theta)

    # --- sawtooth family ---

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

    # --- discontinuous family ---

    if variant == 'step2':
        # one jump: -1 on [0, π), +1 on [π, 2π)
        return -1.0 if theta < np.pi else 1.0

    if variant == 'step3':
        # three levels: -1, 0, +1
        if theta < 2 * np.pi / 3:
            return -1.0
        elif theta < 4 * np.pi / 3:
            return 0.0
        else:
            return 1.0

    if variant == 'step4':
        # four levels: -1, -1/3, +1/3, +1
        levels = [-1.0, -1/3, 1/3, 1.0]
        idx = int(theta / (np.pi / 2))
        return levels[min(idx, 3)]

    if variant == 'half_half':
        # sin on [0, π), constant -1 on [π, 2π)
        return np.sin(theta) if theta < np.pi else -1.0

    if variant == 'random_step':
        # deterministic pseudorandom steps (hash-based, reproducible)
        k = int(theta / (np.pi / 4))
        vals = [0.3, -0.7, 1.0, -0.2, 0.8, -1.0, 0.5, -0.4]
        return vals[k % 8]

    if variant == 'cantor_like':
        # middle-thirds removal: 0 in gap regions, 1 elsewhere
        t = theta / (2 * np.pi)
        for _ in range(4):
            frac = (t * 3) % 1
            if 1/3 <= (t * 3) % 3 < 2/3:
                return 0.0
            t = frac
        return 1.0
    if variant == 'quarter_triangle':
        if theta <= np.pi / 2:
            return (2 / np.pi) * theta
        else:
            return 1 - (theta - np.pi / 2) / (3 * np.pi / 2)



# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph(cfg: Config) -> tuple[nx.DiGraph, list[float], list[float]]:
    """
    Build directed graph from n sample points on S¹.

    Returns: (graph, heights, angles)
    """
    n = cfg.n
    angles = sample_angles(cfg)
    heights = [height(a, cfg.height_variant, cfg.dip_depth, cfg.epsilon0) for a in angles]

    if cfg.perturb_index is not None and 0 <= cfg.perturb_index < n:
        heights[cfg.perturb_index] += cfg.perturb_delta

    def arc_len(i: int, j: int) -> float:
        diff = abs(angles[i] - angles[j])
        return min(diff, 2*np.pi - diff)

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, height=heights[i], angle=angles[i])

    for i in range(n):
        for skip in range(1, n//2 + 1):
            j = (i + skip) % n
            hi, hj = heights[i], heights[j]
            w = arc_len(i, j)

            if abs(hi - hj) < 1e-10:
                if cfg.double_edges:
                    G.add_edge(i, j, weight=w)
                    G.add_edge(j, i, weight=w)
            elif hi < hj:
                G.add_edge(i, j, weight=w)
            else:
                G.add_edge(j, i, weight=w)

    return G, heights, angles


# =============================================================================
# HOMOLOGY COMPUTATION
# =============================================================================

def compute_homology(G: nx.DiGraph,
                     homology: HomologyVariant = 'path') -> list[tuple[float, float]]:
    """Compute persistent homology barcode.

    Parameters
    ----------
    G        : directed graph with edge weights
    homology : 'path' for regular path homology (default),
               'flag' for directed flag homology
    """
    pipeline = make_standard_pipeline(
        EdgeWeightFiltration,
        _HOMOLOGY_CLASSES[homology],
        backend=LoPHATBackend(with_reps=False),
        optimisation_strat=component_appendage_empty,
    )
    return pipeline(G).barcode


# Keep old name as a convenience alias
def compute_pph(G: nx.DiGraph) -> list[tuple[float, float]]:
    """Alias for compute_homology(G, 'path')."""
    return compute_homology(G, 'path')


@dataclass
class Result:
    """Analysis result."""
    n: int
    mod4: int
    height_variant: HeightVariant
    double_edges: bool
    n_edges: int
    barcode: list[tuple[float, float]]
    homology: HomologyVariant = 'path'

    @property
    def n_bars(self) -> int:
        return len(self.barcode)

    @property
    def max_death(self) -> float:
        if not self.barcode:
            return 0.0
        return max(b[1] for b in self.barcode if np.isfinite(b[1]))

    def __repr__(self):
        return f"Result(n={self.n}, homology={self.homology!r}, bars={self.n_bars}, max_death={self.max_death/np.pi:.3f}π)"


def analyze(cfg: Config) -> Result:
    """Run full analysis for given configuration, using a disk cache.

    Results are keyed by all Config fields (including cfg.homology) and stored
    under .homology_cache/circle/.  A cache hit avoids re-running the pipeline.
    """
    key = _cfg_key(cfg)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    G, _, _ = build_graph(cfg)
    barcode = compute_homology(G, cfg.homology)

    result = Result(
        n=cfg.n,
        mod4=cfg.n % 4,
        height_variant=cfg.height_variant,
        double_edges=cfg.double_edges,
        n_edges=G.number_of_edges(),
        barcode=barcode,
        homology=cfg.homology,
    )
    _save_cached(key, result)
    return result


def analyze_simple(n: int, double_edges: bool = True,
                   height_variant: HeightVariant = 'standard',
                   sampling: SamplingVariant = 'uniform',
                   homology: HomologyVariant = 'path') -> Result:
    """Convenience wrapper."""
    return analyze(Config(n=n, double_edges=double_edges,
                          height_variant=height_variant, sampling=sampling,
                          homology=homology))


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def batch_analyze(n_range: range, **kwargs) -> list[Result]:
    """Analyze multiple n values."""
    return [analyze_simple(n, **kwargs) for n in n_range]


def compare(n_range: range, height_variant: HeightVariant = 'standard',
            homology: HomologyVariant = 'path') -> dict:
    """Compare with/without double edges."""
    return {
        'without': batch_analyze(n_range, double_edges=False, height_variant=height_variant, homology=homology),
        'with': batch_analyze(n_range, double_edges=True, height_variant=height_variant, homology=homology),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

MOD4_COLORS = {0: '#27ae60', 1: '#3498db', 2: '#e74c3c', 3: '#9b59b6'}


def plot_death_times(results_no: list[Result], results_yes: list[Result],
                     output: str | Path | None = None,
                     homology: HomologyVariant = 'path') -> plt.Figure:
    """Plot death times comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [(axes[0], results_no, 'WITHOUT Double Edges'),
                                (axes[1], results_yes, 'WITH Double Edges')]:
        for r in results:
            for bar in r.barcode:
                death = bar[1]/np.pi if np.isfinite(bar[1]) else 1.05
                ax.scatter(r.n, death, c=MOD4_COLORS[r.mod4], s=40, alpha=0.7)

        ax.axhline(y=0.5, color='black', ls='--', lw=2, label='π/2 (limit)')
        ax.set_xlabel('n (sample size)')
        ax.set_ylabel('Death time (units of π)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        handles = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=MOD4_COLORS[i], markersize=10,
                   label=f'n ≡ {i} (mod 4)') for i in range(4)]
        ax.legend(handles=handles, loc='upper right')

    hom_label = 'Path homology' if homology == 'path' else 'Directed flag homology'
    plt.suptitle(f'{hom_label} — Death Times: Effect of Double Edges', fontweight='bold', y=1.02)
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

    ax.set_xlabel('n')
    ax.set_ylabel('Number of H₁ bars')
    ax.set_title('Bar Count: Effect of Double Edges', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yticks([0, 1, 2, 3])

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def print_table(results_no: list[Result], results_yes: list[Result]) -> None:
    """Print comparison table."""
    print(f"\n{'n':>3} {'mod4':>5} │ {'Without':>8} │ {'With':>8} │ {'Δ':>4}")
    print("─" * 35)

    for r_no, r_yes in zip(results_no, results_yes):
        diff = r_yes.n_bars - r_no.n_bars
        marker = '←' if diff != 0 else ''
        print(f"{r_no.n:3d} {r_no.mod4:5d} │ {r_no.n_bars:8d} │ {r_yes.n_bars:8d} │ {diff:+3d} {marker}")

def plot_circle_graph(n, height_variant='standard', sampling='uniform', double_edges=True, 
                      arc_min=0, arc_max=np.pi/2):
    G, heights, angles = build_graph(Config(
        n=n, height_variant=height_variant,
        sampling=sampling, double_edges=double_edges,
    ))
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
    h_min, h_max = min(heights), max(heights)
    node_colors = plt.cm.RdYlBu_r([(h - h_min) / (h_max - h_min + 1e-9) for h in heights])
    filtered_edges = [(u, v) for u, v in G.edges() 
                      if arc_min <= G.edges[u, v]['weight'] <= arc_max]
    weights = [G.edges[u, v]['weight'] for u, v in filtered_edges]
    fig, ax = plt.subplots(figsize=(7, 7))
    if weights:
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=0, vmax=np.pi)  # ← fixed range, independent of arc_min/max
        for (u, v), w in zip(filtered_edges, weights):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                   arrows=True, arrowsize=15,
                                   edge_color=[cmap(norm(w))],
                                   width=2.0, connectionstyle='arc3,rad=0.1')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='arc length', shrink=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='white', font_weight='bold')
    for i in range(n):
        x, y = pos[i]
        ax.text(x * 1.25, y * 1.25, f'h={heights[i]:.2f}', fontsize=8, ha='center')
    ax.set_title(f'{height_variant} | {sampling} | n={n} | arc=[{arc_min/np.pi:.2g}π, {arc_max/np.pi:.2g}π]', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPH analysis on S¹")
    parser.add_argument('--n', type=int, help='Single n value to analyze')
    parser.add_argument('--quick', action='store_true', help='Quick test (n=3..10)')
    parser.add_argument('--full',  action='store_true', help='Full sweep (n=3..50); slow')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Output directory')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.n:
        for de in [False, True]:
            r = analyze_simple(args.n, double_edges=de)
            de_str = "with" if de else "without"
            print(f"n={args.n} {de_str} double edges: {r.n_bars} bars")
            for b in r.barcode:
                print(f"  [{b[0]/np.pi:.3f}π, {b[1]/np.pi:.3f}π]")
        return

    # Default: n=3..20 (interactive-safe).  --full extends to 50; --quick to 10.
    if args.full:
        n_range = range(3, 51)
    elif args.quick:
        n_range = range(3, 11)
    else:
        n_range = range(3, 21)

    print("=" * 60)
    print("PERSISTENT PATH HOMOLOGY ON S¹")
    print("=" * 60)

    print("\n[Standard height function]")
    results = compare(n_range, 'standard')
    print_table(results['without'], results['with'])

    plot_death_times(results['without'], results['with'],
                     args.output_dir / 'pph_death_times.png')
    plot_bar_counts(results['without'], results['with'],
                    args.output_dir / 'pph_bar_counts.png')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    mod2_no = [r for r in results['without'] if r.mod4 == 2 and r.n >= 6]
    mod2_yes = [r for r in results['with'] if r.mod4 == 2 and r.n >= 6]

    print(f"\nFor n ≡ 2 (mod 4):")
    print(f"  Without double edges: {mod2_no[0].n_bars} bars (anomaly)")
    print(f"  With double edges:    {mod2_yes[0].n_bars} bar (fixed)")

    if all(r.n_bars == 1 for r in mod2_yes):
        print("\n✓ Double edges fix the anomaly for all tested n values.")

    print(f"\nOutputs saved to: {args.output_dir}")

def plot_persistence_diagram(result: Result,
                              output: str | Path | None = None) -> plt.Figure:
    """Plot full persistence diagram for a single result."""
    fig, ax = plt.subplots(figsize=(6, 6))

    max_val = np.pi
    ax.plot([0, max_val], [0, max_val], 'k--', lw=0.8, alpha=0.4)

    for birth, death in result.barcode:
        if np.isfinite(death):
            ax.scatter(birth, death, color='#3498db', s=40, alpha=0.8, zorder=3)
        else:
            ax.scatter(birth, max_val * 1.05, color='#e74c3c',
                       marker='^', s=50, zorder=3)

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'Persistence diagram — n={result.n}, {result.height_variant}', fontweight='bold')
    ax.set_xlim(-0.1, max_val * 1.1)
    ax.set_ylim(-0.1, max_val * 1.2)
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π'])
    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π'])
    ax.grid(True, alpha=0.3)

    ax.scatter([], [], marker='o', color='#3498db', s=40, label='finite death')
    ax.scatter([], [], marker='^', color='#e74c3c', s=50, label='infinite death')
    ax.legend(fontsize=8)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


if __name__ == "__main__":
    main()