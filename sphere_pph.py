"""
Persistent Path Homology on S²
================================

Clean, consolidated implementation for analyzing PPH on sphere samples.

What PPH on S² computes here
-----------------------------
We sample n points on the 2-sphere S², assign a real-valued "height function"
f : S² → ℝ to each sample, and build a directed graph G where each edge
u → v is added when f(u) < f(v) (lower-to-higher) with weight equal to the
geometric distance between u and v.  We then feed G into the grpphati
Persistent Path Homology pipeline, which computes a barcode of H₁ bars
recording when directed loops are born and killed as we include edges in
increasing weight order.

For the round 2-sphere with height f(θ,φ) = cos(θ) (the "z_coord" variant)
PPH should recover one infinite H₁ bar corresponding to the equatorial loop,
matching the topological expectation.

Edge weight metrics
-------------------
Three metrics are available, selectable from the Streamlit UI:

- 'great_circle'   (DEFAULT)
      ω(u,v) = R · arccos(⟨û,v̂⟩)
  Geodesic arc-length on the sphere of radius R.  Natural intrinsic metric;
  weight diagram shifts linearly when R is changed.

- 'chord'
      ω(u,v) = R · ‖û − v̂‖₂ = R · √(2 − 2⟨û,v̂⟩)
  Straight-line (Euclidean) distance through the ambient ℝ³.  Faster to
  compute, topologically similar to great_circle for nearby points; also
  scales linearly with R.

- 'param_euclidean'
      ω(u,v) = √(Δθ² + Δφ_circ²)
  Flat-parameter-box distance.  Ignores the actual geometry of S²; in
  particular it does NOT scale with R and will clump distances near the
  poles.  Useful as a pedagogical contrast: changing the R slider does
  NOT move the persistence diagram under this metric.

R and the persistence diagram
------------------------------
Changing R scales great_circle and chord weights by the same factor, so
every birth/death value in the barcode shifts proportionally — the diagram
"zooms".  param_euclidean weights are R-independent, so its diagram is
immune to the R slider.  This difference is intentional and pedagogically
useful.

Usage:
    python sphere_pph.py                    # Full demo analysis
    python sphere_pph.py --quick            # Quick test
    python sphere_pph.py --n 50
    python sphere_pph.py --n 50 --height Y22_re --metric chord
    python sphere_pph.py --n 50 --sampling fibonacci
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

_CACHE_DIR = Path(".pph_cache") / "sphere"


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
    # Core (cheap, always show)
    'z_coord', 'x_coord', 'y_coord',
    'sin_theta', 'cos_theta',
    'sin_phi', 'cos_phi',
    # Tilted / combined
    'tilted_z', 'double_tilt',
    'saddle', 'quadrupole', 'dipole_x',
    # Spherical harmonic style
    'Y20', 'Y21_re', 'Y22_re', 'Y32', 'Y44',
    # Pathological / interesting
    'abs_z', 'rectified_z',
    'gaussian_cap', 'two_bumps',
    'triangle_z', 'noisy',
]

SamplingVariant = Literal[
    'fibonacci',   # golden-ratio spiral; near-uniform; DEFAULT
    'grid',        # lat-lon rings (count ≈ (4/π)n²; varies with n)
    'jittered',    # fibonacci + Gaussian jitter on (θ, φ)
    'random',      # uniform on S² via area-preserving map
    'clustered',   # two Gaussian clusters
]

WeightMetric = Literal[
    'great_circle',     # geodesic arc length; scales with R
    'chord',            # ambient Euclidean distance; scales with R
    'param_euclidean',  # flat-box (θ,φ) distance; DOES NOT scale with R
]


@dataclass
class Config:
    """Analysis configuration for S²."""
    n: int                                        # sample count (see sampling docstrings)
    height_variant: HeightVariant = 'z_coord'
    double_edges: bool = True
    R: float = 1.0                                # sphere radius
    sampling: SamplingVariant = 'fibonacci'
    weight_metric: WeightMetric = 'great_circle'
    jitter_std: float = 0.1
    random_seed: int | None = None


# =============================================================================
# GEOMETRY — SpherePoint
# =============================================================================

@dataclass(frozen=True)
class SpherePoint:
    """
    A point on S² stored in both parametric and Cartesian form.

    theta : polar angle (colatitude) in [0, π]
    phi   : azimuthal angle in [0, 2π)
    xyz   : unit-vector (R=1); multiply by R only at the point of use.
    """
    theta: float
    phi: float
    xyz: tuple[float, float, float]


def _angles_to_unit(theta: float, phi: float) -> tuple[float, float, float]:
    """Physics convention: x=sin θ cos φ, y=sin θ sin φ, z=cos θ."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return (float(x), float(y), float(z))


def _xyz_to_angles(x: float, y: float, z: float) -> tuple[float, float]:
    """Recover (θ, φ) from a unit vector.  Returns θ ∈ [0,π], φ ∈ [0,2π)."""
    theta = float(np.arccos(np.clip(z, -1.0, 1.0)))
    phi = float(np.arctan2(y, x) % (2 * np.pi))
    return theta, phi


# =============================================================================
# SAMPLING
# =============================================================================

def sample_sphere(cfg: Config) -> list[SpherePoint]:
    """
    Generate SpherePoint samples on S² according to cfg.sampling.

    fibonacci  : n points via the golden-ratio / Fibonacci spiral.
                 Produces near-uniform coverage with no pole clumping.
                 cfg.n is the exact point count.

    grid       : lat-lon rings.  Ring k has θ_k = (k+0.5)π/n for
                 k ∈ [0,n); ring k gets n_k ≈ round(2n·sin θ_k) points
                 (at least 1).  Total count is ≈(4/π)n² and varies
                 with n — it is NOT equal to cfg.n.

    jittered   : Fibonacci base + Gaussian noise on (θ,φ) with std
                 cfg.jitter_std, then renormalized onto the sphere.

    random     : cfg.n points uniform on S² via the area-preserving
                 map  z = 1−2u, φ = 2πv (u,v ~ Uniform(0,1)).  Sampling
                 (θ,φ) uniformly in the parameter box would clump at
                 the poles — this variant avoids that.

    clustered  : Two Gaussian clusters centered at the north pole
                 (θ≈0) and south pole (θ≈π), with spread cfg.jitter_std.
    """
    n = cfg.n
    rng = np.random.default_rng(cfg.random_seed)
    golden = np.pi * (3.0 - np.sqrt(5.0))  # golden-ratio angle step

    if cfg.sampling == 'fibonacci':
        pts = []
        for i in range(n):
            z = 1.0 - 2.0 * (i + 0.5) / n
            r = np.sqrt(max(0.0, 1.0 - z * z))
            phi = (golden * i) % (2 * np.pi)
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            theta = float(np.arccos(np.clip(z, -1.0, 1.0)))
            phi = float(phi)
            pts.append(SpherePoint(theta=theta, phi=phi, xyz=(float(x), float(y), float(z))))
        return pts

    elif cfg.sampling == 'grid':
        pts = []
        for k in range(n):
            theta_k = (k + 0.5) * np.pi / n
            n_k = max(1, round(2 * n * np.sin(theta_k)))
            for j in range(n_k):
                phi_j = 2 * np.pi * j / n_k
                xyz = _angles_to_unit(float(theta_k), float(phi_j))
                pts.append(SpherePoint(theta=float(theta_k), phi=float(phi_j), xyz=xyz))
        return pts

    elif cfg.sampling == 'jittered':
        base = sample_sphere(Config(
            n=n, height_variant=cfg.height_variant,
            double_edges=cfg.double_edges, R=cfg.R,
            sampling='fibonacci',
            weight_metric=cfg.weight_metric,
            jitter_std=cfg.jitter_std,
            random_seed=cfg.random_seed,
        ))
        pts = []
        for p in base:
            theta_j = p.theta + rng.normal(0.0, cfg.jitter_std)
            phi_j = p.phi + rng.normal(0.0, cfg.jitter_std)
            theta_j = float(np.clip(theta_j, 0.0, np.pi))
            phi_j = float(phi_j % (2 * np.pi))
            xyz = _angles_to_unit(theta_j, phi_j)
            pts.append(SpherePoint(theta=theta_j, phi=phi_j, xyz=xyz))
        return pts

    elif cfg.sampling == 'random':
        us = rng.uniform(0.0, 1.0, size=n)
        vs = rng.uniform(0.0, 1.0, size=n)
        zs = 1.0 - 2.0 * us
        phis = 2.0 * np.pi * vs
        rs = np.sqrt(np.clip(1.0 - zs ** 2, 0.0, None))
        xs = rs * np.cos(phis)
        ys = rs * np.sin(phis)
        pts = []
        for x, y, z, phi in zip(xs, ys, zs, phis):
            theta = float(np.arccos(np.clip(float(z), -1.0, 1.0)))
            pts.append(SpherePoint(
                theta=theta, phi=float(phi % (2 * np.pi)),
                xyz=(float(x), float(y), float(z)),
            ))
        return pts

    elif cfg.sampling == 'clustered':
        half = n // 2
        rest = n - half
        std = cfg.jitter_std
        pts = []
        # North-pole cluster: θ ≈ 0
        for _ in range(half):
            theta = abs(rng.normal(0.0, std))
            phi = rng.uniform(0.0, 2 * np.pi)
            theta = float(np.clip(theta, 0.0, np.pi))
            xyz = _angles_to_unit(theta, float(phi))
            pts.append(SpherePoint(theta=theta, phi=float(phi), xyz=xyz))
        # South-pole cluster: θ ≈ π
        for _ in range(rest):
            theta = np.pi - abs(rng.normal(0.0, std))
            phi = rng.uniform(0.0, 2 * np.pi)
            theta = float(np.clip(theta, 0.0, np.pi))
            xyz = _angles_to_unit(theta, float(phi))
            pts.append(SpherePoint(theta=theta, phi=float(phi), xyz=xyz))
        return pts

    else:
        raise ValueError(f"Unknown sampling variant: {cfg.sampling!r}")


# =============================================================================
# HEIGHT FUNCTIONS
# =============================================================================

def height(theta: float, phi: float, variant: HeightVariant = 'z_coord',
           R: float = 1.0) -> float:
    """
    Height function on S².

    Parameters
    ----------
    theta   : polar angle in [0, π]
    phi     : azimuthal angle in [0, 2π)
    variant : name of height function
    R       : sphere radius (only affects coordinate-based variants)
    """
    # Core
    if variant == 'z_coord':
        return R * np.cos(theta)
    if variant == 'x_coord':
        return R * np.sin(theta) * np.cos(phi)
    if variant == 'y_coord':
        return R * np.sin(theta) * np.sin(phi)
    if variant == 'sin_theta':
        return float(np.sin(theta))
    if variant == 'cos_theta':
        return float(np.cos(theta))
    if variant == 'sin_phi':
        return float(np.sin(phi))
    if variant == 'cos_phi':
        return float(np.cos(phi))

    # Tilted / combined
    if variant == 'tilted_z':
        return float(np.cos(theta) + 0.5 * np.sin(theta) * np.cos(phi))
    if variant == 'double_tilt':
        return float(np.cos(theta)
                     + 0.3 * np.sin(theta) * np.cos(phi)
                     + 0.3 * np.sin(theta) * np.sin(phi))
    if variant == 'saddle':
        return float(np.sin(theta) ** 2 * np.cos(2 * phi))
    if variant == 'quadrupole':
        return float(3 * np.cos(theta) ** 2 - 1)
    if variant == 'dipole_x':
        return float(np.sin(theta) * np.cos(phi))

    # Spherical harmonic style
    if variant == 'Y20':
        return float(3 * np.cos(theta) ** 2 - 1)
    if variant == 'Y21_re':
        return float(np.sin(theta) * np.cos(theta) * np.cos(phi))
    if variant == 'Y22_re':
        return float(np.sin(theta) ** 2 * np.cos(2 * phi))
    if variant == 'Y32':
        return float(np.sin(theta) ** 2 * np.cos(theta) * np.cos(2 * phi))
    if variant == 'Y44':
        return float(np.sin(theta) ** 4 * np.cos(4 * phi))

    # Pathological / interesting
    if variant == 'abs_z':
        return float(abs(np.cos(theta)))
    if variant == 'rectified_z':
        return float(max(0.0, np.cos(theta)))
    if variant == 'gaussian_cap':
        return float(np.exp(-5.0 * (1.0 - np.cos(theta))))
    if variant == 'two_bumps':
        return float(np.exp(-5.0 * (1.0 - np.cos(theta)))
                     + np.exp(-5.0 * (1.0 + np.cos(theta))))
    if variant == 'triangle_z':
        return float(1.0 - abs(2.0 * theta / np.pi - 1.0))
    if variant == 'noisy':
        return float(np.cos(theta) + 0.1 * np.sin(5 * theta) * np.cos(3 * phi))

    raise ValueError(f"Unknown height variant: {variant!r}")


# =============================================================================
# EDGE WEIGHT METRIC
# =============================================================================

def sphere_dist(p1: SpherePoint, p2: SpherePoint,
                R: float, metric: WeightMetric) -> float:
    """
    Distance between two points on S² of radius R.

    great_circle : R · arccos(⟨û,v̂⟩) — geodesic; scales with R.
    chord        : R · ‖û − v̂‖₂    — ambient chord; scales with R.
    param_euclidean : √(Δθ² + Δφ_circ²) — flat box; does NOT depend on R.
    """
    if metric == 'great_circle':
        dot = (p1.xyz[0] * p2.xyz[0]
               + p1.xyz[1] * p2.xyz[1]
               + p1.xyz[2] * p2.xyz[2])
        return float(R * np.arccos(np.clip(dot, -1.0, 1.0)))

    elif metric == 'chord':
        dx = p1.xyz[0] - p2.xyz[0]
        dy = p1.xyz[1] - p2.xyz[1]
        dz = p1.xyz[2] - p2.xyz[2]
        return float(R * np.sqrt(dx * dx + dy * dy + dz * dz))

    elif metric == 'param_euclidean':
        dtheta = p1.theta - p2.theta
        dphi = abs(p1.phi - p2.phi)
        dphi = min(dphi, 2 * np.pi - dphi)   # wrap in φ
        return float(np.sqrt(dtheta ** 2 + dphi ** 2))

    else:
        raise ValueError(f"Unknown weight metric: {metric!r}")


# =============================================================================
# FILTRATION  (mirrors torus_pph.py exactly)
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


def build_graph(cfg: Config) -> tuple[nx.DiGraph, list[float], list[SpherePoint]]:
    """
    Build a directed graph on all sample points of S².

    Iterates over every unordered pair {u, v} and adds directed edges via
    _add_directed_edge:

    * |f(u) − f(v)| < 1e-10 (tie): add u→v AND v→u if cfg.double_edges,
      otherwise skip the pair entirely.
    * f(u) < f(v)  →  add edge u→v with weight sphere_dist(u, v).
    * f(u) > f(v)  →  add edge v→u with weight sphere_dist(v, u).

    Yields at most N(N−1)/2 unordered pairs, up to N(N−1) directed edges.

    Returns: (graph, heights, points)
    """
    points = sample_sphere(cfg)
    N = len(points)
    heights = [height(p.theta, p.phi, cfg.height_variant, cfg.R) for p in points]

    G = nx.DiGraph()
    for i, p in enumerate(points):
        G.add_node(i, height=heights[i], theta=p.theta, phi=p.phi,
                   x=p.xyz[0], y=p.xyz[1], z=p.xyz[2])

    for u in range(N):
        for v in range(u + 1, N):
            w = sphere_dist(points[u], points[v], cfg.R, cfg.weight_metric)
            _add_directed_edge(G, u, v, w, heights[u], heights[v], cfg.double_edges)

    return G, heights, points


# =============================================================================
# PPH COMPUTATION  (mirrors torus_pph.py exactly)
# =============================================================================

def compute_pph(G: nx.DiGraph) -> list[tuple[float, float]]:
    """Compute PPH barcode for a directed graph G."""
    pipeline = make_standard_pipeline(
        EdgeWeightFiltration,
        RegularPathHomology,
        backend=LoPHATBackend(with_reps=False),
        optimisation_strat=component_appendage_empty,
    )
    return pipeline(G).barcode


@dataclass
class Result:
    """Analysis result for a single sphere configuration."""
    n: int
    height_variant: HeightVariant
    double_edges: bool
    weight_metric: WeightMetric
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
        return (f"Result(n={self.n}, bars={self.n_bars}, "
                f"max_death={self.max_death:.4f}, metric={self.weight_metric!r})")


def analyze(cfg: Config) -> Result:
    """Run full analysis for given configuration, using a disk cache."""
    key = _cfg_key(cfg)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    G, _, _ = build_graph(cfg)
    barcode = compute_pph(G)
    result = Result(
        n=cfg.n,
        height_variant=cfg.height_variant,
        double_edges=cfg.double_edges,
        weight_metric=cfg.weight_metric,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        barcode=barcode,
    )
    _save_cached(key, result)
    return result


def analyze_simple(n: int,
                   double_edges: bool = True,
                   height_variant: HeightVariant = 'z_coord',
                   sampling: SamplingVariant = 'fibonacci',
                   weight_metric: WeightMetric = 'great_circle',
                   R: float = 1.0) -> Result:
    """Convenience wrapper around analyze()."""
    return analyze(Config(
        n=n,
        height_variant=height_variant,
        double_edges=double_edges,
        sampling=sampling,
        weight_metric=weight_metric,
        R=R,
    ))


# =============================================================================
# VISUALIZATION
# =============================================================================

MOD4_COLORS = {0: '#27ae60', 1: '#3498db', 2: '#e74c3c', 3: '#9b59b6'}


def plot_persistence_diagram(
    result: Result,
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Plot the persistence diagram for a sphere PPH Result.

    Finite bars are blue dots; infinite bars are red upward triangles
    pinned above the diagonal.  A dashed y=x reference line is drawn.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    finite = [(b, d) for b, d in result.barcode if np.isfinite(d)]
    infinite = [b for b, d in result.barcode if not np.isfinite(d)]

    all_vals = ([b for b, d in finite] + [d for b, d in finite] + infinite)
    if not all_vals:
        all_vals = [0.0, 1.0]
    lo = min(all_vals)
    hi = max(all_vals)
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
        f'Persistence diagram — S²\n'
        f'n={result.n}, height={result.height_variant!r}, '
        f'metric={result.weight_metric!r}, {de_str}',
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


def plot_sphere_graph_flat(
    cfg: Config,
    arc_min: float = 0.0,
    arc_max: float = float('inf'),
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the directed sphere graph in the flat (θ, φ) parameter domain.

    θ on x-axis (0 … π) — NOT periodic.
    φ on y-axis (0 … 2π) — periodic; wrap-around edges drawn as thin lines.

    Nodes are coloured by height (RdBu_r), edges by weight (plasma).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    G, heights, points = build_graph(cfg)

    edges_to_draw = [
        (u, v, G.edges[u, v]['weight'])
        for u, v in G.edges()
        if arc_min <= G.edges[u, v]['weight'] <= arc_max
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    if edges_to_draw:
        weights = [w for _, _, w in edges_to_draw]
        w_min, w_max = min(weights), max(weights)
        e_norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
    else:
        e_norm = mcolors.Normalize(vmin=0, vmax=1)
    e_cmap = cm.plasma

    for u, v, w in edges_to_draw:
        x0, y0 = points[u].theta / np.pi, points[u].phi / np.pi
        x1, y1 = points[v].theta / np.pi, points[v].phi / np.pi
        col = e_cmap(e_norm(w))
        # φ wraps, θ does not
        dphi = abs(y1 - y0)
        if dphi > 1.0:  # wrap in φ
            ax.plot([x0, x1], [y0, y1], color=col, lw=0.5, alpha=0.35, zorder=1)
        else:
            length = np.hypot(x1 - x0, y1 - y0)
            rad = float(np.clip(length * 0.3, 0.0, 0.4))
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

    h_arr = np.array(heights)
    h_min, h_max = float(h_arr.min()), float(h_arr.max())
    if h_max - h_min < 1e-12:
        n_norm = mcolors.Normalize(vmin=h_min - 1e-6, vmax=h_max + 1e-6)
    else:
        vc = float(np.median(h_arr))
        vc = max(h_min + 1e-9, min(vc, h_max - 1e-9))
        n_norm = mcolors.TwoSlopeNorm(vmin=h_min, vcenter=vc, vmax=h_max)

    sc = ax.scatter(
        [p.theta / np.pi for p in points],
        [p.phi / np.pi for p in points],
        c=heights, cmap='RdBu_r', norm=n_norm,
        s=50, zorder=4, edgecolors='white', linewidths=0.4,
    )

    plt.colorbar(sc, ax=ax, label='Height', fraction=0.046, pad=0.04)
    if edges_to_draw:
        sm = cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Edge weight', fraction=0.046, pad=0.08)

    ax.set_xlabel('θ / π  (polar)')
    ax.set_ylabel('φ / π  (azimuth)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 2.05)
    ax.set_title(
        f'Sphere graph — flat (θ,φ) view\n'
        f'n={cfg.n}, height={cfg.height_variant!r}, sampling={cfg.sampling!r}, '
        f'metric={cfg.weight_metric!r}\n'
        f'arc ∈ [{arc_min:.3f}, {"∞" if not np.isfinite(arc_max) else f"{arc_max:.3f}"}]',
        fontweight='bold',
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")

    return fig


def plot_sphere_3d(
    cfg: Config,
    arc_min: float = 0.0,
    arc_max: float = float('inf'),
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the directed sphere graph in 3D over a faint sphere surface.

    Edges are straight chord segments in ℝ³ (not geodesics).
    Nodes are coloured by height (RdBu_r); edges by weight (plasma).
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    R = cfg.R
    G, heights, points = build_graph(cfg)

    # sphere surface mesh
    u_arr = np.linspace(0, 2 * np.pi, 80)
    v_arr = np.linspace(0, np.pi, 40)
    U, V = np.meshgrid(u_arr, v_arr)
    Xs = R * np.sin(V) * np.cos(U)
    Ys = R * np.sin(V) * np.sin(U)
    Zs = R * np.cos(V)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=25, azim=45)

    ax.plot_surface(Xs, Ys, Zs, color='lightgrey', alpha=0.08,
                    linewidth=0, antialiased=True, zorder=0)

    node_xyz = [(p.xyz[0] * R, p.xyz[1] * R, p.xyz[2] * R) for p in points]

    edges_to_draw = [
        (u, v, G.edges[u, v]['weight'])
        for u, v in G.edges()
        if arc_min <= G.edges[u, v]['weight'] <= arc_max
    ]

    if edges_to_draw:
        weights = [w for _, _, w in edges_to_draw]
        w_min, w_max = min(weights), max(weights)
        e_norm = mcolors.Normalize(vmin=w_min, vmax=w_max)
    else:
        e_norm = mcolors.Normalize(vmin=0, vmax=1)
    e_cmap = cm.plasma

    for u_idx, v_idx, w in edges_to_draw:
        x0, y0, z0 = node_xyz[u_idx]
        x1, y1, z1 = node_xyz[v_idx]
        col = e_cmap(e_norm(w))
        ax.plot([x0, x1], [y0, y1], [z0, z1],
                color=col, lw=0.7, alpha=0.55, zorder=2)

    h_arr = np.array(heights)
    h_norm = mcolors.Normalize(vmin=h_arr.min(), vmax=h_arr.max())
    n_cmap = cm.RdBu_r
    node_colors = [n_cmap(h_norm(h)) for h in heights]

    xs, ys, zs = zip(*node_xyz)
    ax.scatter(xs, ys, zs, c=node_colors, s=60, zorder=6,
               depthshade=False, edgecolors='white', linewidths=0.5)

    sm_nodes = cm.ScalarMappable(cmap=n_cmap, norm=h_norm)
    sm_nodes.set_array([])
    cb_n = fig.colorbar(sm_nodes, ax=ax, shrink=0.5, pad=0.0, location='left')
    cb_n.set_label('Height', fontsize=9)

    if edges_to_draw:
        sm_edges = cm.ScalarMappable(cmap=e_cmap, norm=e_norm)
        sm_edges.set_array([])
        cb_e = fig.colorbar(sm_edges, ax=ax, shrink=0.5, pad=0.02, location='right')
        cb_e.set_label('Edge weight', fontsize=9)

    ax.set_title(
        f'Sphere graph — {cfg.height_variant}, {cfg.sampling}, '
        f'double_edges={cfg.double_edges}, metric={cfg.weight_metric!r}',
        fontsize=10,
    )
    ax.set_axis_off()
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved: {output}")

    return fig


# =============================================================================
# MAIN / CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPH analysis on S²")
    parser.add_argument('--n', type=int, default=None,
                        help='Sample count for a single analysis')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test (n=15)')
    parser.add_argument('--height', default='z_coord',
                        help='Height variant (e.g. z_coord, Y22_re, tilted_z)')
    parser.add_argument('--metric', default='great_circle',
                        choices=['great_circle', 'chord', 'param_euclidean'],
                        help='Edge weight metric')
    parser.add_argument('--sampling', default='fibonacci',
                        choices=['fibonacci', 'grid', 'jittered', 'random', 'clustered'],
                        help='Sampling strategy')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                        help='Output directory for plots')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        print("=" * 55)
        print("QUICK TEST — Persistent Path Homology on S²")
        print("=" * 55)
        configs = [
            Config(n=15, height_variant='z_coord',   sampling='fibonacci'),
            Config(n=15, height_variant='tilted_z',  sampling='fibonacci'),
            Config(n=15, height_variant='Y22_re',    sampling='fibonacci',
                   weight_metric='chord'),
        ]
        for cfg in configs:
            G, heights, pts = build_graph(cfg)
            barcode = compute_pph(G)
            print(f"\n  n={cfg.n}  height={cfg.height_variant!r}  "
                  f"metric={cfg.weight_metric!r}  sampling={cfg.sampling!r}")
            print(f"  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  "
                  f"bars={len(barcode)}")
            for b, d in barcode:
                d_str = f"{d:.4f}" if np.isfinite(d) else "∞"
                print(f"    [{b:.4f},  {d_str}]")
        print("\nQuick test complete.")
        return

    if args.n:
        cfg = Config(
            n=args.n,
            height_variant=args.height,    # type: ignore[arg-type]
            sampling=args.sampling,        # type: ignore[arg-type]
            weight_metric=args.metric,     # type: ignore[arg-type]
        )
        G, heights, pts = build_graph(cfg)
        barcode = compute_pph(G)
        result = Result(
            n=cfg.n,
            height_variant=cfg.height_variant,
            double_edges=cfg.double_edges,
            weight_metric=cfg.weight_metric,
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            barcode=barcode,
        )
        print(f"n={cfg.n}, nodes={result.n_nodes}, edges={result.n_edges}, "
              f"bars={result.n_bars}")
        for b, d in barcode:
            d_str = f"{d:.4f}" if np.isfinite(d) else "∞"
            print(f"  [{b:.4f},  {d_str}]")

        stem = f"sphere_{args.height}_{args.metric}"
        plot_persistence_diagram(result, output=args.output_dir / f'{stem}_pd.png')
        plot_sphere_graph_flat(cfg,      output=args.output_dir / f'{stem}_flat.png')
        plot_sphere_3d(cfg,              output=args.output_dir / f'{stem}_3d.png')
        return

    # Default: run a few representative configurations
    print("=" * 55)
    print("PERSISTENT PATH HOMOLOGY ON S²")
    print("=" * 55)

    demo_cfgs = [
        Config(n=30, height_variant='z_coord',   sampling='fibonacci'),
        Config(n=30, height_variant='tilted_z',  sampling='fibonacci'),
        Config(n=30, height_variant='Y22_re',    sampling='fibonacci',
               weight_metric='chord'),
        Config(n=30, height_variant='gaussian_cap', sampling='random',
               random_seed=42),
        Config(n=30, height_variant='two_bumps',  sampling='fibonacci',
               weight_metric='param_euclidean'),
    ]

    for cfg in demo_cfgs:
        G, heights, pts = build_graph(cfg)
        barcode = compute_pph(G)
        result = Result(
            n=cfg.n,
            height_variant=cfg.height_variant,
            double_edges=cfg.double_edges,
            weight_metric=cfg.weight_metric,
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            barcode=barcode,
        )
        print(f"\n  height={cfg.height_variant!r}  metric={cfg.weight_metric!r}"
              f"  sampling={cfg.sampling!r}")
        print(f"  nodes={result.n_nodes}  edges={result.n_edges}  bars={result.n_bars}")
        for b, d in barcode:
            d_str = f"{d:.4f}" if np.isfinite(d) else "∞"
            print(f"    [{b:.4f},  {d_str}]")

        stem = f"sphere_{cfg.height_variant}_{cfg.weight_metric}"
        plot_persistence_diagram(result, output=args.output_dir / f'{stem}_pd.png')
        plot_sphere_graph_flat(cfg,      output=args.output_dir / f'{stem}_flat.png')
        plot_sphere_3d(cfg,              output=args.output_dir / f'{stem}_3d.png')

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
