"""
Persistent Path Homology on Wedge Sums and Gluings of Circles
==============================================================

Analyzes PPH on a variety of topological spaces built by gluing circles:
wedge sums, theta graphs, lollipops, eyeglasses, chains, necklaces, etc.

Usage:
    python wedge_pph.py                    # Run full analysis + sanity checks
    python wedge_pph.py --quick            # Skip sweep table
    python wedge_pph.py --output-dir out/  # Save plots to out/
"""

from __future__ import annotations

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

os.environ.setdefault("RAYON_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Re-use everything stable from circle_pph_2
# ---------------------------------------------------------------------------
from circle_pph_2 import (
    HomologyVariant,
    SamplingVariant,
    EdgeWeightFiltration,
    MOD4_COLORS,
    compute_homology,
    Config as _CircleConfig,
    sample_angles as _sample_angles_circle,
)

# ---------------------------------------------------------------------------
# HEIGHT FUNCTIONS — wedge-specific, defined on angle ∈ [0, 2π)
# ---------------------------------------------------------------------------

HeightVariant = Literal[
    # --- local (per-circle angle) ---
    'standard',          # sin(θ) — one peak, one trough
    'cos',               # cos(θ) — peak at 0, trough at π
    'sin2',              # sin(2θ) — two peaks, two troughs
    'cos2',              # cos(2θ) — two peaks, two troughs, phase-shifted
    'abs_sin',           # |sin(θ)| — always non-negative, two bumps
    'triangle',          # piecewise-linear triangle wave on local angle
    'sawtooth',          # linear drop from 1 to -1 on local angle
    'square',            # +1 on [0,π), -1 on [π, 2π) of local angle
    # --- local R² embedding-based ---
    'x_coord',           # x = cos(θ) — projection onto x-axis
    'y_coord',           # y = sin(θ) — same as standard
    'radial_dist',       # negative distance from glue point (1,0)
    'angle_from_top',    # how close to top of local circle
    'dist_from_right',   # angular distance from θ=0 (glue point)
    # --- local modulated ---
    'dip',               # sin(θ) with a dip near the peak
    'double_max',        # two local maxima
    'plateau',           # sin with flat region near π
    'squeeze',           # compressed on [0,π), normal on [π,2π)
    'shallow',           # amplitude-modulated: one side taller
    'damped',            # exp(-0.5θ)·sin(4θ) — decaying oscillation
    # --- global (uses 2D position, continuous across gluing) ---
    'global_x',          # x-coordinate of node in R² — rises left to right
    'global_y',          # y-coordinate — rises bottom to top
    'global_r',          # distance from origin (0,0) — glue point is lowest for wedge
    'global_angle',      # polar angle of 2D position — sawtooth around origin
    'global_sawtooth_x', # continuous sawtooth along x-axis across all petals
    'global_sin_x',      # sin(x) — smooth wave along x-axis
    'global_sin_y',      # sin(y) — smooth wave along y-axis
    'global_sin_r',      # sin(2πr) — concentric rings
    'global_diagonal',   # (x + y)/√2 — diagonal ramp
    'global_saddle',     # x² - y² — saddle point at origin
    'global_ripple',     # sin(x)·cos(y) — 2D ripple pattern
    'global_dist_max',   # distance from farthest node — peak at periphery
    # --- constant / trivial ---
    'constant',          # always 0 — all edges are ties
]


# Marker set for variants that use 2D position rather than local angle
_GLOBAL_VARIANTS: set[str] = {
    'global_x', 'global_y', 'global_r', 'global_angle',
    'global_sawtooth_x', 'global_sin_x', 'global_sin_y', 'global_sin_r',
    'global_diagonal', 'global_saddle', 'global_ripple', 'global_dist_max',
}


def global_height(x: float, y: float, variant: HeightVariant,
                  scale: float = 1.0) -> float:
    """
    Height function based on 2D position (x, y) in the R² embedding.
    These are continuous across gluing points — two nodes close in arc
    distance will have similar heights regardless of which circle they
    live on.  `scale` is a characteristic length (e.g. mean radius)
    used to normalise distances.
    """
    if variant == 'global_x':
        return x / max(scale, 1e-9)

    if variant == 'global_y':
        return y / max(scale, 1e-9)

    if variant == 'global_r':
        return float(np.sqrt(x**2 + y**2)) / max(scale, 1e-9)

    if variant == 'global_angle':
        # polar angle in [-π, π], remapped to [-1, 1]
        return float(np.arctan2(y, x)) / np.pi

    if variant == 'global_sawtooth_x':
        # sawtooth along x: period = 2*scale, continuous, range [-1,1]
        p = 2 * max(scale, 1e-9)
        t = (x % p) / p   # in [0, 1)
        return 2 * t - 1  # in [-1, 1)

    if variant == 'global_sin_x':
        # full sine wave with period 2*scale along x
        return float(np.sin(np.pi * x / max(scale, 1e-9)))

    if variant == 'global_sin_y':
        return float(np.sin(np.pi * y / max(scale, 1e-9)))

    if variant == 'global_sin_r':
        r = np.sqrt(x**2 + y**2)
        return float(np.sin(2 * np.pi * r / max(scale, 1e-9)))

    if variant == 'global_diagonal':
        return (x + y) / (np.sqrt(2) * max(scale, 1e-9))

    if variant == 'global_saddle':
        s = max(scale, 1e-9) ** 2
        return float((x**2 - y**2) / s)

    if variant == 'global_ripple':
        s = max(scale, 1e-9)
        return float(np.sin(np.pi * x / s) * np.cos(np.pi * y / s))

    if variant == 'global_dist_max':
        # placeholder — actual max distance injected after graph is built
        return float(np.sqrt(x**2 + y**2)) / max(scale, 1e-9)

    raise ValueError(f"Not a global variant: {variant!r}")


def height(angle: float, variant: HeightVariant,
           dip_depth: float = 1.0, epsilon0: float = 0.05) -> float:
    """Evaluate the height function at *angle* ∈ [0, 2π)."""
    t = angle % (2 * np.pi)

    if variant == 'standard':
        return float(np.sin(t))
    if variant == 'cos':
        return float(np.cos(t))
    if variant == 'sin2':
        return float(np.sin(2 * t))
    if variant == 'cos2':
        return float(np.cos(2 * t))
    if variant == 'abs_sin':
        return float(abs(np.sin(t)))
    if variant == 'triangle':
        # rises 0→1 on [0,π/2], falls 1→-1 on [π/2,3π/2], rises -1→0 on [3π/2,2π]
        if t < np.pi / 2:
            return t / (np.pi / 2)
        elif t < 3 * np.pi / 2:
            return 1.0 - 2 * (t - np.pi / 2) / np.pi
        else:
            return -1.0 + (t - 3 * np.pi / 2) / (np.pi / 2)
    if variant == 'sawtooth':
        return 1.0 - t / np.pi   # drops from 1 at 0 to -1 at 2π
    if variant == 'square':
        return 1.0 if t < np.pi else -1.0

    # --- R² embedding-based ---
    if variant == 'x_coord':
        return float(np.cos(t))
    if variant == 'y_coord':
        return float(np.sin(t))
    if variant == 'radial_dist':
        # Unit circle: all points equidistant — use distance from centroid (0,0)
        # More interesting: distance from a fixed reference point (1, 0)
        x, y = np.cos(t), np.sin(t)
        return -float(np.sqrt((x - 1)**2 + y**2))  # negative so glue point is min
    if variant == 'angle_from_top':
        # Height = how close to the top of the circle (θ=π/2)
        return -float(abs(t - np.pi / 2) if t <= np.pi else abs(t - 3 * np.pi / 2))
    if variant == 'dist_from_right':
        # Distance from θ=0 (the glue point for wedge topologies)
        d = min(t, 2 * np.pi - t)
        return -d / np.pi   # 0 at glue point, -1 at antipodal point

    # --- modulated ---
    if variant == 'dip':
        base = np.sin(t)
        if np.pi / 4 <= t <= 3 * np.pi / 4:
            base -= dip_depth * np.sin(2 * (t - np.pi / 4))
        return float(base)
    if variant == 'double_max':
        return float(np.sin(t) ** 2 * np.cos(t) + 0.5 * np.sin(2 * t))
    if variant == 'plateau':
        if 0.75 * np.pi < t < 1.25 * np.pi:
            return 0.0
        return float(np.sin(t))
    if variant == 'squeeze':
        if t < np.pi:
            return float(np.sin(2 * t - np.pi / 2))
        return float(np.sin(t))
    if variant == 'shallow':
        return float(np.sin(t) * (0.75 + 0.25 * np.cos(t)))
    if variant == 'damped':
        return float(np.exp(-0.5 * t) * np.sin(4 * t))

    if variant == 'constant':
        return 0.0

    raise ValueError(f"Unknown height variant: {variant!r}")

# ---------------------------------------------------------------------------
# Topology variants
# ---------------------------------------------------------------------------
TopologyVariant = Literal[
    'wedge2',
    'wedge3',
    'wedge_k',
    'theta',
    'lollipop',
    'eyeglasses',
    'figure8_asymmetric',
    'chain',
    'necklace',
    'necklace_full',   # like necklace but samples from full circles, not just outer arc
]

DistanceVariant = Literal[
    'within_only',    # original behaviour: no cross-component edges
    'euclidean',      # straight-line distance between 2D positions
    'geodesic',       # arc to nearest glue node + arc from glue node on other circle
    'arc_sum',        # sum of arc lengths to each node's nearest glue point (symmetric geodesic)
    'global_arc',     # treat all nodes as on one big circle by angle; distance = angle diff
]

# ---------------------------------------------------------------------------
# RESULT CACHE
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(".homology_cache") / "wedge"


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


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Analysis configuration for wedge/gluing topologies."""
    topology: TopologyVariant
    k: int = 2
    radii: list[float] | None = None
    bridge_length: int = 0
    double_edges: bool = True
    height_variant: HeightVariant = 'standard'
    height_per_component: list[HeightVariant] | None = None
    component_phase: list[float] | None = None
    dip_depth: float = 1.0
    epsilon0: float = 0.05
    homology: HomologyVariant = 'path'
    distance: DistanceVariant = 'within_only'  # cross-component edge distance function
    # --- sampling ---
    total_n: int | None = None
    n_per_circle: int | list[int] | None = None
    allocation: Literal['uniform', 'random', 'proportional'] | None = None
    sampling: SamplingVariant = 'uniform'
    sampling_per_component: list[SamplingVariant] | None = None
    jitter_std: float = 0.3
    cluster_std: float = 0.3
    beta_a: float = 0.5
    beta_b: float = 0.5
    random_seed: int | None = None


# ---------------------------------------------------------------------------
# RESULT
# ---------------------------------------------------------------------------

@dataclass
class Result:
    """Analysis result for a wedge/gluing topology."""
    topology: TopologyVariant
    n_total: int
    n_components: int
    barcode: list[tuple[float, float]]
    homology: HomologyVariant = 'path'

    @property
    def n_bars(self) -> int:
        """Number of H₁ bars in the barcode."""
        return len(self.barcode)

    @property
    def max_death(self) -> float:
        """Maximum finite death time in the barcode."""
        if not self.barcode:
            return 0.0
        finite = [b[1] for b in self.barcode if np.isfinite(b[1])]
        return max(finite) if finite else 0.0

    def __repr__(self) -> str:
        return (f"Result(topology={self.topology!r}, n={self.n_total}, "
                f"homology={self.homology!r}, bars={self.n_bars}, "
                f"max_death={self.max_death/np.pi:.3f}π)")


# ---------------------------------------------------------------------------
# SAMPLING HELPERS
# ---------------------------------------------------------------------------

def _n_components(cfg: Config) -> int:
    """Return the number of circle components for a given topology."""
    topo = cfg.topology
    if topo in ('wedge2', 'eyeglasses', 'figure8_asymmetric', 'theta'):
        return 2
    if topo == 'wedge3':
        return 3
    if topo in ('wedge_k', 'chain', 'necklace', 'necklace_full'):
        return cfg.k
    if topo == 'lollipop':
        return 1
    raise ValueError(f"Unknown topology: {topo!r}")


def _component_radii(cfg: Config, n_comp: int) -> list[float]:
    """Return per-component radii, defaulting to 1.0."""
    if cfg.radii is not None:
        if len(cfg.radii) != n_comp:
            raise ValueError(
                f"radii has {len(cfg.radii)} entries but topology has {n_comp} components"
            )
        return list(cfg.radii)
    return [1.0] * n_comp


def _allocate_samples(cfg: Config) -> list[int]:
    """Determine how many sample points (including gluing node) go on each circle.

    Returns a list of per-component counts, each >= 2.
    """
    n_comp = _n_components(cfg)
    radii = _component_radii(cfg, n_comp)

    # Mode 1: explicit n_per_circle
    if cfg.n_per_circle is not None:
        if isinstance(cfg.n_per_circle, int):
            return [max(2, cfg.n_per_circle)] * n_comp
        counts = list(cfg.n_per_circle)
        if len(counts) != n_comp:
            raise ValueError(
                f"n_per_circle has {len(counts)} entries but topology has {n_comp} components"
            )
        return [max(2, c) for c in counts]

    # Mode 2: total_n with allocation strategy
    if cfg.total_n is not None:
        total = cfg.total_n
        alloc = cfg.allocation or 'uniform'
        rng = np.random.default_rng(cfg.random_seed)

        if alloc == 'uniform':
            base = total // n_comp
            rem = total % n_comp
            counts = [base + (1 if i < rem else 0) for i in range(n_comp)]

        elif alloc == 'random':
            # Dirichlet partition, minimum 2 each
            alpha = np.ones(n_comp)
            weights = rng.dirichlet(alpha)
            raw = np.round(weights * (total - 2 * n_comp)).astype(int) + 2
            # Adjust to hit exactly total
            diff = total - int(raw.sum())
            for i in range(abs(diff)):
                if diff > 0:
                    raw[i % n_comp] += 1
                else:
                    if raw[i % n_comp] > 2:
                        raw[i % n_comp] -= 1
            counts = raw.tolist()

        elif alloc == 'proportional':
            # Weight by circumference (2π * r)
            circumferences = [2 * np.pi * r for r in radii]
            total_circ = sum(circumferences)
            weights = [c / total_circ for c in circumferences]
            raw = [max(2, round(w * total)) for w in weights]
            # Adjust to hit total
            diff = total - sum(raw)
            i = 0
            while diff != 0:
                idx = i % n_comp
                if diff > 0:
                    raw[idx] += 1
                    diff -= 1
                elif raw[idx] > 2:
                    raw[idx] -= 1
                    diff += 1
                i += 1
            counts = raw

        else:
            raise ValueError(f"Unknown allocation: {alloc!r}")

        return [max(2, c) for c in counts]

    # Mode 3: default — 8 per circle
    return [8] * n_comp


def _sample_circle_angles(
    n: int,
    sampling: SamplingVariant,
    phase: float,
    rng_seed: int | None,
    jitter_std: float,
    cluster_std: float,
    beta_a: float,
    beta_b: float,
) -> list[float]:
    """Sample n-1 interior angles in (0, 2π) for one circle, shifted by phase.

    The gluing point at angle `phase` (≡ 0 in local coords) is NOT included
    here — it is added by the graph builder as a shared node.
    Returns sorted angles in [0, 2π), all != phase % (2π).
    """
    n_interior = max(1, n - 1)
    tmp = _CircleConfig(
        n=n_interior,
        sampling=sampling,
        random_seed=rng_seed,
        jitter_std=jitter_std,
        cluster_std=cluster_std,
        beta_a=beta_a,
        beta_b=beta_b,
    )
    raw = _sample_angles_circle(tmp)

    glue_norm = phase % (2 * np.pi)
    # Shift raw angles so 0.0 maps to phase, and avoid exact collision
    shifted = []
    for a in raw:
        # Rotate into local frame starting just after the gluing angle
        # Add a small open-interval epsilon so we stay in (0, 2π) open
        a_local = (a + 1e-9) % (2 * np.pi)
        a_global = (glue_norm + a_local) % (2 * np.pi)
        # Avoid landing exactly on gluing angle
        if abs(a_global - glue_norm) < 1e-9:
            a_global = (a_global + 1e-9) % (2 * np.pi)
        shifted.append(a_global)

    return sorted(shifted)


# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION HELPERS
# ---------------------------------------------------------------------------

def _height_for(angle: float, variant: HeightVariant, dip_depth: float,
                epsilon0: float) -> float:
    """Evaluate the height function at *angle* with given parameters.
    For global variants (position-based), returns a placeholder value of 0.0
    — the real height is computed after all node positions are set in build_graph.
    """
    if variant in _GLOBAL_VARIANTS:
        return 0.0   # placeholder; overwritten by global recompute pass
    return height(angle, variant, dip_depth, epsilon0)


def _add_directed_edge(G: nx.DiGraph, u: int, v: int, w: float,
                       double_edges: bool) -> None:
    """Add a directed edge u→v (or both directions for ties) with weight w."""
    hu = G.nodes[u]['height']
    hv = G.nodes[v]['height']
    if abs(hu - hv) < 1e-10:
        if double_edges:
            G.add_edge(u, v, weight=w)
            G.add_edge(v, u, weight=w)
    elif hu < hv:
        G.add_edge(u, v, weight=w)
    else:
        G.add_edge(v, u, weight=w)


# ---------------------------------------------------------------------------
# TOPOLOGY-SPECIFIC GRAPH BUILDERS
# ---------------------------------------------------------------------------


def build_graph(cfg: Config) -> tuple[nx.DiGraph, dict[int, tuple[float, float]]]:
    """Build the full directed graph for the configured topology.

    Returns (G, pos) where pos maps node_id → (x, y) for visualization.
    """
    topo = cfg.topology
    n_comp = _n_components(cfg)
    counts = _allocate_samples(cfg)
    radii = _component_radii(cfg, n_comp)
    phases = cfg.component_phase or [0.0] * n_comp
    h_variants: list[HeightVariant] = (
        cfg.height_per_component
        if cfg.height_per_component is not None
        else [cfg.height_variant] * n_comp
    )
    samplings: list[SamplingVariant] = (
        cfg.sampling_per_component
        if cfg.sampling_per_component is not None
        else [cfg.sampling] * n_comp
    )

    rng = np.random.default_rng(cfg.random_seed)

    def _seed(i: int) -> int | None:
        return None if cfg.random_seed is None else int(rng.integers(0, 2**31))

    G = nx.DiGraph()
    pos: dict[int, tuple[float, float]] = {}
    next_nid = 0  # rolling node-id counter

    # ------------------------------------------------------------------
    # Shared helper: build one circle component with a pre-existing glue node
    # ------------------------------------------------------------------

    def _build_circle_component(
        comp_idx: int,
        glue_nid: int,
        glue_ang: float,
        center_xy: tuple[float, float],
    ) -> int:
        """Populate G with one circle component; return next free node id."""
        nonlocal next_nid
        n_i = counts[comp_idx]
        r_i = radii[comp_idx]
        hv_i = h_variants[comp_idx]
        samp_i = samplings[comp_idx]
        seed_i = _seed(comp_idx)
        ph_i = phases[comp_idx] if comp_idx < len(phases) else 0.0

        interior = _sample_circle_angles(
            n_i, samp_i, glue_ang + ph_i, seed_i,
            cfg.jitter_std, cfg.cluster_std, cfg.beta_a, cfg.beta_b,
        )
        cx, cy = center_xy

        # Assign node ids for interior nodes.
        # Layout: circle centered at (cx, cy) = r_i*(cos ca, sin ca); the
        # angle-0 (glue) point maps to the origin via the shape_pph formula:
        #   pos = (cx + r_i*cos(ang + π + ca), cy + r_i*sin(ang + π + ca))
        # At ang=0: pos = (cx - r_i*cos(ca), cy - r_i*sin(ca)) = (0, 0) ✓
        ca = np.arctan2(cy, cx) if (cx != 0.0 or cy != 0.0) else 0.0
        interior_nids: list[int] = []
        for ang in interior:
            h = _height_for(ang, hv_i, cfg.dip_depth, cfg.epsilon0)
            # Arc distance from this node to the glue node (shorter arc)
            diff = abs(ang - (glue_ang % (2 * np.pi)))
            arc_to_g = r_i * min(diff, 2 * np.pi - diff)
            G.add_node(next_nid, height=h, angle=ang, is_glue=False,
                       component=comp_idx, arc_to_glue=arc_to_g, glue_nid=glue_nid)
            pos[next_nid] = (
                cx + r_i * np.cos(ang + np.pi + ca),
                cy + r_i * np.sin(ang + np.pi + ca),
            )
            interior_nids.append(next_nid)
            next_nid += 1

        # Also tag the glue node with this component if not already set
        if 'component' not in G.nodes[glue_nid] or G.nodes[glue_nid]['component'] == -1:
            G.nodes[glue_nid]['component'] = comp_idx
            G.nodes[glue_nid]['arc_to_glue'] = 0.0
            G.nodes[glue_nid]['glue_nid'] = glue_nid

        # Build sorted node list for this circle
        glue_norm = glue_ang % (2 * np.pi)
        all_ang_node = [(glue_norm, glue_nid)] + list(zip(interior, interior_nids))
        all_ang_node.sort(key=lambda t: t[0])
        ordered_ang = [a for a, _ in all_ang_node]
        ordered_nid = [n for _, n in all_ang_node]
        n_total = len(ordered_nid)

        def arc_len(i: int, j: int) -> float:
            diff = abs(ordered_ang[i] - ordered_ang[j])
            return r_i * min(diff, 2 * np.pi - diff)

        for i in range(n_total):
            for skip in range(1, n_total // 2 + 1):
                j = (i + skip) % n_total
                _add_directed_edge(
                    G, ordered_nid[i], ordered_nid[j],
                    arc_len(i, j), cfg.double_edges,
                )
        return next_nid

    # ------------------------------------------------------------------
    # wedge2 / wedge3 / wedge_k / figure8_asymmetric
    # ------------------------------------------------------------------
    if topo in ('wedge2', 'wedge3', 'wedge_k', 'figure8_asymmetric'):
        # Single gluing node at origin
        glue_h = _height_for(0.0, h_variants[0], cfg.dip_depth, cfg.epsilon0)
        G.add_node(0, height=glue_h, angle=0.0, is_glue=True,
                   component=-1, arc_to_glue=0.0, glue_nid=0)
        pos[0] = (0.0, 0.0)
        next_nid = 1

        for p in range(n_comp):
            r_p = radii[p]
            ca = 2 * np.pi * p / n_comp
            # Circle center r_p away from origin in petal direction ca;
            # the shape_pph.py layout formula then places ang=0 at (0, 0).
            cx = r_p * np.cos(ca)
            cy = r_p * np.sin(ca)
            next_nid = _build_circle_component(p, 0, 0.0, (cx, cy))

    # ------------------------------------------------------------------
    # theta — two gluing nodes, three paths between them
    # ------------------------------------------------------------------
    elif topo == 'theta':
        # Two circles sharing both endpoints (angle 0 and angle π)
        r0, r1 = radii[0], radii[1]

        # Glue node 0 at angle 0 (rightmost point of circle 0)
        glue0_h = _height_for(0.0, h_variants[0], cfg.dip_depth, cfg.epsilon0)
        G.add_node(0, height=glue0_h, angle=0.0, is_glue=True)
        pos[0] = (r0, 0.0)

        # Glue node 1 at angle π (leftmost point of circle 0)
        glue1_h = _height_for(np.pi, h_variants[0], cfg.dip_depth, cfg.epsilon0)
        G.add_node(1, height=glue1_h, angle=np.pi, is_glue=True)
        pos[1] = (-r0, 0.0)

        next_nid = 2

        # Upper arc: angles in (0, π) — component 0
        def _build_theta_arc(
            comp_idx: int,
            angle_lo: float, angle_hi: float,
            nid_lo: int, nid_hi: int,
            r: float, center_xy: tuple[float, float],
        ) -> None:
            nonlocal next_nid
            n_i = counts[comp_idx]
            samp_i = samplings[comp_idx]
            hv_i = h_variants[comp_idx]
            seed_i = _seed(comp_idx)
            ph_i = phases[comp_idx] if comp_idx < len(phases) else 0.0
            cx, cy = center_xy

            # Sample n_i - 2 interior points in (angle_lo, angle_hi)
            n_interior = max(0, n_i - 2)
            interior_angs: list[float] = []
            if n_interior > 0:
                tmp = _CircleConfig(
                    n=n_interior, sampling=samp_i,
                    random_seed=seed_i,
                    jitter_std=cfg.jitter_std, cluster_std=cfg.cluster_std,
                    beta_a=cfg.beta_a, beta_b=cfg.beta_b,
                )
                raw = _sample_angles_circle(tmp)
                span = angle_hi - angle_lo
                for a in raw:
                    mapped = angle_lo + (a / (2 * np.pi)) * span + ph_i
                    interior_angs.append(mapped % (2 * np.pi))
                interior_angs.sort()

            # Build node list: lo endpoint + interior + hi endpoint
            all_seq: list[tuple[float, int]] = [(angle_lo, nid_lo)]
            for ang in interior_angs:
                h = _height_for(ang, hv_i, cfg.dip_depth, cfg.epsilon0)
                G.add_node(next_nid, height=h, angle=ang, is_glue=False)
                pos[next_nid] = (cx + r * np.cos(ang), cy + r * np.sin(ang))
                all_seq.append((ang, next_nid))
                next_nid += 1
            all_seq.append((angle_hi, nid_hi))

            # Edges: only consecutive along the arc (no skip edges — theta arcs are paths)
            for i in range(len(all_seq) - 1):
                a0, n0 = all_seq[i]
                a1, n1 = all_seq[i + 1]
                w = r * abs(a1 - a0)
                _add_directed_edge(G, n0, n1, w, cfg.double_edges)

        # Three arcs: upper half of circle 0, lower half of circle 0,
        # and upper half of circle 1 (with different radius)
        _build_theta_arc(0, 0.0, np.pi, 0, 1, r0, (0.0, 0.0))
        _build_theta_arc(0, np.pi, 2 * np.pi, 1, 0, r0, (0.0, 0.0))
        _build_theta_arc(1, 0.0, np.pi, 0, 1, r1, (0.0, r1 - r0))

    # ------------------------------------------------------------------
    # lollipop — circle + directed tail path
    # ------------------------------------------------------------------
    elif topo == 'lollipop':
        r = radii[0]
        # Glue node at origin — angle 0 of the circle, which the layout formula
        # maps to (0, 0) when the circle centre is at (r, 0).
        glue_h = _height_for(0.0, h_variants[0], cfg.dip_depth, cfg.epsilon0)
        G.add_node(0, height=glue_h, angle=0.0, is_glue=True)
        pos[0] = (0.0, 0.0)
        next_nid = 1

        # Circle centered at (r, 0); the layout formula places ang=0 at (0, 0).
        next_nid = _build_circle_component(0, 0, 0.0, (r, 0.0))

        # Build the tail: bridge_length intermediate nodes directed toward glue.
        # Tail extends in the -x direction from the glue node.
        n_tail = max(0, cfg.bridge_length)
        tail_spacing = r / (n_tail + 1) if n_tail > 0 else r
        tail_nodes: list[int] = []
        for i in range(n_tail):
            tx = -(i + 1) * tail_spacing
            ty = 0.0
            # Height decreases along tail away from glue node
            tail_h = glue_h - (i + 1) * 0.1
            G.add_node(next_nid, height=tail_h, angle=0.0, is_glue=False)
            pos[next_nid] = (tx, ty)
            tail_nodes.append(next_nid)
            next_nid += 1

        # Connect tail: tip → ... → glue (tip lowest → glue highest)
        tail_chain = list(reversed(tail_nodes)) + [0]
        for i in range(len(tail_chain) - 1):
            u, v = tail_chain[i], tail_chain[i + 1]
            _add_directed_edge(G, u, v, tail_spacing, cfg.double_edges)

    # ------------------------------------------------------------------
    # eyeglasses — two circles connected by a bridge path
    # ------------------------------------------------------------------
    elif topo == 'eyeglasses':
        r0, r1 = radii[0], radii[1]

        # Glue node for circle 0 (right attachment, angle 0)
        h_g0 = _height_for(0.0, h_variants[0], cfg.dip_depth, cfg.epsilon0)
        G.add_node(0, height=h_g0, angle=0.0, is_glue=True)
        pos[0] = (r0, 0.0)

        # Glue node for circle 1 (left attachment, angle π)
        bridge_gap = max(0.5, cfg.bridge_length * 0.5)
        h_g1 = _height_for(np.pi, h_variants[1], cfg.dip_depth, cfg.epsilon0)
        G.add_node(1, height=h_g1, angle=np.pi, is_glue=True)
        bridge_x = r0 + bridge_gap + r1
        pos[1] = (bridge_x - r1, 0.0)

        next_nid = 2

        # Circle 0 centered at origin
        next_nid = _build_circle_component(0, 0, 0.0, (0.0, 0.0))
        # Circle 1 centered to the right
        cx1 = bridge_x
        next_nid = _build_circle_component(1, 1, np.pi, (cx1, 0.0))

        # Bridge path: node 0 → intermediates → node 1
        n_bridge = max(0, cfg.bridge_length)
        g0_pos = pos[0]
        g1_pos = pos[1]
        dist = np.sqrt((g1_pos[0] - g0_pos[0])**2 + (g1_pos[1] - g0_pos[1])**2)
        w_bridge = dist / (n_bridge + 1) if n_bridge > 0 else dist

        bridge_nodes = [0]
        for i in range(n_bridge):
            t = (i + 1) / (n_bridge + 1)
            bx = g0_pos[0] + t * (g1_pos[0] - g0_pos[0])
            by = g0_pos[1] + t * (g1_pos[1] - g0_pos[1])
            bh = h_g0 + t * (h_g1 - h_g0)
            G.add_node(next_nid, height=bh, angle=0.0, is_glue=False)
            pos[next_nid] = (bx, by)
            bridge_nodes.append(next_nid)
            next_nid += 1
        bridge_nodes.append(1)

        for i in range(len(bridge_nodes) - 1):
            u, v = bridge_nodes[i], bridge_nodes[i + 1]
            _add_directed_edge(G, u, v, w_bridge, cfg.double_edges)

    # ------------------------------------------------------------------
    # chain — circles in a line, adjacent pairs sharing one node
    # ------------------------------------------------------------------
    elif topo == 'chain':
        # Glue nodes: 0=leftmost, 1=between c0&c1, 2=between c1&c2, ...
        glue_nids: list[int] = []
        x_cursor = 0.0

        for i in range(n_comp + 1):
            ang = 0.0  # all glue nodes use angle 0 for height
            h_g = _height_for(ang, h_variants[min(i, n_comp - 1)],
                               cfg.dip_depth, cfg.epsilon0)
            G.add_node(next_nid, height=h_g, angle=ang, is_glue=True)
            pos[next_nid] = (x_cursor, 0.0)
            glue_nids.append(next_nid)
            next_nid += 1
            if i < n_comp:
                x_cursor += 2 * radii[i] * 1.1

        # Build each circle between glue_nids[i] and glue_nids[i+1]
        # Each circle has two glue nodes: left (angle π) and right (angle 0)
        for i in range(n_comp):
            r_i = radii[i]
            cx = (pos[glue_nids[i]][0] + pos[glue_nids[i + 1]][0]) / 2
            cy = 0.0

            n_i = counts[i]
            samp_i = samplings[i]
            hv_i = h_variants[i]
            seed_i = _seed(i)
            ph_i = phases[i] if i < len(phases) else 0.0

            g_left = glue_nids[i]
            g_right = glue_nids[i + 1]
            ang_left = np.pi   # left glue at angle π
            ang_right = 0.0    # right glue at angle 0

            # Sample interior angles in (0, π) ∪ (π, 2π), i.e. excluding both endpoints
            n_interior = max(0, n_i - 2)
            interior_angs: list[float] = []
            if n_interior > 0:
                tmp = _CircleConfig(
                    n=n_interior, sampling=samp_i,
                    random_seed=seed_i,
                    jitter_std=cfg.jitter_std, cluster_std=cfg.cluster_std,
                    beta_a=cfg.beta_a, beta_b=cfg.beta_b,
                )
                raw = _sample_angles_circle(tmp)
                for a in raw:
                    shifted = (a + ph_i + 1e-9) % (2 * np.pi)
                    # Avoid exact glue angles
                    if abs(shifted) < 1e-8 or abs(shifted - np.pi) < 1e-8:
                        shifted = (shifted + 0.05) % (2 * np.pi)
                    interior_angs.append(shifted)
                interior_angs.sort()

            # Build sorted all-node list by angle
            all_ang_nid: list[tuple[float, int]] = [
                (ang_right, g_right),
                (ang_left, g_left),
            ]
            for ang in interior_angs:
                h = _height_for(ang, hv_i, cfg.dip_depth, cfg.epsilon0)
                G.add_node(next_nid, height=h, angle=ang, is_glue=False)
                pos[next_nid] = (cx + r_i * np.cos(ang), cy + r_i * np.sin(ang))
                all_ang_nid.append((ang, next_nid))
                next_nid += 1

            all_ang_nid.sort(key=lambda t: t[0])
            ordered_ang2 = [a for a, _ in all_ang_nid]
            ordered_nid2 = [n for _, n in all_ang_nid]
            n_tot2 = len(ordered_nid2)

            def arc_len2(ii: int, jj: int, _r=r_i, _oa=ordered_ang2) -> float:
                diff = abs(_oa[ii] - _oa[jj])
                return _r * min(diff, 2 * np.pi - diff)

            for ii in range(n_tot2):
                for skip in range(1, n_tot2 // 2 + 1):
                    jj = (ii + skip) % n_tot2
                    _add_directed_edge(
                        G, ordered_nid2[ii], ordered_nid2[jj],
                        arc_len2(ii, jj), cfg.double_edges,
                    )

    # ------------------------------------------------------------------
    # necklace — circles in a ring, adjacent pairs sharing one node
    # ------------------------------------------------------------------
    elif topo == 'necklace':
        k = n_comp
        # Place glue nodes evenly around a ring whose radius is chosen so
        # adjacent glue nodes are 2*r apart (chord = diameter of each circle)
        r_avg = float(np.mean(radii))
        # chord between adjacent glue nodes = 2*r_avg
        # chord = 2 * ring_r * sin(π/k)  →  ring_r = r_avg / sin(π/k)
        ring_r = r_avg / np.sin(np.pi / k) if k > 1 else r_avg * 2

        glue_nids = []
        for i in range(k):
            ang_on_ring = 2 * np.pi * i / k
            gx = ring_r * np.cos(ang_on_ring)
            gy = ring_r * np.sin(ang_on_ring)
            h_g = _height_for(ang_on_ring, h_variants[i],
                               cfg.dip_depth, cfg.epsilon0)
            G.add_node(next_nid, height=h_g, angle=ang_on_ring, is_glue=True)
            pos[next_nid] = (gx, gy)
            glue_nids.append(next_nid)
            next_nid += 1

        for i in range(k):
            j = (i + 1) % k
            g_left  = glue_nids[i]
            g_right = glue_nids[j]
            r_i   = radii[i]
            hv_i  = h_variants[i]
            samp_i = samplings[i]
            seed_i = _seed(i)
            ph_i   = phases[i] if i < len(phases) else 0.0

            # Positions of the two glue nodes
            px0, py0 = pos[g_left]
            px1, py1 = pos[g_right]

            # Circle centre: midpoint offset outward (away from ring centre)
            mx, my = (px0 + px1) / 2, (py0 + py1) / 2
            chord = np.hypot(px1 - px0, py1 - py0)
            # Sagitta: how far the circle bows outward from the chord
            # For a circle of radius r_i and chord c: h = r_i - sqrt(r_i²-(c/2)²)
            half_chord = chord / 2
            if r_i <= half_chord:
                r_i = half_chord * 1.05   # ensure circle can span the chord
            sagitta = r_i - np.sqrt(max(r_i**2 - half_chord**2, 0.0))

            # Outward normal (away from ring centre (0,0))
            out_x, out_y = mx, my
            out_len = np.hypot(out_x, out_y) + 1e-12
            out_x, out_y = out_x / out_len, out_y / out_len

            # Circle centre is inward from midpoint by (r_i - sagitta)
            inset = r_i - sagitta
            cx = mx + out_x * sagitta
            cy = my + out_y * sagitta

            # Angles of glue nodes as seen from this circle centre
            ang_left_local  = np.arctan2(py0 - cy, px0 - cx)
            ang_right_local = np.arctan2(py1 - cy, px1 - cx)

            # Sample interior angles between the two glue angles (going outward)
            # Normalise to [0, 2π) and find the arc that bows outward
            a0 = ang_left_local % (2 * np.pi)
            a1 = ang_right_local % (2 * np.pi)

            # The outward arc goes via the angle pointing away from ring centre
            out_ang = np.arctan2(out_y, out_x)
            # Pick the arc direction that contains out_ang
            def _arc_contains(start, end, test):
                """True if test is on the CCW arc from start to end."""
                start, end, test = start%(2*np.pi), end%(2*np.pi), test%(2*np.pi)
                if start <= end:
                    return start <= test <= end
                return test >= start or test <= end

            # We want angles strictly between a0 and a1 on the outward arc
            n_i    = counts[i]
            n_interior = max(0, n_i - 2)
            interior_angs: list[float] = []
            if n_interior > 0:
                tmp = _CircleConfig(
                    n=n_interior, sampling=samp_i, random_seed=seed_i,
                    jitter_std=cfg.jitter_std, cluster_std=cfg.cluster_std,
                    beta_a=cfg.beta_a, beta_b=cfg.beta_b,
                )
                raw = _sample_angles_circle(tmp)
                # Map raw uniform angles to the outward arc
                # Determine arc span
                span = (a1 - a0) % (2 * np.pi)
                if not _arc_contains(a0, a1, out_ang % (2*np.pi)):
                    # use the other arc
                    span = 2 * np.pi - span
                    a0, a1 = a1, a0
                for ra in raw:
                    # ra in [0,2π) → map to (a0, a0+span) excluding endpoints
                    frac = (ra / (2 * np.pi)) * 0.9 + 0.05   # avoid exact endpoints
                    local_ang = (a0 + frac * span) % (2 * np.pi)
                    interior_angs.append(local_ang)
                interior_angs.sort()

            # Build node list including glue nodes at their local angles
            all_ang_nid2: list[tuple[float, int]] = [
                (a0 % (2*np.pi), g_left),
                (a1 % (2*np.pi), g_right),
            ]
            for ang in interior_angs:
                h = _height_for(ang + ph_i, hv_i, cfg.dip_depth, cfg.epsilon0)
                nx_pos = cx + r_i * np.cos(ang)
                ny_pos = cy + r_i * np.sin(ang)
                G.add_node(next_nid, height=h, angle=ang, is_glue=False)
                pos[next_nid] = (nx_pos, ny_pos)
                all_ang_nid2.append((ang, next_nid))
                next_nid += 1

            all_ang_nid2.sort(key=lambda t: t[0])
            ordered_ang3 = [a for a, _ in all_ang_nid2]
            ordered_nid3 = [n for _, n in all_ang_nid2]
            n_tot3 = len(ordered_nid3)

            def arc_len3(ii: int, jj: int, _r=r_i, _oa=ordered_ang3) -> float:
                diff = abs(_oa[ii] - _oa[jj])
                return _r * min(diff, 2 * np.pi - diff)

            for ii in range(n_tot3):
                for skip in range(1, n_tot3 // 2 + 1):
                    jj = (ii + skip) % n_tot3
                    _add_directed_edge(
                        G, ordered_nid3[ii], ordered_nid3[jj],
                        arc_len3(ii, jj), cfg.double_edges,
                    )

    # ------------------------------------------------------------------
    # necklace_full — like necklace but samples from the FULL circle at
    # each junction, so nodes appear on both inner and outer arcs.
    # Each circle is centred at the midpoint between its two glue nodes,
    # with radius chosen to pass through them. Interior nodes are sampled
    # uniformly from the full 2π of that circle (excluding the two glue angles).
    # ------------------------------------------------------------------
    elif topo == 'necklace_full':
        k = n_comp
        r_avg = float(np.mean(radii))
        # Place glue nodes on a ring so adjacent ones are exactly 2*r_avg apart
        ring_r = r_avg / np.sin(np.pi / k) if k > 1 else r_avg * 2

        glue_nids = []
        for i in range(k):
            ang_on_ring = 2 * np.pi * i / k
            gx = ring_r * np.cos(ang_on_ring)
            gy = ring_r * np.sin(ang_on_ring)
            h_g = _height_for(ang_on_ring, h_variants[i],
                               cfg.dip_depth, cfg.epsilon0)
            G.add_node(next_nid, height=h_g, angle=ang_on_ring, is_glue=True,
                       component=-1, arc_to_glue=0.0, glue_nid=next_nid)
            pos[next_nid] = (gx, gy)
            glue_nids.append(next_nid)
            next_nid += 1

        for i in range(k):
            j = (i + 1) % k
            g_left  = glue_nids[i]
            g_right = glue_nids[j]
            r_i    = radii[i]
            hv_i   = h_variants[i]
            samp_i = samplings[i]
            seed_i = _seed(i)
            ph_i   = phases[i] if i < len(phases) else 0.0

            px0, py0 = pos[g_left]
            px1, py1 = pos[g_right]

            # Chord between the two glue nodes
            half_chord = np.hypot(px1 - px0, py1 - py0) / 2

            # Use a radius larger than half_chord so the full circle is visible
            # r_i from config, but must be at least half_chord to span the chord
            if r_i < half_chord:
                r_i = half_chord * 1.5

            # Circle centre lies on the perpendicular bisector of the chord.
            # We place it so the circle bows OUTWARD from the ring centre.
            mx, my = (px0 + px1) / 2, (py0 + py1) / 2
            # Distance from chord midpoint to circle centre along perpendicular
            # d = sqrt(r_i² - half_chord²)
            d_perp = np.sqrt(max(r_i**2 - half_chord**2, 0.0))

            # Outward direction (away from ring centre at origin)
            out_len = np.hypot(mx, my) + 1e-12
            out_x, out_y = mx / out_len, my / out_len

            # Centre offset outward so the circle bows away from ring centre
            cx = mx + out_x * d_perp
            cy = my + out_y * d_perp

            # Local angles of the two glue nodes as seen from circle centre
            ang_left_local  = np.arctan2(py0 - cy, px0 - cx) % (2 * np.pi)
            ang_right_local = np.arctan2(py1 - cy, px1 - cx) % (2 * np.pi)

            # Sample n_i - 2 interior angles from the FULL circle [0, 2π),
            # excluding small windows around the two glue angles
            n_i = counts[i]
            n_interior = max(0, n_i - 2)
            interior_angs: list[float] = []
            if n_interior > 0:
                tmp = _CircleConfig(
                    n=n_interior * 4,   # oversample then filter
                    sampling=samp_i, random_seed=seed_i,
                    jitter_std=cfg.jitter_std, cluster_std=cfg.cluster_std,
                    beta_a=cfg.beta_a, beta_b=cfg.beta_b,
                )
                raw = _sample_angles_circle(tmp)
                eps_ang = 0.08  # exclusion window around glue angles (radians)
                candidates = []
                for ra in raw:
                    a = (ra + ph_i) % (2 * np.pi)
                    # Exclude angles too close to either glue node
                    d0 = min(abs(a - ang_left_local),
                             2*np.pi - abs(a - ang_left_local))
                    d1 = min(abs(a - ang_right_local),
                             2*np.pi - abs(a - ang_right_local))
                    if d0 > eps_ang and d1 > eps_ang:
                        candidates.append(a)
                # Take the first n_interior candidates
                candidates.sort()
                # Spread selection evenly from candidates
                if len(candidates) >= n_interior:
                    step = len(candidates) / n_interior
                    interior_angs = [candidates[int(ii * step)]
                                     for ii in range(n_interior)]
                else:
                    interior_angs = candidates

            # Build node list: both glue nodes + interior
            all_ang_nid: list[tuple[float, int]] = [
                (ang_left_local,  g_left),
                (ang_right_local, g_right),
            ]
            for ang in interior_angs:
                nx_x = cx + r_i * np.cos(ang)
                nx_y = cy + r_i * np.sin(ang)
                # Arc to nearest glue node
                d0 = r_i * min(abs(ang - ang_left_local),
                               2*np.pi - abs(ang - ang_left_local))
                d1 = r_i * min(abs(ang - ang_right_local),
                               2*np.pi - abs(ang - ang_right_local))
                nearest_g = g_left if d0 <= d1 else g_right
                arc_g = min(d0, d1)
                h = _height_for(ang + ph_i, hv_i, cfg.dip_depth, cfg.epsilon0)
                G.add_node(next_nid, height=h, angle=ang, is_glue=False,
                           component=i, arc_to_glue=arc_g, glue_nid=nearest_g)
                pos[next_nid] = (nx_x, nx_y)
                all_ang_nid.append((ang, next_nid))
                next_nid += 1

            all_ang_nid.sort(key=lambda t: t[0])
            ordered_ang = [a for a, _ in all_ang_nid]
            ordered_nid = [n for _, n in all_ang_nid]
            n_tot = len(ordered_nid)

            def arc_len_full(ii: int, jj: int, _r=r_i, _oa=ordered_ang) -> float:
                diff = abs(_oa[ii] - _oa[jj])
                return _r * min(diff, 2 * np.pi - diff)

            for ii in range(n_tot):
                for skip in range(1, n_tot // 2 + 1):
                    jj = (ii + skip) % n_tot
                    _add_directed_edge(
                        G, ordered_nid[ii], ordered_nid[jj],
                        arc_len_full(ii, jj), cfg.double_edges,
                    )

    else:
        raise ValueError(f"Unknown topology: {topo!r}")

    # ------------------------------------------------------------------
    # Global height recompute (for position-based variants)
    # Must run after all nodes have their 2D positions assigned.
    # ------------------------------------------------------------------
    if cfg.height_variant in _GLOBAL_VARIANTS or any(
            hv in _GLOBAL_VARIANTS for hv in h_variants):
        # Characteristic scale: mean radius
        scale = float(np.mean(radii)) if radii else 1.0
        # For global_dist_max: find max distance from origin first
        if cfg.height_variant == 'global_dist_max' or any(
                hv == 'global_dist_max' for hv in h_variants):
            max_dist = max(np.sqrt(x**2 + y**2) for x, y in pos.values()) or 1.0
        else:
            max_dist = 1.0

        for nid in G.nodes():
            x, y = pos[nid]
            # Determine which variant applies to this node
            comp = G.nodes[nid].get('component', 0)
            variant = (h_variants[comp]
                       if comp is not None and 0 <= comp < len(h_variants)
                       else cfg.height_variant)
            if variant not in _GLOBAL_VARIANTS:
                continue
            if variant == 'global_dist_max':
                d = np.sqrt(x**2 + y**2)
                h = float((max_dist - d) / max_dist)   # 1 at origin, 0 at periphery
            else:
                h = global_height(x, y, variant, scale=scale)
            G.nodes[nid]['height'] = h

    # ------------------------------------------------------------------
    # Cross-component edges (if distance != 'within_only')
    # ------------------------------------------------------------------
    if cfg.distance != 'within_only':
        node_list = list(G.nodes())
        N = len(node_list)

        # Find all glue nodes for geodesic computation
        glue_nodes = [n for n in node_list if G.nodes[n].get('is_glue', False)]

        def _arc_to_nearest_glue(n: int) -> tuple[float, int]:
            """Return (arc_length_to_nearest_glue, glue_nid)."""
            d = G.nodes[n].get('arc_to_glue', None)
            g = G.nodes[n].get('glue_nid', None)
            if d is not None and g is not None:
                return d, g
            # Fallback: find nearest glue by Euclidean distance
            px, py = pos[n]
            best_d, best_g = float('inf'), glue_nodes[0]
            for gn in glue_nodes:
                gx, gy = pos[gn]
                ed = np.hypot(px - gx, py - gy)
                if ed < best_d:
                    best_d, best_g = ed, gn
            return best_d, best_g

        for i in range(N):
            u = node_list[i]
            cu = G.nodes[u].get('component', -1)
            for j in range(i + 1, N):
                v = node_list[j]
                cv = G.nodes[v].get('component', -1)

                # Skip same-component pairs (already handled within-component)
                if cu == cv and cu != -1:
                    continue

                ux, uy = pos[u]
                vx, vy = pos[v]

                if cfg.distance == 'euclidean':
                    w = float(np.hypot(ux - vx, uy - vy))

                elif cfg.distance == 'geodesic':
                    # Shortest path through any shared glue node:
                    # d(u,v) = min over glue nodes g of (arc(u→g) + arc(g→v))
                    # For topologies with one glue node this is just arc_u + arc_v
                    best = float('inf')
                    for gn in glue_nodes:
                        gx, gy = pos[gn]
                        # Arc from u to gn: use stored value if same glue, else Euclidean
                        d_u = G.nodes[u].get('arc_to_glue', np.hypot(ux-gx, uy-gy)) \
                              if G.nodes[u].get('glue_nid') == gn \
                              else np.hypot(ux - gx, uy - gy)
                        d_v = G.nodes[v].get('arc_to_glue', np.hypot(vx-gx, vy-gy)) \
                              if G.nodes[v].get('glue_nid') == gn \
                              else np.hypot(vx - gx, vy - gy)
                        best = min(best, d_u + d_v)
                    w = best

                elif cfg.distance == 'arc_sum':
                    # Sum of each node's arc to its nearest glue node
                    d_u, _ = _arc_to_nearest_glue(u)
                    d_v, _ = _arc_to_nearest_glue(v)
                    w = d_u + d_v

                elif cfg.distance == 'global_arc':
                    # Treat angles as if on one shared circle; distance = angle diff
                    au = G.nodes[u].get('angle', 0.0)
                    av = G.nodes[v].get('angle', 0.0)
                    diff = abs(au - av)
                    r_mean = float(np.mean(radii))
                    w = r_mean * min(diff, 2 * np.pi - diff)

                else:
                    continue

                _add_directed_edge(G, u, v, w, cfg.double_edges)

    return G, pos


# ---------------------------------------------------------------------------
# ENTRY POINTS
# ---------------------------------------------------------------------------

def analyze(cfg: Config) -> Result:
    """Run full PPH analysis for given Config, using a disk cache."""
    key = _cfg_key(cfg)
    cached = _load_cached(key)
    if cached is not None:
        return cached

    G, _ = build_graph(cfg)
    barcode = compute_homology(G, cfg.homology)

    n_comp = _n_components(cfg)
    result = Result(
        topology=cfg.topology,
        n_total=G.number_of_nodes(),
        n_components=n_comp,
        barcode=barcode,
        homology=cfg.homology,
    )
    _save_cached(key, result)
    return result


def analyze_simple(
    topology: TopologyVariant,
    total_n: int = 16,
    allocation: Literal['uniform', 'random', 'proportional'] = 'uniform',
    double_edges: bool = True,
    height_variant: HeightVariant = 'standard',
    homology: HomologyVariant = 'path',
    **kwargs,
) -> Result:
    """Convenience wrapper for common single-topology analyses."""
    return analyze(Config(
        topology=topology,
        total_n=total_n,
        allocation=allocation,
        double_edges=double_edges,
        height_variant=height_variant,
        homology=homology,
        **kwargs,
    ))


def batch_analyze(
    topology: TopologyVariant,
    total_n_range: range,
    allocation: Literal['uniform', 'random', 'proportional'] = 'uniform',
    **kwargs,
) -> list[Result]:
    """Analyze a topology across a range of total_n values."""
    return [
        analyze_simple(topology, total_n=n, allocation=allocation, **kwargs)
        for n in total_n_range
    ]


def compare(
    topology: TopologyVariant,
    total_n_range: range,
    height_variant: HeightVariant = 'standard',
    homology: HomologyVariant = 'path',
) -> dict:
    """Compare with/without double edges for a topology."""
    return {
        'without': batch_analyze(topology, total_n_range,
                                 double_edges=False,
                                 height_variant=height_variant,
                                 homology=homology),
        'with':    batch_analyze(topology, total_n_range,
                                 double_edges=True,
                                 height_variant=height_variant,
                                 homology=homology),
    }


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def plot_topology_graph(
    cfg: Config,
    output: str | Path | None = None,
) -> plt.Figure:
    """Draw the assembled graph for cfg; nodes by height, edges by weight."""
    G, pos = build_graph(cfg)

    heights = [G.nodes[n]['height'] for n in G.nodes]
    h_arr = np.array(heights)
    h_min, h_max = h_arr.min(), h_arr.max()
    node_norm = plt.Normalize(vmin=h_min, vmax=h_max)
    node_cmap = plt.cm.RdYlBu_r

    weights = [G.edges[u, v]['weight'] for u, v in G.edges()]
    w_min = min(weights) if weights else 0.0
    w_max = max(weights) if weights else 1.0
    edge_norm = plt.Normalize(vmin=w_min, vmax=w_max + 1e-9)
    edge_cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw edges
    for (u, v), w in zip(G.edges(), weights):
        ec = edge_cmap(edge_norm(w))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            arrows=True, arrowsize=12,
            edge_color=[ec], width=1.5,
            connectionstyle='arc3,rad=0.08',
        )

    # Non-gluing nodes
    regular = [n for n in G.nodes if not G.nodes[n].get('is_glue', False)]
    glue_ns = [n for n in G.nodes if G.nodes[n].get('is_glue', False)]

    if regular:
        node_colors = [node_cmap(node_norm(G.nodes[n]['height'])) for n in regular]
        nx.draw_networkx_nodes(
            G, pos, nodelist=regular, ax=ax,
            node_color=node_colors, node_size=200,
        )

    # Gluing nodes — larger, black border
    if glue_ns:
        gc = [node_cmap(node_norm(G.nodes[n]['height'])) for n in glue_ns]
        nx.draw_networkx_nodes(
            G, pos, nodelist=glue_ns, ax=ax,
            node_color=gc, node_size=500,
            edgecolors='black', linewidths=2.0,
        )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_color='black')

    # Colorbars
    sm_h = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_h.set_array([])
    fig.colorbar(sm_h, ax=ax, label='Node height', fraction=0.03, pad=0.01)

    sm_e = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm_e.set_array([])
    fig.colorbar(sm_e, ax=ax, label='Edge weight', fraction=0.03, pad=0.05)

    ax.set_title(
        f'{cfg.topology}  |  n={G.number_of_nodes()} nodes  |  '
        f'{G.number_of_edges()} edges  |  {cfg.height_variant}',
        fontweight='bold',
    )
    ax.axis('off')
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")
    return fig


def plot_death_times(
    results_no: list[Result],
    results_yes: list[Result],
    output: str | Path | None = None,
    homology: HomologyVariant = 'path',
) -> plt.Figure:
    """Plot death times comparison (with vs without double edges)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [
        (axes[0], results_no, 'WITHOUT Double Edges'),
        (axes[1], results_yes, 'WITH Double Edges'),
    ]:
        for r in results:
            mod4 = r.n_total % 4
            for bar in r.barcode:
                death = bar[1] / np.pi if np.isfinite(bar[1]) else 1.05
                ax.scatter(r.n_total, death, c=MOD4_COLORS[mod4],
                           s=40, alpha=0.7)

        ax.axhline(y=0.5, color='black', ls='--', lw=2, label='π/2')
        ax.set_xlabel('total n')
        ax.set_ylabel('Death time (units of π)')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.15)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=MOD4_COLORS[i], markersize=10,
                       label=f'n ≡ {i} (mod 4)')
            for i in range(4)
        ]
        ax.legend(handles=handles, loc='upper right')

    hom_label = 'Path homology' if homology == 'path' else 'Flag homology'
    plt.suptitle(f'{hom_label} — Death Times: Effect of Double Edges',
                 fontweight='bold', y=1.02)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")
    return fig


def plot_bar_counts(
    results_no: list[Result],
    results_yes: list[Result],
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot H₁ bar count comparison (with vs without double edges)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ns = [r.n_total for r in results_no]
    ax.plot(ns, [r.n_bars for r in results_no], 'o-',
            label='Without double edges', color='#2980b9', markersize=5)
    ax.plot(ns, [r.n_bars for r in results_yes], 's--',
            label='With double edges', color='#e74c3c', markersize=5)
    ax.set_xlabel('total n')
    ax.set_ylabel('Number of H₁ bars')
    ax.set_title('Bar Count: Effect of Double Edges', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output}")
    return fig


def plot_persistence_diagram(
    result: Result,
    output: str | Path | None = None,
) -> plt.Figure:
    """Plot persistence diagram for a single Result."""
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
    ax.set_title(
        f'Persistence diagram — {result.topology}, n={result.n_total}',
        fontweight='bold',
    )
    ax.set_xlim(-0.1, max_val * 1.1)
    ax.set_ylim(-0.1, max_val * 1.2)
    ax.set_xticks([0, np.pi / 2, np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π'])
    ax.set_yticks([0, np.pi / 2, np.pi])
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


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    """Self-check, sweep, and plot for wedge topologies."""
    import argparse

    parser = argparse.ArgumentParser(description='PPH analysis on wedge sums of circles')
    parser.add_argument('--quick', action='store_true',
                        help='Skip total_n sweep table')
    parser.add_argument('--output-dir', type=Path, default=Path('.'),
                        help='Directory for saved plots')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.ioff()  # non-interactive for batch output

    print('=' * 65)
    print('PERSISTENT PATH HOMOLOGY — WEDGE SUMS OF CIRCLES')
    print('=' * 65)

    # ------------------------------------------------------------------
    # Sanity check 1: wedge2, total_n=20, uniform allocation → 2 H₁ bars
    # ------------------------------------------------------------------
    print('\n[Sanity check 1] wedge2, total_n=20, uniform, standard height')
    r1 = analyze(Config(
        topology='wedge2',
        total_n=20,
        allocation='uniform',
        height_variant='standard',
        double_edges=True,
    ))
    expected1 = 2
    status1 = '✓ PASS' if r1.n_bars == expected1 else f'✗ FAIL (got {r1.n_bars})'
    print(f'  bars={r1.n_bars}, expected={expected1} → {status1}')

    # ------------------------------------------------------------------
    # Sanity check 2: wedge3, n_per_circle=8, standard height → 3 H₁ bars
    # ------------------------------------------------------------------
    print('\n[Sanity check 2] wedge3, n_per_circle=8, standard height')
    r2 = analyze(Config(
        topology='wedge3',
        n_per_circle=8,
        height_variant='standard',
        double_edges=True,
    ))
    expected2 = 3
    status2 = '✓ PASS' if r2.n_bars == expected2 else f'✗ FAIL (got {r2.n_bars})'
    print(f'  bars={r2.n_bars}, expected={expected2} → {status2}')

    # ------------------------------------------------------------------
    # Sweep total_n 10..40 step 2 for wedge2 and wedge3
    # ------------------------------------------------------------------
    if not args.quick:
        print('\n[Sweep] total_n ∈ [10, 40] step 2 — wedge2 and wedge3')
        print(f'\n{"topology":>10}  {"total_n":>8}  {"allocation":>12}  {"n_bars":>6}')
        print('-' * 45)
        for alloc in ('uniform', 'proportional'):
            for topo in ('wedge2', 'wedge3'):
                for tn in range(10, 41, 2):
                    r = analyze(Config(
                        topology=topo,
                        total_n=tn,
                        allocation=alloc,
                        height_variant='standard',
                        double_edges=True,
                    ))
                    print(f'{topo:>10}  {tn:>8}  {alloc:>12}  {r.n_bars:>6}')

    # ------------------------------------------------------------------
    # Demonstrate figure8_asymmetric with radii=[1.0, 2.0], proportional
    # ------------------------------------------------------------------
    print('\n[Demo] figure8_asymmetric, radii=[1.0, 2.0], proportional, total_n=20')
    r_asym = analyze(Config(
        topology='figure8_asymmetric',
        radii=[1.0, 2.0],
        total_n=20,
        allocation='proportional',
        height_variant='standard',
        double_edges=True,
    ))
    print(f'  bars={r_asym.n_bars}, n_total={r_asym.n_total}')
    for b, d in r_asym.barcode:
        d_str = f'{d/np.pi:.3f}π' if np.isfinite(d) else '∞'
        print(f'    [{b/np.pi:.3f}π, {d_str}]')

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------
    print('\n[Plots] Generating and saving figures …')

    # Topology graph — wedge2
    cfg_w2 = Config(topology='wedge2', total_n=16, allocation='uniform')
    plot_topology_graph(cfg_w2,
                        output=args.output_dir / 'wedge_graph_wedge2.png')

    # Topology graph — wedge3
    cfg_w3 = Config(topology='wedge3', n_per_circle=6)
    plot_topology_graph(cfg_w3,
                        output=args.output_dir / 'wedge_graph_wedge3.png')

    # Topology graph — necklace k=4
    cfg_neck = Config(topology='necklace', k=4, n_per_circle=6)
    plot_topology_graph(cfg_neck,
                        output=args.output_dir / 'wedge_graph_necklace.png')

    # Topology graph — lollipop
    cfg_loll = Config(topology='lollipop', n_per_circle=8, bridge_length=3)
    plot_topology_graph(cfg_loll,
                        output=args.output_dir / 'wedge_graph_lollipop.png')

    # Topology graph — eyeglasses
    cfg_eye = Config(topology='eyeglasses', n_per_circle=6, bridge_length=2)
    plot_topology_graph(cfg_eye,
                        output=args.output_dir / 'wedge_graph_eyeglasses.png')

    # Persistence diagram — wedge2 sanity result
    plot_persistence_diagram(r1,
                             output=args.output_dir / 'wedge_persistence_wedge2.png')

    # Persistence diagram — figure8_asymmetric
    plot_persistence_diagram(r_asym,
                             output=args.output_dir / 'wedge_persistence_asym.png')

    # Death times + bar counts — wedge2
    cmp_w2 = compare('wedge2', range(10, 31, 2))
    plot_death_times(cmp_w2['without'], cmp_w2['with'],
                     output=args.output_dir / 'wedge_death_times_wedge2.png')
    plot_bar_counts(cmp_w2['without'], cmp_w2['with'],
                    output=args.output_dir / 'wedge_bar_counts_wedge2.png')

    print(f'\nAll outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
