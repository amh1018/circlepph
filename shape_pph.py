"""
shape_pph.py — PPH pipeline extended to wedges of circles and deformed circles.

Imports the core pipeline from circle_pph_2 and adds:
  • WedgeConfig / build_wedge_graph / analyze_wedge  (Part 1)
  • DeformConfig / build_deform_graph / analyze_deform (Part 2)
  • DeformedWedgeConfig / build_deformed_wedge_graph  (Part 4)
  • Combined CLI (Part 3 + 4c)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

# ── re-export everything from the original pipeline ──────────────────────────
from circle_pph_2 import (
    Config,
    HeightVariant,
    SamplingVariant,
    EdgeWeightFiltration,
    compute_pph,
    Result,
    analyze,
    height,
    sample_angles,
    batch_analyze,
    MOD4_COLORS,
    plot_death_times,
)

__all__ = [
    # re-exports
    "Config", "HeightVariant", "SamplingVariant",
    "EdgeWeightFiltration", "compute_pph", "Result", "analyze",
    "height", "sample_angles", "batch_analyze",
    # wedge
    "WedgeConfig", "build_wedge_graph", "analyze_wedge",
    "plot_wedge_graph", "batch_analyze_wedge",
    # deform
    "DeformVariant", "DeformConfig", "parametric_curve",
    "build_deform_graph", "analyze_deform", "plot_deform_graph",
    # deformed wedge
    "DeformedWedgeConfig", "build_deformed_wedge_graph",
    "analyze_deformed_wedge", "plot_deformed_wedge_graph",
    "batch_analyze_deformed_wedge",
]


# =============================================================================
# PART 1 — WEDGE OF CIRCLES
# =============================================================================

@dataclass
class WedgeConfig:
    """
    Configuration for a wedge (bouquet) of k circles W_k.

    The wedge W_k is the 1-complex obtained by identifying k circles at a
    single basepoint.  Each 'petal' is a copy of S¹ sharing node 0.
    """
    k: int
    n_per_petal: int
    height_variant: HeightVariant = 'standard'
    petal_phases: list[float] | None = None   # per-petal phase offset; None → all 0
    sampling: SamplingVariant = 'uniform'
    double_edges: bool = False
    dip_depth: float = 1.0
    epsilon0: float = 0.05
    random_seed: int | None = None
    jitter_std: float = 0.3
    cluster_std: float = 0.3
    beta_a: float = 0.5
    beta_b: float = 0.5


def build_wedge_graph(
    cfg: WedgeConfig,
) -> tuple[nx.DiGraph, list[float], list[tuple[float, float]]]:
    """
    Build the directed graph for the wedge W_k of k circles.

    The graph has k*n_per_petal + 1 nodes.  Node 0 is the shared basepoint
    (image of θ=0 on every petal).  For petal p the interior nodes are
    numbered 1 + p*n_per_petal … p*n_per_petal + n_per_petal.

    Heights are computed with the same `height` function as the circle
    pipeline; an optional per-petal phase shifts the argument.

    Directed edges obey the same rule as the circle: an edge runs from the
    lower-height endpoint to the higher-height endpoint.  The arc-length
    weight is the chord length along that petal (equal to the parameter
    difference for a unit circle).

    Returns
    -------
    G       : directed graph with 'weight' and 'height' attributes
    heights : list of node heights (index = node id)
    coords  : list of (x, y) embedding coordinates (for visualisation)
    """
    phases = cfg.petal_phases if cfg.petal_phases is not None else [0.0] * cfg.k

    # Build a minimal Config adapter so sample_angles works unchanged.
    angle_cfg = Config(
        n=cfg.n_per_petal,
        sampling=cfg.sampling,
        random_seed=cfg.random_seed,
        jitter_std=cfg.jitter_std,
        cluster_std=cfg.cluster_std,
        beta_a=cfg.beta_a,
        beta_b=cfg.beta_b,
    )

    # Sample angles in (0, 2π) — open interval, so the basepoint (θ=0) is
    # not included among petal-interior nodes.
    raw_angles = sample_angles(angle_cfg)          # n_per_petal angles in [0,2π)
    # Shift away from exact 0 and 2π endpoints for a truly open interval.
    petal_angles = [(a if a != 0.0 else 1e-9) for a in raw_angles]
    petal_angles = [(a if a < 2*np.pi - 1e-9 else 2*np.pi - 1e-9) for a in petal_angles]

    G = nx.DiGraph()
    heights: list[float] = []
    coords: list[tuple[float, float]] = []

    # --- basepoint (node 0) ---
    bp_height = height(0.0, cfg.height_variant, cfg.dip_depth, cfg.epsilon0)
    G.add_node(0, height=bp_height, petal=-1)
    heights.append(bp_height)
    coords.append((0.0, 0.0))

    # --- petals ---
    for p in range(cfg.k):
        phase = phases[p]
        # Centre direction for this petal in the layout
        centre_angle = 2 * np.pi * p / cfg.k

        # Offset so petals are visually separated: place each petal's
        # unit circle centred at (2*cos(centre_angle), 2*sin(centre_angle))
        cx = 2.0 * np.cos(centre_angle)
        cy = 2.0 * np.sin(centre_angle)

        for q, ang in enumerate(petal_angles):
            node_id = 1 + p * cfg.n_per_petal + q
            shifted = (ang + phase) % (2 * np.pi)
            h = height(shifted, cfg.height_variant, cfg.dip_depth, cfg.epsilon0)

            # Visual position: point on the unit circle for this petal,
            # rotated so that θ=0 coincides with the global basepoint.
            # We rotate so that the petal's θ=0 touches (0,0) from its centre.
            # The petal circle is centred at distance 2 from origin;
            # the θ=0 point of the petal is at origin.
            # A point at angle `ang` on the petal circle (centre cx,cy,r=1) is:
            #   centre + (cos(ang + π + centre_angle), sin(ang + π + centre_angle))
            # which gives (0,0) at ang=0.  Simplify:
            vx = cx + np.cos(ang + np.pi + centre_angle)
            vy = cy + np.sin(ang + np.pi + centre_angle)

            G.add_node(node_id, height=h, petal=p)
            heights.append(h)
            coords.append((vx, vy))

    # --- edges ---
    def _add_directed_edge(u: int, v: int, w: float) -> None:
        hu, hv = heights[u], heights[v]
        if abs(hu - hv) < 1e-10:
            if cfg.double_edges:
                G.add_edge(u, v, weight=w)
                G.add_edge(v, u, weight=w)
        elif hu < hv:
            G.add_edge(u, v, weight=w)
        else:
            G.add_edge(v, u, weight=w)

    for p in range(cfg.k):
        # Nodes in this petal (interior, in order of petal_angles)
        petal_nodes = [1 + p * cfg.n_per_petal + q for q in range(cfg.n_per_petal)]
        all_nodes = [0] + petal_nodes  # 0 = basepoint

        # Build a circle-like sequence: 0 → interior nodes → back to 0
        # All consecutive pairs share an edge; also 0 and last interior node.
        seq = all_nodes + [0]  # closed loop: 0, n1, n2, …, n_k, 0
        full_angles = [0.0] + petal_angles + [2 * np.pi]

        for i in range(len(seq) - 1):
            u = seq[i]
            v = seq[i + 1]
            a_u = full_angles[i]
            a_v = full_angles[i + 1]
            arc = abs(a_v - a_u)
            _add_directed_edge(u, v, arc)

        # Also add skip edges (as in the circle pipeline: skip up to n//2 steps)
        for start in range(len(all_nodes)):
            for skip in range(2, len(all_nodes) // 2 + 1):
                end = (start + skip) % len(all_nodes)
                u = all_nodes[start]
                v = all_nodes[end]
                # arc length: sum of intermediate steps (wrap-around safe)
                step_angles = full_angles[:len(all_nodes)] + [2 * np.pi]
                # Use chord-distance proxy (same as circle pipeline)
                a_u_angle = full_angles[start] if start < len(full_angles) else 2*np.pi
                a_v_angle = full_angles[end] if end < len(full_angles) else 2*np.pi
                diff = abs(a_u_angle - a_v_angle)
                arc = min(diff, 2*np.pi - diff)
                _add_directed_edge(u, v, arc)

    return G, heights, coords


def analyze_wedge(cfg: WedgeConfig) -> Result:
    """
    Run the full PPH pipeline on the wedge W_k.

    The total number of sample points is k * n_per_petal + 1 (the +1 is
    the shared basepoint).
    """
    G, heights, _ = build_wedge_graph(cfg)
    barcode = compute_pph(G)
    n_total = cfg.k * cfg.n_per_petal + 1
    return Result(
        n=n_total,
        mod4=n_total % 4,
        height_variant=cfg.height_variant,
        double_edges=cfg.double_edges,
        n_edges=G.number_of_edges(),
        barcode=barcode,
    )


def plot_wedge_graph(
    cfg: WedgeConfig,
    G: nx.DiGraph,
    heights: list[float],
    coords: list[tuple[float, float]],
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the wedge W_k in ℝ².

    Each petal is laid out as a unit circle centred at distance 2 from the
    origin.  Nodes are coloured by height using a diverging colormap; directed
    edges are drawn as arrows; the basepoint (node 0) is marked distinctively.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    h_arr = np.array(heights)
    vmin, vmax = h_arr.min(), h_arr.max()
    norm = mcolors.TwoSlopeNorm(
        vmin=vmin, vcenter=float(np.median(h_arr)), vmax=vmax
    ) if vmin < vmax else mcolors.Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6)
    cmap = cm.RdBu_r

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Draw background petal circles
    for p in range(cfg.k):
        ca = 2 * np.pi * p / cfg.k
        cx, cy = 2.0 * np.cos(ca), 2.0 * np.sin(ca)
        theta_bg = np.linspace(0, 2*np.pi, 200)
        ax.plot(
            cx + np.cos(theta_bg + np.pi + ca),
            cy + np.sin(theta_bg + np.pi + ca),
            color='lightgrey', lw=1, zorder=0,
        )

    # Draw edges
    for u, v, data in G.edges(data=True):
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        ax.annotate(
            '', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8),
            zorder=1,
        )

    # Draw non-basepoint nodes
    non_bp = [i for i in G.nodes if i != 0]
    if non_bp:
        sc = ax.scatter(
            [xs[i] for i in non_bp],
            [ys[i] for i in non_bp],
            c=[heights[i] for i in non_bp],
            cmap=cmap, norm=norm,
            s=60, zorder=3, edgecolors='white', linewidths=0.5,
        )
        plt.colorbar(sc, ax=ax, label='Height')

    # Basepoint
    ax.scatter([xs[0]], [ys[0]],
               c=[heights[0]], cmap=cmap, norm=norm,
               s=200, zorder=4, edgecolors='black', linewidths=1.5,
               marker='*')

    ax.set_aspect('equal')
    ax.set_title(f'Wedge W_{cfg.k}  (n/petal={cfg.n_per_petal}, '
                 f'height={cfg.height_variant})', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def batch_analyze_wedge(
    k_range: range,
    n_per_petal: int,
    **kwargs,
) -> list[Result]:
    """
    Analyze wedge W_k for each k in k_range with fixed n_per_petal.

    Keyword arguments are forwarded to WedgeConfig.  Returns one Result per k.
    """
    return [analyze_wedge(WedgeConfig(k=k, n_per_petal=n_per_petal, **kwargs))
            for k in k_range]


# =============================================================================
# PART 2 — DEFORMED CIRCLES
# =============================================================================

DeformVariant = Literal[
    'circle',
    'ellipse',
    'limacon',
    'cardioid',
    'rose3',
    'rose4',
    'lemniscate_like',
    'squircle',
    'gear',
    'teardrop',
    'epitrochoid',
    'hypotrochoid',
]


@dataclass
class DeformConfig:
    """
    Configuration for a deformed-circle PPH analysis.

    A deformed circle is a topological S¹ embedded in ℝ² via a parametric
    curve other than the standard unit circle.  Height is induced by a
    real-valued function applied to the embedded (x, y) coordinates.
    """
    n: int
    deform: DeformVariant = 'circle'
    deform_params: dict = field(default_factory=dict)
    height_fn: Literal['proj_x', 'proj_y', 'proj_diag', 'radial', 'angle_based'] = 'proj_x'
    proj_angle: float = 0.0          # rotation angle for 'proj_diag'
    sampling: SamplingVariant = 'uniform'
    double_edges: bool = False
    exact_arclength: bool = False    # reserved for future dense arc-length approximation
    random_seed: int | None = None
    jitter_std: float = 0.3
    cluster_std: float = 0.3
    beta_a: float = 0.5
    beta_b: float = 0.5


# ---------------------------------------------------------------------------
# PARAMETRIC CURVES
# ---------------------------------------------------------------------------

def parametric_curve(deform: DeformVariant, params: dict, theta: float) -> tuple[float, float]:
    """
    Return (x, y) on the named parametric curve at parameter angle theta ∈ [0, 2π).

    Each variant uses its standard polar or parametric equations; params
    supplies optional shape parameters (e.g. semi-axes for an ellipse).

    Variants
    --------
    circle        : unit circle
    ellipse       : params a, b (default 1, 0.5)
    limacon       : r = a + b cos θ,  default a=1, b=0.5
    cardioid      : r = a(1 - cos θ), default a=1
    rose3         : r = a cos(3θ),    default a=1  (3-petal rose)
    rose4         : r = a cos(2θ),    default a=1  (4-petal rose)
    lemniscate_like : smooth figure-8 (lemniscate projection)
    squircle      : |x|^p + |y|^p = 1,  default p=4
    gear          : r = 1 + amplitude·cos(teeth·θ), default amplitude=0.1, teeth=8
    teardrop      : teardrop/delta curve
    epitrochoid   : params R, r, d (default 1, 0.5, 0.5)
    hypotrochoid  : params R, r, d (default 1, 0.3, 0.5)
    """
    if deform == 'circle':
        return float(np.cos(theta)), float(np.sin(theta))

    elif deform == 'ellipse':
        a = params.get('a', 1.0)
        b = params.get('b', 0.5)
        return float(a * np.cos(theta)), float(b * np.sin(theta))

    elif deform == 'limacon':
        a = params.get('a', 1.0)
        b = params.get('b', 0.5)
        r = a + b * np.cos(theta)
        return float(r * np.cos(theta)), float(r * np.sin(theta))

    elif deform == 'cardioid':
        a = params.get('a', 1.0)
        r = a * (1.0 - np.cos(theta))
        return float(r * np.cos(theta)), float(r * np.sin(theta))

    elif deform == 'rose3':
        a = params.get('a', 1.0)
        r = a * np.cos(3.0 * theta)
        return float(r * np.cos(theta)), float(r * np.sin(theta))

    elif deform == 'rose4':
        a = params.get('a', 1.0)
        r = a * np.cos(2.0 * theta)
        return float(r * np.cos(theta)), float(r * np.sin(theta))

    elif deform == 'lemniscate_like':
        # Smooth figure-8: x = cos θ / (1 + sin²θ),  y = sin θ cos θ / (1 + sin²θ)
        denom = 1.0 + np.sin(theta) ** 2
        x = np.cos(theta) / denom
        y = np.sin(theta) * np.cos(theta) / denom
        scale = params.get('scale', 1.0)
        return float(scale * x), float(scale * y)

    elif deform == 'squircle':
        p = params.get('p', 4.0)
        exp = 2.0 / p
        c, s = np.cos(theta), np.sin(theta)
        return float(np.sign(c) * abs(c) ** exp), float(np.sign(s) * abs(s) ** exp)

    elif deform == 'gear':
        amplitude = params.get('amplitude', 0.1)
        teeth = int(params.get('teeth', 8))
        r = 1.0 + amplitude * np.cos(teeth * theta)
        return float(r * np.cos(theta)), float(r * np.sin(theta))

    elif deform == 'teardrop':
        # x = (1 - cos θ)/2 - 0.5,  y = sin θ·(1 - cos θ)/2
        x = (1.0 - np.cos(theta)) / 2.0 - 0.5
        y = np.sin(theta) * (1.0 - np.cos(theta)) / 2.0
        return float(x), float(y)

    elif deform == 'epitrochoid':
        R = params.get('R', 1.0)
        r = params.get('r', 0.5)
        d = params.get('d', 0.5)
        x = (R + r) * np.cos(theta) - d * np.cos((R + r) * theta / r)
        y = (R + r) * np.sin(theta) - d * np.sin((R + r) * theta / r)
        return float(x), float(y)

    elif deform == 'hypotrochoid':
        R = params.get('R', 1.0)
        r = params.get('r', 0.3)
        d = params.get('d', 0.5)
        x = (R - r) * np.cos(theta) + d * np.cos((R - r) * theta / r)
        y = (R - r) * np.sin(theta) - d * np.sin((R - r) * theta / r)
        return float(x), float(y)

    else:
        raise ValueError(f"Unknown deform variant: {deform!r}")


def _deform_height(x: float, y: float, height_fn: str, proj_angle: float = 0.0) -> float:
    """Compute scalar height from (x, y) according to the chosen height_fn."""
    if height_fn == 'proj_x':
        return x
    elif height_fn == 'proj_y':
        return y
    elif height_fn == 'proj_diag':
        return x * np.cos(proj_angle) + y * np.sin(proj_angle)
    elif height_fn == 'radial':
        return float(np.hypot(x, y))
    elif height_fn == 'angle_based':
        return float(np.arctan2(y, x))
    else:
        raise ValueError(f"Unknown height_fn: {height_fn!r}")


def build_deform_graph(
    cfg: DeformConfig,
) -> tuple[nx.DiGraph, list[float], list[tuple[float, float]]]:
    """
    Build the directed graph for a deformed circle embedded in ℝ².

    Samples n angles, maps them to (x, y) via parametric_curve, assigns
    scalar heights via cfg.height_fn, then adds directed edges (lower →
    higher height) for all pairs within n//2 cyclic steps, weighted by the
    chord distance ‖p_i − p_j‖ in the parametric embedding.

    Returns
    -------
    G       : nx.DiGraph
    heights : list[float] — one per node
    points  : list[(x,y)] — parametric positions
    """
    # Reuse sample_angles via a Config adapter
    angle_cfg = Config(
        n=cfg.n,
        sampling=cfg.sampling,
        random_seed=cfg.random_seed,
        jitter_std=cfg.jitter_std,
        cluster_std=cfg.cluster_std,
        beta_a=cfg.beta_a,
        beta_b=cfg.beta_b,
    )
    angles = sample_angles(angle_cfg)

    points: list[tuple[float, float]] = [
        parametric_curve(cfg.deform, cfg.deform_params, a) for a in angles
    ]
    pts_arr = np.array(points)  # (n, 2)

    heights: list[float] = [
        _deform_height(x, y, cfg.height_fn, cfg.proj_angle) for x, y in points
    ]

    G = nx.DiGraph()
    for i in range(cfg.n):
        G.add_node(i, height=heights[i], x=points[i][0], y=points[i][1])

    for i in range(cfg.n):
        for skip in range(1, cfg.n // 2 + 1):
            j = (i + skip) % cfg.n
            hi, hj = heights[i], heights[j]
            w = float(np.linalg.norm(pts_arr[i] - pts_arr[j]))

            if abs(hi - hj) < 1e-10:
                if cfg.double_edges:
                    G.add_edge(i, j, weight=w)
                    G.add_edge(j, i, weight=w)
            elif hi < hj:
                G.add_edge(i, j, weight=w)
            else:
                G.add_edge(j, i, weight=w)

    return G, heights, points


def analyze_deform(cfg: DeformConfig) -> Result:
    """
    Run the full PPH pipeline on a deformed-circle configuration.

    The height_variant field in the returned Result is set to the string
    '{deform}/{height_fn}' for identification in downstream analysis.
    """
    G, _, _ = build_deform_graph(cfg)
    barcode = compute_pph(G)
    return Result(
        n=cfg.n,
        mod4=cfg.n % 4,
        height_variant=f"{cfg.deform}/{cfg.height_fn}",   # type: ignore[arg-type]
        double_edges=cfg.double_edges,
        n_edges=G.number_of_edges(),
        barcode=barcode,
    )


def plot_deform_graph(
    cfg: DeformConfig,
    G: nx.DiGraph,
    coords: list[tuple[float, float]],
    heights: list[float],
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the deformed circle in ℝ².

    A background polyline traces the full parametric curve; nodes are
    coloured by height (RdBu_r diverging map); directed edges are drawn as
    arrows offset slightly outward from the curve centroid so overlapping
    forward/backward arrows remain readable.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Background curve
    theta_bg = np.linspace(0, 2 * np.pi, 300)
    bg = np.array([parametric_curve(cfg.deform, cfg.deform_params, t) for t in theta_bg])
    ax.plot(bg[:, 0], bg[:, 1], color='#cccccc', lw=1.5, zorder=0)

    pts = np.array(coords)
    centroid = pts.mean(axis=0)

    h_arr = np.array(heights)
    vmin, vmax = h_arr.min(), h_arr.max()
    spread = vmax - vmin if vmax - vmin > 1e-12 else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdBu_r

    offset_scale = spread * 0.03

    # Edges
    for u, v in G.edges():
        p0, p1 = pts[u], pts[v]
        mid = 0.5 * (p0 + p1)
        outward = mid - centroid
        n_ = np.linalg.norm(outward)
        if n_ > 1e-9:
            outward = outward / n_
        shift = outward * offset_scale
        ax.annotate('', xy=p1 + shift, xytext=p0 + shift,
                    arrowprops=dict(arrowstyle='->', color='#777777', lw=0.7),
                    zorder=2)

    # Nodes
    sc = ax.scatter(pts[:, 0], pts[:, 1],
                    c=heights, cmap=cmap, norm=norm,
                    s=70, zorder=3, edgecolors='black', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Height')

    ax.set_aspect('equal')
    ax.set_title(f'Deformed circle: {cfg.deform!r}  |  height_fn={cfg.height_fn!r}\n'
                 f'n={cfg.n}, sampling={cfg.sampling!r}', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


# =============================================================================
# PART 4 — DEFORMED WEDGE
# =============================================================================

import math


@dataclass
class DeformedWedgeConfig:
    """
    Configuration for a wedge of k deformed circles sharing a single basepoint.

    The deformed wedge W_k(γ₁, …, γ_k) is the 1-complex obtained by
    identifying k parametric closed curves γ_p : S¹ → ℝ² at their common
    image of θ = 0.  Each petal p is described by a DeformConfig specifying
    its curve shape and height function.  The total node count is
    k * n_per_petal + 1.

    Parameters
    ----------
    k               : number of petals
    n_per_petal     : interior sample points per petal (basepoint excluded)
    petals          : length-k list of DeformConfig; only deform, deform_params,
                      height_fn, proj_angle, exact_arclength are read per petal
    double_edges    : add both directions for equal-height pairs
    sampling        : angle-sampling strategy (applied identically to all petals)
    layout          : visual arrangement of petal clouds; 'radial'/'star',
                      'linear', or 'custom'
    layout_radius   : separation distance between petal centres
    custom_offsets  : (dx, dy) per petal when layout='custom'
    basepoint_height: override for node-0 height; None → average over petals
    """
    k: int
    n_per_petal: int
    petals: list[DeformConfig]
    double_edges: bool = False
    sampling: SamplingVariant = 'uniform'
    layout: Literal['radial', 'linear', 'star', 'custom'] = 'radial'
    layout_radius: float = 2.5
    custom_offsets: list[tuple[float, float]] | None = None
    basepoint_height: float | None = None
    random_seed: int | None = None
    jitter_std: float = 0.3
    cluster_std: float = 0.3
    beta_a: float = 0.5
    beta_b: float = 0.5

    @classmethod
    def uniform(
        cls,
        k: int,
        n_per_petal: int,
        deform: DeformVariant,
        deform_params: dict,
        height_fn: str,
        **kwargs,
    ) -> "DeformedWedgeConfig":
        """
        Build a homogeneous deformed wedge where every petal uses the same
        curve, differentiated only by a proj_angle that rotates by 2π/k
        between petals so that height gradients are not identical.
        """
        petals = [
            DeformConfig(
                n=n_per_petal,
                deform=deform,
                deform_params=deform_params,
                height_fn=height_fn,  # type: ignore[arg-type]
                proj_angle=2.0 * math.pi * p / k,
            )
            for p in range(k)
        ]
        return cls(k=k, n_per_petal=n_per_petal, petals=petals, **kwargs)


def _dw_layout_offset(cfg: "DeformedWedgeConfig", p: int) -> tuple[float, float]:
    """Return the (dx, dy) visual layout offset for petal p."""
    if cfg.layout in ('radial', 'star'):
        angle = 2.0 * math.pi * p / cfg.k
        return cfg.layout_radius * math.cos(angle), cfg.layout_radius * math.sin(angle)
    elif cfg.layout == 'linear':
        return float(p) * cfg.layout_radius, 0.0
    elif cfg.layout == 'custom':
        if cfg.custom_offsets is None or len(cfg.custom_offsets) < cfg.k:
            raise ValueError("custom_offsets must have length k when layout='custom'")
        return cfg.custom_offsets[p]
    else:
        raise ValueError(f"Unknown layout: {cfg.layout!r}")


def build_deformed_wedge_graph(
    cfg: DeformedWedgeConfig,
) -> tuple[nx.DiGraph, list[float], list[tuple[float, float]], list[int]]:
    """
    Build the directed graph for the deformed wedge W_k(γ₁, …, γ_k).

    Node 0 is the basepoint at the origin.  For petal p, n_per_petal angles
    are sampled from (0, 2π); each maps to (x, y) via parametric_curve,
    then translated so θ=0 lands at origin.  Arc-length weights are chord
    distances in the UN-translated coordinates.  Directed edges (lower →
    higher height) are added for consecutive pairs and skip edges within each
    petal, plus basepoint connections.  No cross-petal edges except through
    node 0.

    Returns (G, heights, coords, petal_ids) where petal_ids[0] = -1.
    """
    angle_cfg = Config(
        n=cfg.n_per_petal,
        sampling=cfg.sampling,
        random_seed=cfg.random_seed,
        jitter_std=cfg.jitter_std,
        cluster_std=cfg.cluster_std,
        beta_a=cfg.beta_a,
        beta_b=cfg.beta_b,
    )
    raw_angles = sample_angles(angle_cfg)
    # Ensure open interval (0, 2π)
    petal_angles: list[float] = []
    for a in raw_angles:
        if a <= 0.0:
            a = 1e-9
        elif a >= 2 * math.pi - 1e-9:
            a = 2 * math.pi - 1e-9
        petal_angles.append(float(a))

    # Basepoint height
    if cfg.basepoint_height is not None:
        bp_height = float(cfg.basepoint_height)
    else:
        bp_hs = []
        for pc in cfg.petals:
            x0, y0 = parametric_curve(pc.deform, pc.deform_params, 0.0)
            bp_hs.append(_deform_height(x0, y0, pc.height_fn, pc.proj_angle))
        bp_height = float(np.mean(bp_hs)) if bp_hs else 0.0

    G = nx.DiGraph()
    heights: list[float] = [bp_height]
    coords: list[tuple[float, float]] = [(0.0, 0.0)]
    petal_ids: list[int] = [-1]
    G.add_node(0, height=bp_height, petal=-1)

    def _add_edge(u: int, v: int, w: float) -> None:
        hu, hv = heights[u], heights[v]
        if abs(hu - hv) < 1e-10:
            if cfg.double_edges:
                G.add_edge(u, v, weight=w)
                G.add_edge(v, u, weight=w)
        elif hu < hv:
            G.add_edge(u, v, weight=w)
        else:
            G.add_edge(v, u, weight=w)

    petal_node_lists: list[list[int]] = []
    petal_raw_origin: list[tuple[float, float]] = []
    petal_raw_pts: list[list[tuple[float, float]]] = []

    for p, pc in enumerate(cfg.petals):
        dx, dy = _dw_layout_offset(cfg, p)
        x0r, y0r = parametric_curve(pc.deform, pc.deform_params, 0.0)
        petal_raw_origin.append((x0r, y0r))

        node_list: list[int] = []
        raw_pts: list[tuple[float, float]] = []

        for q, ang in enumerate(petal_angles):
            node_id = 1 + p * cfg.n_per_petal + q
            xr, yr = parametric_curve(pc.deform, pc.deform_params, ang)
            raw_pts.append((xr, yr))
            xt = (xr - x0r) + dx
            yt = (yr - y0r) + dy
            h = _deform_height(xr, yr, pc.height_fn, pc.proj_angle)
            G.add_node(node_id, height=h, petal=p)
            heights.append(h)
            coords.append((xt, yt))
            petal_ids.append(p)
            node_list.append(node_id)

        petal_node_lists.append(node_list)
        petal_raw_pts.append(raw_pts)

    # Edges
    for p in range(cfg.k):
        pc = cfg.petals[p]
        x0r, y0r = petal_raw_origin[p]
        node_list = petal_node_lists[p]
        raw_pts = petal_raw_pts[p]
        n = cfg.n_per_petal

        full_raw = [(x0r, y0r)] + raw_pts + [(x0r, y0r)]
        full_nodes = [0] + node_list + [0]

        # Consecutive edges (includes basepoint ↔ first/last)
        for i in range(len(full_nodes) - 1):
            u, v = full_nodes[i], full_nodes[i + 1]
            w = float(np.linalg.norm(
                np.array(full_raw[i]) - np.array(full_raw[i + 1])
            ))
            _add_edge(u, v, w)

        # Skip edges within the petal ring
        ring_nodes = [0] + node_list
        ring_raw = [(x0r, y0r)] + raw_pts
        ring_len = len(ring_nodes)
        for start in range(ring_len):
            for skip in range(2, n // 2 + 1):
                end = (start + skip) % ring_len
                u, v = ring_nodes[start], ring_nodes[end]
                w = float(np.linalg.norm(
                    np.array(ring_raw[start]) - np.array(ring_raw[end])
                ))
                _add_edge(u, v, w)

    return G, heights, coords, petal_ids


def analyze_deformed_wedge(cfg: DeformedWedgeConfig) -> Result:
    """
    Run the full PPH pipeline on the deformed wedge W_k(γ₁, …, γ_k).

    Total nodes = k * n_per_petal + 1.  height_variant in the returned
    Result encodes petal shapes as '|'-separated 'deform/height_fn' tokens.
    """
    G, _, _, _ = build_deformed_wedge_graph(cfg)
    barcode = compute_pph(G)
    n_total = cfg.k * cfg.n_per_petal + 1
    hv_str = "|".join(f"{pc.deform}/{pc.height_fn}" for pc in cfg.petals)
    return Result(
        n=n_total,
        mod4=n_total % 4,
        height_variant=hv_str,  # type: ignore[arg-type]
        double_edges=cfg.double_edges,
        n_edges=G.number_of_edges(),
        barcode=barcode,
    )


def plot_deformed_wedge_graph(
    cfg: DeformedWedgeConfig,
    G: nx.DiGraph,
    coords: list[tuple[float, float]],
    heights: list[float],
    petal_ids: list[int],
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Draw the deformed wedge in ℝ².

    Each petal's background curve is drawn in a distinct colour; nodes are
    coloured by height (RdBu_r, shared scale); edges incident to the
    basepoint are drawn thicker; node 0 is a large star with a black border.
    Legend maps petal index to deform variant and height_fn.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    def petal_col(p: int) -> str:
        return cycle_colors[p % len(cycle_colors)]

    theta_bg = np.linspace(0, 2 * math.pi, 300)

    # Background curves
    for p, pc in enumerate(cfg.petals):
        dx, dy = _dw_layout_offset(cfg, p)
        x0r, y0r = parametric_curve(pc.deform, pc.deform_params, 0.0)
        bg = np.array([parametric_curve(pc.deform, pc.deform_params, t) for t in theta_bg])
        ax.plot(bg[:, 0] - x0r + dx, bg[:, 1] - y0r + dy,
                color=petal_col(p), lw=1.2, alpha=0.35, zorder=0,
                label=f'petal {p}: {pc.deform!r}/{pc.height_fn!r}')

    h_arr = np.array(heights)
    vmin, vmax = float(h_arr.min()), float(h_arr.max())
    if vmax - vmin < 1e-12:
        norm: mcolors.Normalize = mcolors.Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6)
    else:
        vc = float(np.median(h_arr))
        vc = max(vmin + 1e-9, min(vc, vmax - 1e-9))
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vc, vmax=vmax)
    cmap = cm.RdBu_r

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    # Edges
    for u, v in G.edges():
        incident = (u == 0 or v == 0)
        lw = 1.5 if incident else 0.6
        col = '#222222' if incident else '#999999'
        ax.annotate('', xy=(xs[v], ys[v]), xytext=(xs[u], ys[u]),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw), zorder=1)

    # Non-basepoint nodes
    non_bp = [i for i in G.nodes if i != 0]
    if non_bp:
        sc = ax.scatter([xs[i] for i in non_bp], [ys[i] for i in non_bp],
                        c=[heights[i] for i in non_bp], cmap=cmap, norm=norm,
                        s=60, zorder=3, edgecolors='white', linewidths=0.4)
        plt.colorbar(sc, ax=ax, label='Height', fraction=0.03, pad=0.04)

    # Basepoint
    ax.scatter([xs[0]], [ys[0]], c=[heights[0]], cmap=cmap, norm=norm,
               s=250, zorder=4, marker='*', edgecolors='black', linewidths=1.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=7, framealpha=0.8, title='Petals')

    ax.set_aspect('equal')
    ax.set_title(f'Deformed Wedge  k={cfg.k}, n/petal={cfg.n_per_petal}, '
                 f'layout={cfg.layout!r}', fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    return fig


def batch_analyze_deformed_wedge(
    k_range: range,
    n_per_petal: int,
    deform: DeformVariant,
    deform_params: dict | None = None,
    height_fn: str = 'proj_x',
    **kwargs,
) -> list[Result]:
    """
    Analyze the homogeneous deformed wedge W_k for each k in k_range.

    Uses DeformedWedgeConfig.uniform so each petal has the same shape with
    proj_angle rotating by 2π/k.

    Parameters
    ----------
    k_range       : iterable of k values
    n_per_petal   : interior nodes per petal (fixed)
    deform        : curve variant shared by all petals
    deform_params : shape parameters forwarded to parametric_curve
    height_fn     : height function for all petals
    **kwargs      : forwarded to DeformedWedgeConfig.uniform
    """
    if deform_params is None:
        deform_params = {}
    return [
        analyze_deformed_wedge(
            DeformedWedgeConfig.uniform(
                k=k, n_per_petal=n_per_petal,
                deform=deform, deform_params=deform_params,
                height_fn=height_fn, **kwargs,
            )
        )
        for k in k_range
    ]


def plot_heterogeneous_comparison(
    n_per_petal: int,
    petal_specs: list[tuple[DeformVariant, dict, str]],
    n_range: range,
    output: str | Path | None = None,
) -> plt.Figure:
    """
    Compare PPH barcodes of a heterogeneous deformed wedge as n_per_petal varies.

    petal_specs is a list of (deform_variant, deform_params, height_fn) tuples.
    For each n in n_range a fresh wedge is built with those petals.

    Figure layout
    -------------
    Top-left  : graph layout for the smallest n in n_range
    Top-right : graph layout for the largest n in n_range
    Bottom    : one axis per petal type — death times vs n, coloured by n%4
    """
    k = len(petal_specs)
    if k == 0:
        raise ValueError("petal_specs must be non-empty")
    n_list = list(n_range)
    if not n_list:
        raise ValueError("n_range must contain at least one value")

    cycle_colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    def pc_(p: int) -> str:
        return cycle_colors[p % len(cycle_colors)]

    # Collect results + graph data
    entries = []
    for n_val in n_list:
        petals = [
            DeformConfig(n=n_val, deform=dv, deform_params=dp,
                         height_fn=hf)  # type: ignore[arg-type]
            for dv, dp, hf in petal_specs
        ]
        cfg = DeformedWedgeConfig(k=k, n_per_petal=n_val, petals=petals)
        G, heights, coords, pids = build_deformed_wedge_graph(cfg)
        barcode = compute_pph(G)
        n_total = k * n_val + 1
        result = Result(
            n=n_total, mod4=n_total % 4,
            height_variant="|".join(f"{dv}/{hf}" for dv, _, hf in petal_specs),  # type: ignore
            double_edges=False, n_edges=G.number_of_edges(), barcode=barcode,
        )
        entries.append((n_val, result, cfg, G, heights, coords, pids))

    n_bottom = max(k, 1)
    fig = plt.figure(figsize=(max(10, 4 * n_bottom), 10))
    gs = fig.add_gridspec(2, max(n_bottom, 2), hspace=0.45, wspace=0.35)
    ax_small = fig.add_subplot(gs[0, 0])
    ax_large = fig.add_subplot(gs[0, -1])

    def _draw_layout(ax: plt.Axes, idx: int, title: str) -> None:
        _, _, cfg_l, G_l, heights_l, coords_l, _ = entries[idx]
        h_a = np.array(heights_l)
        vmin_l, vmax_l = float(h_a.min()), float(h_a.max())
        norm_l = mcolors.Normalize(
            vmin=vmin_l - 1e-6 if vmax_l - vmin_l < 1e-12 else vmin_l,
            vmax=vmax_l + 1e-6 if vmax_l - vmin_l < 1e-12 else vmax_l,
        )
        xs_l = [c[0] for c in coords_l]
        ys_l = [c[1] for c in coords_l]
        for u, v in G_l.edges():
            inc = (u == 0 or v == 0)
            ax.annotate('', xy=(xs_l[v], ys_l[v]), xytext=(xs_l[u], ys_l[u]),
                        arrowprops=dict(arrowstyle='->', color='#444' if inc else '#bbb',
                                        lw=1.2 if inc else 0.5), zorder=1)
        non_bp = [i for i in G_l.nodes if i != 0]
        if non_bp:
            ax.scatter([xs_l[i] for i in non_bp], [ys_l[i] for i in non_bp],
                       c=[heights_l[i] for i in non_bp], cmap=cm.RdBu_r, norm=norm_l,
                       s=20, zorder=3, edgecolors='none')
        ax.scatter([xs_l[0]], [ys_l[0]], c=[heights_l[0]], cmap=cm.RdBu_r, norm=norm_l,
                   s=120, zorder=4, marker='*', edgecolors='black', linewidths=1.0)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=9, fontweight='bold')

    _draw_layout(ax_small, 0, f'n/petal = {n_list[0]}')
    _draw_layout(ax_large, len(entries) - 1, f'n/petal = {n_list[-1]}')

    # Bottom: death times per petal type
    for p in range(k):
        dv, _, hf = petal_specs[p]
        ax_bt = fig.add_subplot(gs[1, p % max(n_bottom, 2)])
        for i, n_val in enumerate(n_list):
            _, result_i, *_ = entries[i]
            finite = [d for _, d in result_i.barcode if np.isfinite(d)]
            petal_d = [finite[j] for j in range(len(finite)) if j % k == p]
            col = MOD4_COLORS[n_val % 4]
            if petal_d:
                ax_bt.scatter([n_val] * len(petal_d), petal_d,
                              color=col, s=25, alpha=0.7, zorder=2)
        ax_bt.set_xlabel('n per petal', fontsize=8)
        ax_bt.set_ylabel('Death time', fontsize=8)
        ax_bt.set_title(f'Petal {p}: {dv!r}\n{hf!r}', fontsize=8, fontweight='bold')
        ax_bt.tick_params(labelsize=7)
        ax_bt.grid(True, alpha=0.3)

    fig.suptitle(
        f'Heterogeneous Deformed Wedge  k={k}  '
        f'(n_per_petal sweep {n_list[0]}–{n_list[-1]})',
        fontsize=11, fontweight='bold',
    )
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='white')
    return fig


# =============================================================================
# PART 3 — BATCH COMPARISON PLOTS & CLI
# =============================================================================

import argparse


def plot_wedge_death_times(
    k_range: range,
    n_per_petal_values: list[int],
    height_variant: HeightVariant = 'standard',
    **kwargs,
) -> plt.Figure:
    """
    Plot PPH death times for the wedge W_k as k varies, one subplot per
    n_per_petal value.

    x-axis: k (number of petals); y-axis: death times in units of π.
    Points are coloured by (k * n_per_petal + 1) % 4 using MOD4_COLORS.
    A horizontal dashed line marks the π/2 reference.

    Parameters
    ----------
    k_range           : range of k values to sweep
    n_per_petal_values: list of n_per_petal values (one subplot each)
    height_variant    : height function applied to all petals
    **kwargs          : forwarded to WedgeConfig (e.g. sampling, double_edges)
    """
    n_panels = len(n_per_petal_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), squeeze=False)

    for ax, npp in zip(axes[0], n_per_petal_values):
        for k in k_range:
            cfg = WedgeConfig(k=k, n_per_petal=npp,
                              height_variant=height_variant, **kwargs)
            result = analyze_wedge(cfg)
            col = MOD4_COLORS[(k * npp + 1) % 4]
            for bar in result.barcode:
                death = bar[1] / np.pi if np.isfinite(bar[1]) else 1.05
                ax.scatter(k, death, color=col, s=50, alpha=0.8, zorder=3)

        ax.axhline(0.5, color='black', ls='--', lw=1.5, label='π/2 limit')
        ax.set_xlabel('k (number of petals)')
        ax.set_ylabel('Death time (units of π)')
        ax.set_title(f'Wedge W_k PPH — n_per_petal={npp}', fontweight='bold')
        ax.set_xticks(list(k_range))
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=MOD4_COLORS[i], markersize=9,
                              label=f'(k·n+1) ≡ {i} (mod 4)') for i in range(4)]
        ax.legend(handles=handles, loc='upper right', fontsize=7)

    plt.suptitle(f'Wedge PPH death times  |  height={height_variant!r}',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_deform_comparison(
    deform_variants: list[str],
    height_fns: list[str],
    n_range: range,
    double_edges: bool = False,
) -> plt.Figure:
    """
    Grid of (len(deform_variants) × len(height_fns)) subplots comparing PPH
    death times for deformed circles.

    Each panel: death times vs n, scatter coloured by n % 4 via MOD4_COLORS.
    """
    n_rows, n_cols = len(deform_variants), len(height_fns)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    for row, deform in enumerate(deform_variants):
        for col, hfn in enumerate(height_fns):
            ax = axes[row][col]
            for n in n_range:
                cfg = DeformConfig(n=n, deform=deform,  # type: ignore[arg-type]
                                   height_fn=hfn, double_edges=double_edges)  # type: ignore[arg-type]
                result = analyze_deform(cfg)
                col_ = MOD4_COLORS[n % 4]
                for bar in result.barcode:
                    death = bar[1] / np.pi if np.isfinite(bar[1]) else 1.05
                    ax.scatter(n, death, color=col_, s=35, alpha=0.75, zorder=3)
            ax.axhline(0.5, color='black', ls='--', lw=1.2)
            ax.set_title(f'{deform} / {hfn}', fontweight='bold', fontsize=9)
            ax.set_xlabel('n'); ax.set_ylabel('Death (×π)')
            ax.set_ylim(0, 1.15); ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=MOD4_COLORS[i], markersize=8,
                          label=f'n ≡ {i} (mod 4)') for i in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    plt.suptitle('Deformed-circle PPH death times', fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_deformed_wedge_comparison(
    k_values: list[int],
    deform_variants: list[str],
    n_per_petal: int = 8,
    height_fn: str = 'proj_x',
) -> plt.Figure:
    """
    Grid of (len(k_values) × len(deform_variants)) subplots.

    Each panel sweeps n_per_petal from 4..12 and plots death times vs
    n = k * n_per_petal + 1, coloured by n % 4.
    """
    n_rows, n_cols = len(k_values), len(deform_variants)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    for row, k in enumerate(k_values):
        for col, deform in enumerate(deform_variants):
            ax = axes[row][col]
            for npp in range(4, 13):
                n_total = k * npp + 1
                cfg = DeformedWedgeConfig.uniform(
                    k=k, n_per_petal=npp,
                    deform=deform, deform_params={},  # type: ignore[arg-type]
                    height_fn=height_fn,
                )
                result = analyze_deformed_wedge(cfg)
                col_ = MOD4_COLORS[n_total % 4]
                for bar in result.barcode:
                    death = bar[1] / np.pi if np.isfinite(bar[1]) else 1.05
                    ax.scatter(n_total, death, color=col_, s=35, alpha=0.75, zorder=3)
            ax.axhline(0.5, color='black', ls='--', lw=1.2)
            ax.set_title(f'k={k}, {deform}', fontweight='bold', fontsize=9)
            ax.set_xlabel('n = k·n_per_petal + 1'); ax.set_ylabel('Death (×π)')
            ax.set_ylim(0, 1.15); ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=MOD4_COLORS[i], markersize=8,
                          label=f'n ≡ {i} (mod 4)') for i in range(4)]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=8)
    plt.suptitle(f'Deformed-wedge PPH  |  height_fn={height_fn!r}',
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def _print_barcode(barcode: list[tuple[float, float]], label: str = '') -> None:
    """Print a human-readable barcode summary."""
    if label:
        print(f"\n{'─'*55}\n  Barcode: {label}\n{'─'*55}")
    if not barcode:
        print("  (empty barcode)")
        return
    for birth, death in barcode:
        if np.isfinite(death):
            print(f"  [{birth/np.pi:.4f}π,  {death/np.pi:.4f}π]  "
                  f"(length {(death-birth)/np.pi:.4f}π)")
        else:
            print(f"  [{birth/np.pi:.4f}π,  ∞)")


def main() -> None:
    """CLI entry point for shape_pph.py."""
    parser = argparse.ArgumentParser(
        description='PPH analysis on wedges and deformed circles.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  wedge                  Single wedge W_k (graph + barcode)
  deform                 Single deformed circle (graph + barcode)
  compare-wedge          Sweep k=1..5, n_per_petal in [4,6,8,12]
  compare-deform         circle/ellipse/limacon × proj_x/proj_y
  deformed-wedge         Single homogeneous deformed wedge W_k
  deformed-wedge-hetero  Heterogeneous deformed wedge (mixed petals)
  compare-deformed-wedge k=1..4 × {circle,ellipse,cardioid,limacon}
""",
    )
    parser.add_argument('--mode', default='wedge',
        choices=['wedge', 'deform', 'compare-wedge', 'compare-deform',
                 'deformed-wedge', 'deformed-wedge-hetero',
                 'compare-deformed-wedge'])
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n-per-petal', type=int, default=8)
    parser.add_argument('--n', type=int, default=20,
                        help='Sample points for --mode deform')
    parser.add_argument('--deform', default='ellipse',
                        help='Deformation variant')
    parser.add_argument('--height', default='standard',
                        help='Height variant (wedge) or height_fn (deform modes)')
    parser.add_argument('--sampling', default='uniform')
    parser.add_argument('--double-edges', action='store_true')
    parser.add_argument('--petals', nargs='+', default=['ellipse:proj_x'],
                        metavar='DEFORM:HEIGHT_FN',
                        help='Heterogeneous petal specs "deform:height_fn"')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--no-show', action='store_true')

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode

    # ── wedge ─────────────────────────────────────────────────────────────────
    if mode == 'wedge':
        cfg = WedgeConfig(k=args.k, n_per_petal=args.n_per_petal,
                          height_variant=args.height,  # type: ignore[arg-type]
                          sampling=args.sampling,  # type: ignore[arg-type]
                          double_edges=args.double_edges)
        G, heights, coords = build_wedge_graph(cfg)
        result = analyze_wedge(cfg)
        plot_wedge_graph(cfg, G, heights, coords,
                         output=args.output_dir / 'wedge_graph.png')
        _print_barcode(result.barcode,
                       label=f'W_{args.k}, n_per_petal={args.n_per_petal}')

    # ── deform ────────────────────────────────────────────────────────────────
    elif mode == 'deform':
        hfn = args.height if args.height != 'standard' else 'proj_x'
        cfg = DeformConfig(n=args.n, deform=args.deform,  # type: ignore[arg-type]
                           height_fn=hfn,  # type: ignore[arg-type]
                           sampling=args.sampling,  # type: ignore[arg-type]
                           double_edges=args.double_edges)
        G, heights, coords = build_deform_graph(cfg)
        result = analyze_deform(cfg)
        plot_deform_graph(cfg, G, coords, heights,
                          output=args.output_dir / 'deform_graph.png')
        _print_barcode(result.barcode,
                       label=f'{args.deform!r}, n={args.n}, height_fn={hfn!r}')

    # ── compare-wedge ─────────────────────────────────────────────────────────
    elif mode == 'compare-wedge':
        hv = args.height if args.height != 'proj_x' else 'standard'
        fig = plot_wedge_death_times(
            k_range=range(1, 6),
            n_per_petal_values=[4, 6, 8, 12],
            height_variant=hv,  # type: ignore[arg-type]
        )
        out = args.output_dir / 'compare_wedge.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    # ── compare-deform ────────────────────────────────────────────────────────
    elif mode == 'compare-deform':
        fig = plot_deform_comparison(
            deform_variants=['circle', 'ellipse', 'limacon'],
            height_fns=['proj_x', 'proj_y'],
            n_range=range(6, 25),
            double_edges=args.double_edges,
        )
        out = args.output_dir / 'compare_deform.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    # ── deformed-wedge ────────────────────────────────────────────────────────
    elif mode == 'deformed-wedge':
        hfn = args.height if args.height != 'standard' else 'proj_x'
        cfg = DeformedWedgeConfig.uniform(
            k=args.k, n_per_petal=args.n_per_petal,
            deform=args.deform, deform_params={},  # type: ignore[arg-type]
            height_fn=hfn,
            sampling=args.sampling,  # type: ignore[arg-type]
            double_edges=args.double_edges,
        )
        G, heights, coords, pids = build_deformed_wedge_graph(cfg)
        result = analyze_deformed_wedge(cfg)
        plot_deformed_wedge_graph(cfg, G, coords, heights, pids,
                                  output=args.output_dir / 'deformed_wedge.png')
        _print_barcode(result.barcode,
                       label=f'DeformedWedge k={args.k}, '
                             f'deform={args.deform!r}, height_fn={hfn!r}')

    # ── deformed-wedge-hetero ─────────────────────────────────────────────────
    elif mode == 'deformed-wedge-hetero':
        petal_specs: list[tuple[DeformVariant, dict, str]] = []
        for spec in args.petals:
            parts = spec.split(':', 1)
            if len(parts) != 2:
                parser.error(f'--petals entries must be "deform:height_fn", got {spec!r}')
            petal_specs.append((parts[0], {}, parts[1]))  # type: ignore[arg-type]
        fig = plot_heterogeneous_comparison(
            n_per_petal=args.n_per_petal,
            petal_specs=petal_specs,
            n_range=range(4, args.n_per_petal + 5),
            output=args.output_dir / 'deformed_wedge_hetero.png',
        )
        print(f'Saved: {args.output_dir / "deformed_wedge_hetero.png"}')

    # ── compare-deformed-wedge ────────────────────────────────────────────────
    elif mode == 'compare-deformed-wedge':
        hfn = args.height if args.height != 'standard' else 'proj_x'
        fig = plot_deformed_wedge_comparison(
            k_values=list(range(1, 5)),
            deform_variants=['circle', 'ellipse', 'cardioid', 'limacon'],
            n_per_petal=args.n_per_petal,
            height_fn=hfn,
        )
        out = args.output_dir / 'compare_deformed_wedge.png'
        fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
