"""
Wedge PPH Explorer — Streamlit app
====================================
Run with:
    streamlit run wedge_app.py

Features:
- Graph Explorer, Edge Comparison, Height Comparison, PPH Barcode pages
- Arrow or line edges
- Hoverable edge midpoints (with vertex info) and nodes
- Edge set diff with orientation on Height/Edge Comparison
- Height function plot
- π/2 reference lines on persistence diagrams
- Legend outside plot area
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from collections import Counter

from wedge_pph import (Config, build_graph, analyze, _height_for,
                       HeightVariant, DistanceVariant, _GLOBAL_VARIANTS)
import typing

_hv_args = typing.get_args(HeightVariant)
HEIGHT_OPTIONS = list(_hv_args) if _hv_args else [
    'standard', 'cos', 'sin2', 'cos2', 'abs_sin', 'triangle', 'sawtooth',
    'square', 'x_coord', 'y_coord', 'radial_dist', 'angle_from_top',
    'dist_from_right', 'dip', 'double_max', 'plateau', 'squeeze',
    'shallow', 'damped', 'constant',
    'global_x', 'global_y', 'global_r', 'global_angle',
    'global_sawtooth_x', 'global_sin_x', 'global_sin_y', 'global_sin_r',
    'global_diagonal', 'global_saddle', 'global_ripple', 'global_dist_max',
]

HEIGHT_DESCRIPTIONS = {
    'standard':          'standard — sin(θ), local angle',
    'cos':               'cos — cos(θ), local angle',
    'sin2':              'sin2 — sin(2θ), two peaks',
    'cos2':              'cos2 — cos(2θ), two peaks shifted',
    'abs_sin':           'abs_sin — |sin(θ)|, always positive',
    'triangle':          'triangle — piecewise linear wave',
    'sawtooth':          'sawtooth — drops linearly over [0,2π)',
    'square':            'square — ±1 step function',
    'x_coord':           'x_coord — cos(θ), local x projection',
    'y_coord':           'y_coord — sin(θ), local y projection',
    'radial_dist':       'radial_dist — neg. distance from glue point',
    'angle_from_top':    'angle_from_top — closeness to circle top',
    'dist_from_right':   'dist_from_right — neg. arc from θ=0',
    'dip':               'dip — sin with dip near peak',
    'double_max':        'double_max — two local maxima',
    'plateau':           'plateau — sin with flat region',
    'squeeze':           'squeeze — compressed left half',
    'shallow':           'shallow — amplitude-modulated sine',
    'damped':            'damped — decaying oscillation',
    'constant':          'constant — always 0 (all ties)',
    'global_x':          '★ global_x — x coord, continuous across gluing',
    'global_y':          '★ global_y — y coord, continuous across gluing',
    'global_r':          '★ global_r — distance from origin',
    'global_angle':      '★ global_angle — polar angle / π',
    'global_sawtooth_x': '★ global_sawtooth_x — sawtooth along x-axis',
    'global_sin_x':      '★ global_sin_x — sin wave along x-axis',
    'global_sin_y':      '★ global_sin_y — sin wave along y-axis',
    'global_sin_r':      '★ global_sin_r — concentric sine rings',
    'global_diagonal':   '★ global_diagonal — (x+y)/√2 ramp',
    'global_saddle':     '★ global_saddle — x²-y² saddle at origin',
    'global_ripple':     '★ global_ripple — sin(x)·cos(y) 2D ripple',
    'global_dist_max':   '★ global_dist_max — 1 at origin, 0 at periphery',
}

DISTANCE_OPTIONS: list[str] = list(typing.get_args(DistanceVariant)) or [
    'within_only', 'euclidean', 'geodesic', 'arc_sum', 'global_arc',
]

DISTANCE_DESCRIPTIONS = {
    'within_only': 'within_only — no cross-component edges (original)',
    'euclidean':   'euclidean — straight-line distance in 2D',
    'geodesic':    'geodesic — arc to glue node + arc from glue node',
    'arc_sum':     'arc_sum — sum of each node\'s arc to its nearest glue',
    'global_arc':  'global_arc — angle difference on a shared circle',
}

st.set_page_config(page_title="Wedge PPH Explorer", layout="wide")

# =============================================================================
# Constants
# =============================================================================

TOPOLOGY_OPTIONS = [
    'wedge2', 'wedge3', 'wedge_k',
    'theta', 'lollipop', 'eyeglasses',
    'figure8_asymmetric', 'chain', 'necklace', 'necklace_full',
]

SAMPLING_OPTIONS  = ['uniform', 'jittered', 'random', 'clustered', 'beta']
ALLOCATION_OPTIONS = ['uniform', 'random', 'proportional']

# =============================================================================
# Global display toggles
# =============================================================================

page = st.sidebar.radio("Page", [
    "Graph Explorer", "Edge Comparison", "Height Comparison", "PPH Barcode"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Display options**")
show_edge_markers = st.sidebar.checkbox("Edge midpoint hover markers", value=True)
show_node_markers = st.sidebar.checkbox("Node markers", value=True)
use_arrows        = st.sidebar.checkbox("Arrows on edges (slow for large graphs)", value=False)

# =============================================================================
# Topology helpers
# =============================================================================

def needs_k(topo):      return topo in ('wedge_k', 'chain', 'necklace')
def needs_bridge(topo): return topo in ('lollipop', 'eyeglasses')
def needs_radii(topo):  return topo == 'figure8_asymmetric'


def sidebar_topology(prefix=''):
    topo   = st.sidebar.selectbox(f"Topology{' ' + prefix if prefix else ''}", TOPOLOGY_OPTIONS)
    k      = st.sidebar.slider(f"k {prefix}(circles)", 2, 8, 3) if needs_k(topo) else 2
    bridge = st.sidebar.slider(f"Bridge {prefix}", 0, 6, 0) if needs_bridge(topo) else 0
    r1     = st.sidebar.slider(f"Radius 1 {prefix}", 0.5, 3.0, 1.0, step=0.1)
    r2     = st.sidebar.slider(f"Radius 2 {prefix}", 0.5, 3.0, 2.0, step=0.1) if needs_radii(topo) else 1.0
    radii  = [r1, r2] if needs_radii(topo) else None
    return topo, k, bridge, radii


def sidebar_sampling(prefix=''):
    label = f"Sampling mode {prefix}".strip()
    mode = st.sidebar.radio(label, ["total_n", "n_per_circle"])
    if mode == "total_n":
        total_n      = st.sidebar.slider(f"total_n {prefix}".strip(), 6, 100, 20)
        allocation   = st.sidebar.selectbox(f"Allocation {prefix}".strip(), ALLOCATION_OPTIONS)
        n_per_circle = None
    else:
        n_per_circle = st.sidebar.slider(f"n per circle {prefix}".strip(), 2, 30, 8)
        total_n      = None
        allocation   = None
    sampling     = st.sidebar.selectbox(f"Sampling variant {prefix}".strip(), SAMPLING_OPTIONS)
    double_edges = st.sidebar.checkbox(f"Double edges {prefix}".strip(), value=True)
    return total_n, n_per_circle, allocation, sampling, double_edges


def sidebar_distance(prefix=''):
    """Distance function selector with descriptions."""
    label = f"Cross-component distance {prefix}".strip()
    choice = st.sidebar.selectbox(
        label,
        DISTANCE_OPTIONS,
        format_func=lambda x: DISTANCE_DESCRIPTIONS.get(x, x),
    )
    return choice


def make_cfg(topo, k, bridge, radii, height_var, distance,
             total_n, n_per_circle, allocation, sampling, double_edges):
    # Ensure at least one sampling mode is set
    if total_n is None and n_per_circle is None:
        n_per_circle = 8
        allocation   = None
    return Config(
        topology=topo, k=k, bridge_length=bridge, radii=radii,
        height_variant=height_var,
        distance=distance,
        total_n=total_n, n_per_circle=n_per_circle,
        allocation=allocation, sampling=sampling,
        double_edges=double_edges,
    )

# =============================================================================
# Hover text
# =============================================================================

def node_hover_txt(nid, G, pos):
    d = G.nodes[nid]
    x, y = pos[nid]
    glue = " [glue]" if d.get('is_glue') else ""
    return (
        f"<b>Node {nid}{glue}</b><br>"
        f"x={x:.3f}  y={y:.3f}<br>"
        f"angle={d.get('angle', 0.0):.3f} rad<br>"
        f"height={d['height']:.4f}"
    )


def edge_hover_txt(u, v, w, G, pos):
    xu, yu = pos[u]; xv, yv = pos[v]
    gu = " [glue]" if G.nodes[u].get('is_glue') else ""
    gv = " [glue]" if G.nodes[v].get('is_glue') else ""
    return (
        f"<b>Edge {u} → {v}</b><br>"
        f"weight: {w:.4f}<br><br>"
        f"<b>Source node {u}{gu}</b><br>"
        f"  x={xu:.3f}  y={yu:.3f}<br>"
        f"  height={G.nodes[u]['height']:.4f}<br><br>"
        f"<b>Target node {v}{gv}</b><br>"
        f"  x={xv:.3f}  y={yv:.3f}<br>"
        f"  height={G.nodes[v]['height']:.4f}"
    )

# =============================================================================
# Weight colour helper
# =============================================================================

def weight_color(w, w_min, w_max):
    t = (w - w_min) / (w_max - w_min) if w_max > w_min else 0.5
    return pc.sample_colorscale('plasma', float(np.clip(t, 0, 1)))[0]

# =============================================================================
# Core graph trace builder
# =============================================================================

def build_graph_traces(G, pos, heights, arc_min=0.0, arc_max=float('inf'),
                        edge_opacity=0.6, color_map=None,
                        fig=None, row=None, col=None, showscale=True):
    """
    Build all Plotly traces for a wedge graph.
    color_map: optional dict (u,v) -> CSS color string, overrides plasma coloring.
    Midpoint markers added last so they win hover.
    Returns list of traces (arrows added to fig directly if use_arrows).
    """
    h_arr    = np.array(heights)
    node_ids = list(G.nodes())

    edges_to_draw = [
        (u, v, G.edges[u, v]['weight'])
        for u, v in G.edges()
        if arc_min <= G.edges[u, v]['weight'] <= arc_max
    ]

    weights = [w for _, _, w in edges_to_draw]
    w_min = min(weights) if weights else 0.0
    w_max = max(weights) if weights else 1.0
    w_range = w_max - w_min if w_max > w_min else 1.0

    traces = []

    # ---- edges ----
    if use_arrows and fig is not None:
        # Compute correct axis ref strings for this subplot column
        if col is not None and col > 1:
            xref, yref = f'x{col}', f'y{col}'
        else:
            xref, yref = 'x', 'y'
        for u, v, w in edges_to_draw:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            c = color_map.get((u,v), weight_color(w, w_min, w_max)) if color_map else weight_color(w, w_min, w_max)
            fig.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref=xref, yref=yref, axref=xref, ayref=yref,
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=1.5, arrowcolor=c, opacity=edge_opacity,
            )
    else:
        # Group by colour bucket
        N = 20
        if color_map:
            # Each unique colour gets its own trace
            from collections import defaultdict
            groups = defaultdict(list)
            for u, v, w in edges_to_draw:
                c = color_map.get((u,v), weight_color(w, w_min, w_max))
                groups[c].append((u, v))
            for c, pairs in groups.items():
                ex, ey = [], []
                for u, v in pairs:
                    x0,y0=pos[u]; x1,y1=pos[v]
                    ex+=[x0,x1,None]; ey+=[y0,y1,None]
                traces.append(go.Scatter(x=ex, y=ey, mode='lines',
                                          line=dict(color=c, width=1.5),
                                          opacity=edge_opacity,
                                          hoverinfo='skip', showlegend=False))
        else:
            buckets = [[] for _ in range(N)]
            for u, v, w in edges_to_draw:
                b = min(int((w - w_min) / w_range * N), N-1)
                buckets[b].append((u, v, w))
            for bucket in buckets:
                if not bucket: continue
                ex, ey = [], []
                rep_w = bucket[len(bucket)//2][2]
                c = weight_color(rep_w, w_min, w_max)
                for u, v, _ in bucket:
                    x0,y0=pos[u]; x1,y1=pos[v]
                    ex+=[x0,x1,None]; ey+=[y0,y1,None]
                traces.append(go.Scatter(x=ex, y=ey, mode='lines',
                                          line=dict(color=c, width=1.5),
                                          opacity=edge_opacity,
                                          hoverinfo='skip', showlegend=False))

    # edge weight colorbar (only when no color_map override)
    if weights and not color_map and showscale:
        traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(colorscale='plasma', color=[w_min, w_max],
                        colorbar=dict(title='Edge weight', thickness=14, len=0.5, x=1.08),
                        showscale=True, size=0),
            hoverinfo='skip', showlegend=False,
        ))

    # ---- nodes ----
    glue_ids    = [n for n in node_ids if G.nodes[n].get('is_glue')]
    regular_ids = [n for n in node_ids if not G.nodes[n].get('is_glue')]

    if show_node_markers:
        if regular_ids:
            traces.append(go.Scatter(
                x=[pos[n][0] for n in regular_ids],
                y=[pos[n][1] for n in regular_ids],
                mode='markers',
                marker=dict(size=10,
                            color=[G.nodes[n]['height'] for n in regular_ids],
                            colorscale='RdYlBu',
                            cmin=float(h_arr.min()), cmax=float(h_arr.max()),
                            colorbar=dict(title='Height', thickness=14, len=0.5, x=-0.12) if showscale else None,
                            showscale=showscale,
                            line=dict(color='white', width=0.5)),
                hovertemplate='%{text}<extra></extra>',
                text=[node_hover_txt(n, G, pos) for n in regular_ids],
                showlegend=False, name='nodes',
            ))
        if glue_ids:
            traces.append(go.Scatter(
                x=[pos[n][0] for n in glue_ids],
                y=[pos[n][1] for n in glue_ids],
                mode='markers',
                marker=dict(size=16,
                            color=[G.nodes[n]['height'] for n in glue_ids],
                            colorscale='RdYlBu',
                            cmin=float(h_arr.min()), cmax=float(h_arr.max()),
                            showscale=False, symbol='diamond',
                            line=dict(color='black', width=1.5)),
                hovertemplate='%{text}<extra></extra>',
                text=[node_hover_txt(n, G, pos) for n in glue_ids],
                showlegend=False, name='glue nodes',
            ))

    # ---- midpoints LAST ----
    if show_edge_markers and edges_to_draw:
        mx, my, htexts = [], [], []
        for u, v, w in edges_to_draw:
            x0,y0=pos[u]; x1,y1=pos[v]
            mx.append((x0+x1)/2); my.append((y0+y1)/2)
            htexts.append(edge_hover_txt(u, v, w, G, pos))
        traces.append(go.Scatter(
            x=mx, y=my, mode='markers',
            marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(width=0)),
            hovertemplate='%{text}<extra></extra>',
            text=htexts, showlegend=False, name='edge info',
        ))

    return traces


def make_graph_figure(traces, title=''):
    fig = go.Figure(data=traces)
    fig.update_layout(
        title_text=title,
        xaxis=dict(visible=False, scaleanchor='y'),
        yaxis=dict(visible=False),
        height=560, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='monospace'),
        margin=dict(l=60, r=60, t=50, b=20),
    )
    return fig

# =============================================================================
# Barcode figure (with π/2 lines, legend outside, multiplicity)
# =============================================================================

def make_barcode_figure(result, height_px=400):
    finite   = [(b, d) for b, d in result.barcode if np.isfinite(d)]
    infinite = [b for b, d in result.barcode if not np.isfinite(d)]
    fig = go.Figure()

    all_vals = ([b for b,d in finite] + [d for b,d in finite] + infinite) \
               if (finite or infinite) else [0, np.pi]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.12 or 0.2

    if finite:
        counts = Counter(finite)
        bx, dy, sz, htext = [], [], [], []
        for (b, d), cnt in counts.items():
            bx.append(b); dy.append(d)
            sz.append(10 + cnt * 6)
            htext.append(f"birth: {b:.4f}<br>death: {d:.4f}<br>multiplicity: {cnt}")
        fig.add_trace(go.Scatter(
            x=bx, y=dy, mode='markers',
            marker=dict(size=sz, color='#3498db', line=dict(color='#1a5276', width=1)),
            name=f'finite ({len(finite)})',
            hovertemplate='%{text}<extra></extra>', text=htext,
        ))

    if infinite:
        pin_y = hi + pad * 0.6
        icounts = Counter(infinite)
        ix = list(icounts.keys())
        isz = [10 + c*6 for c in icounts.values()]
        itext = [f"birth: {b:.4f}<br>death: ∞<br>multiplicity: {icounts[b]}" for b in ix]
        fig.add_trace(go.Scatter(
            x=ix, y=[pin_y]*len(ix), mode='markers',
            marker=dict(size=isz, symbol='triangle-up', color='#e74c3c',
                        line=dict(color='#922b21', width=1)),
            name=f'infinite ({len(infinite)})',
            hovertemplate='%{text}<extra></extra>', text=itext,
        ))

    # diagonal
    fig.add_trace(go.Scatter(
        x=[lo-pad, hi+pad], y=[lo-pad, hi+pad],
        mode='lines', line=dict(color='gray', dash='dash', width=1),
        showlegend=False, hoverinfo='skip',
    ))

    # π/2 reference lines as explicit traces (more reliable than add_vline)
    half_pi = np.pi / 2
    for xy, is_vert in [(half_pi, True), (half_pi, False)]:
        fig.add_trace(go.Scatter(
            x=[xy, xy] if is_vert else [lo-pad, hi+pad],
            y=[lo-pad, hi+pad] if is_vert else [xy, xy],
            mode='lines',
            line=dict(color='rgba(180,80,80,0.55)', dash='dot', width=1.5),
            showlegend=False, hoverinfo='skip',
        ))
    fig.add_annotation(x=half_pi, y=lo-pad*0.4, text='π/2', showarrow=False,
                       font=dict(size=11, color='rgba(180,80,80,0.9)'))
    fig.add_annotation(x=lo-pad*0.4, y=half_pi, text='π/2', showarrow=False,
                       font=dict(size=11, color='rgba(180,80,80,0.9)'))

    fig.update_layout(
        xaxis_title='Birth', yaxis_title='Death',
        height=height_px,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=60),
    )
    return fig

# =============================================================================
# Height function plot (1-D, over angle)
# =============================================================================

def make_height_plot(height_var, dip_depth=1.0, epsilon0=0.05):
    """Plot the height function over [0, 2π)."""
    from wedge_pph import _height_for
    angles = np.linspace(0, 2*np.pi, 500)
    vals   = [_height_for(a, height_var, dip_depth, epsilon0) for a in angles]
    fig = go.Figure(go.Scatter(
        x=angles / np.pi, y=vals, mode='lines',
        line=dict(color='#3498db', width=2),
        hovertemplate='angle=%{x:.3f}π<br>height=%{y:.4f}<extra></extra>',
    ))
    fig.update_layout(
        xaxis_title='angle / π', yaxis_title='height',
        height=280,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=20, b=40),
    )
    fig.add_vline(x=0.5, line=dict(color='rgba(180,80,80,0.5)', dash='dot', width=1.5),
                  annotation_text='π/2')
    fig.add_vline(x=1.0, line=dict(color='rgba(100,100,100,0.4)', dash='dot', width=1),
                  annotation_text='π')
    return fig

# =============================================================================
# Comparison colour map helper
# =============================================================================

def comparison_color_map(set_here, set_ref):
    """Return {(u,v): color} for edges in set_here."""
    only = set_here - set_ref
    comm = set_here & set_ref
    cmap = {}
    for e in comm:  cmap[e] = 'gray'
    for e in only:  cmap[e] = 'orange'
    return cmap

# =============================================================================
# PAGE 1 — Graph Explorer
# =============================================================================

if page == "Graph Explorer":
    st.title("Wedge Graph Explorer")
    st.caption("Hover over **edge midpoints** for vertex info. Diamonds = glue nodes.")

    st.sidebar.header("Topology")
    topo, k, bridge, radii = sidebar_topology()

    st.sidebar.header("Sampling")
    total_n, n_per_circle, allocation, sampling, double_edges = sidebar_sampling()

    st.sidebar.header("Height")
    height_var = st.sidebar.selectbox("Height function", HEIGHT_OPTIONS, format_func=lambda x: HEIGHT_DESCRIPTIONS.get(x, x))

    st.sidebar.header("Distance")
    distance = sidebar_distance()

    st.sidebar.header("Edge filter")
    arc_min      = st.sidebar.slider("Min edge weight", 0.0, 10.0, 0.0, step=0.05)
    arc_max      = st.sidebar.slider("Max edge weight", 0.1, 20.0, 20.0, step=0.05)
    edge_opacity = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.6, step=0.05)

    st.sidebar.header("Display")
    show_height_fn = st.sidebar.checkbox("Show height function plot", value=False)
    run_pph        = st.sidebar.checkbox("Compute PPH barcode", value=False)

    try:
        cfg = make_cfg(topo, k, bridge, radii, height_var, distance,
                       total_n, n_per_circle, allocation, sampling, double_edges)
        with st.spinner("Building graph..."):
            G, pos = build_graph(cfg)
        heights = [G.nodes[n]['height'] for n in G.nodes()]
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{G.number_of_nodes()} nodes** · **{G.number_of_edges()} edges**")

        traces = build_graph_traces(G, pos, heights, arc_min, arc_max, edge_opacity)
        fig = make_graph_figure(traces,
            title=f"{topo}  |  height: {height_var}  |  "
                  f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        st.plotly_chart(fig, use_container_width=True)

        if show_height_fn:
            st.subheader(f"Height function: {height_var}")
            st.plotly_chart(make_height_plot(height_var), use_container_width=True)

        if run_pph:
            st.subheader("PPH Barcode")
            with st.spinner("Computing PPH..."):
                result = analyze(cfg)
            st.markdown(f"**{result.n_bars} H₁ bars** · max death = {result.max_death/np.pi:.3f}π")
            if result.barcode:
                st.plotly_chart(make_barcode_figure(result), use_container_width=True)
                with st.expander("Raw barcode"):
                    for b, d in result.barcode:
                        d_str = f"{d/np.pi:.4f}π" if np.isfinite(d) else "∞"
                        st.text(f"  [{b/np.pi:.4f}π,  {d_str}]")
            else:
                st.info("No H₁ bars found.")

    except Exception as e:
        st.error(f"Error: {e}")


# =============================================================================
# PAGE 2 — Edge Comparison (two arc lengths, same topology)
# =============================================================================

elif page == "Edge Comparison":
    st.title("Edge Comparison")
    st.markdown(
        "Compare edges at two arc-length thresholds on the same graph. "
        "**Orange** = only at arc A, **green** = only at arc B, **gray** = both."
    )

    st.sidebar.header("Topology")
    topo, k, bridge, radii = sidebar_topology()

    st.sidebar.header("Sampling")
    total_n, n_per_circle, allocation, sampling, double_edges = sidebar_sampling()

    st.sidebar.header("Height")
    height_var = st.sidebar.selectbox("Height function", HEIGHT_OPTIONS, format_func=lambda x: HEIGHT_DESCRIPTIONS.get(x, x))

    st.sidebar.header("Distance")
    distance = sidebar_distance()

    st.sidebar.header("Arc lengths to compare")
    arc_a = st.sidebar.slider("Arc A", 0.0, 10.0, 1.0, step=0.05)
    arc_b = st.sidebar.slider("Arc B", 0.0, 10.0, 2.0, step=0.05)
    eps   = 1e-6

    st.sidebar.header("Display")
    edge_opacity   = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.7, step=0.05)
    show_height_fn = st.sidebar.checkbox("Show height function plot", value=False)

    try:
        cfg = make_cfg(topo, k, bridge, radii, height_var, distance,
                       total_n, n_per_circle, allocation, sampling, double_edges)
        with st.spinner("Building graph..."):
            G, pos = build_graph(cfg)
        heights = [G.nodes[n]['height'] for n in G.nodes()]

        edges_a = {(u,v) for u,v in G.edges() if abs(G.edges[u,v]['weight']-arc_a) < eps}
        edges_b = {(u,v) for u,v in G.edges() if abs(G.edges[u,v]['weight']-arc_b) < eps}
        only_a  = edges_a - edges_b
        only_b  = edges_b - edges_a
        common  = edges_a & edges_b

        c1,c2,c3 = st.columns(3)
        c1.metric(f"Only at arc A ({arc_a:.2f})", len(only_a), help="orange")
        c2.metric(f"Only at arc B ({arc_b:.2f})", len(only_b), help="green")
        c3.metric("Both", len(common), help="gray")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Arc A = {arc_a:.2f}  (orange = unique)",
                                            f"Arc B = {arc_b:.2f}  (green = unique)"],
                            horizontal_spacing=0.08)

        for col_idx, (edge_set, edge_ref, uniq_col) in enumerate(
                [(edges_a, edges_b, 'orange'), (edges_b, edges_a, 'limegreen')], start=1):
            cmap = {}
            for e in (edge_set & edge_ref): cmap[e] = 'gray'
            for e in (edge_set - edge_ref):  cmap[e] = uniq_col
            # Only draw edges in this set
            cfg_edges = [(u,v,G.edges[u,v]['weight']) for u,v in edge_set]
            ts = build_graph_traces(G, pos, heights,
                                     arc_min=0, arc_max=float('inf'),
                                     edge_opacity=edge_opacity,
                                     color_map=cmap,
                                     fig=fig, row=1, col=col_idx,
                                     showscale=(col_idx==2))
            for t in ts:
                fig.add_trace(t, row=1, col=col_idx)

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor='x')
        fig.update_layout(
            height=560, showlegend=False,
            title_text=f"{topo}  |  height: {height_var}  |  gray = both arcs",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='monospace'),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_height_fn:
            st.subheader(f"Height function: {height_var}")
            st.plotly_chart(make_height_plot(height_var), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")


# =============================================================================
# PAGE 3 — Height Comparison (same topology, two height functions)
# =============================================================================

elif page == "Height Comparison":
    st.title("Height Function Comparison")
    st.markdown(
        "Same topology, two height functions. "
        "**Orange** = edges only in A, **green** = only in B, **gray** = same directed edge in both."
    )

    st.sidebar.header("Topology")
    topo, k, bridge, radii = sidebar_topology()

    st.sidebar.header("Sampling")
    total_n, n_per_circle, allocation, sampling, double_edges = sidebar_sampling()

    st.sidebar.header("Height functions")
    height_a = st.sidebar.selectbox("Height A", HEIGHT_OPTIONS, index=0, format_func=lambda x: HEIGHT_DESCRIPTIONS.get(x, x))
    height_b = st.sidebar.selectbox("Height B", HEIGHT_OPTIONS, index=1, format_func=lambda x: HEIGHT_DESCRIPTIONS.get(x, x))

    st.sidebar.header("Distance")
    distance = sidebar_distance()

    st.sidebar.header("Edge filter")
    arc_min = st.sidebar.slider("Min edge weight", 0.0, 10.0, 0.0, step=0.05)
    arc_max = st.sidebar.slider("Max edge weight", 0.1, 20.0, 20.0, step=0.05)

    st.sidebar.header("Display")
    edge_opacity   = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.6, step=0.05)
    show_height_fn = st.sidebar.checkbox("Show height function plots", value=False)
    run_pph        = st.sidebar.checkbox("Compute & compare PPH barcodes", value=False)

    try:
        cfg_a = make_cfg(topo, k, bridge, radii, height_a, distance,
                         total_n, n_per_circle, allocation, sampling, double_edges)
        cfg_b = make_cfg(topo, k, bridge, radii, height_b, distance,
                         total_n, n_per_circle, allocation, sampling, double_edges)

        with st.spinner("Building graphs..."):
            G_a, pos_a = build_graph(cfg_a)
            G_b, pos_b = build_graph(cfg_b)

        heights_a = [G_a.nodes[n]['height'] for n in G_a.nodes()]
        heights_b = [G_b.nodes[n]['height'] for n in G_b.nodes()]

        def filtered(G):
            return {(u,v) for u,v in G.edges()
                    if arc_min <= G.edges[u,v]['weight'] <= arc_max}

        set_a = filtered(G_a) or set()
        set_b = filtered(G_b) or set()
        only_a = set_a - set_b
        only_b = set_b - set_a
        common  = set_a & set_b

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"A: **{G_a.number_of_edges()} edges**")
        st.sidebar.markdown(f"B: **{G_b.number_of_edges()} edges**")

        c1,c2,c3 = st.columns(3)
        c1.metric("Only in A (orange)", len(only_a))
        c2.metric("Only in B (green)",  len(only_b))
        c3.metric("Both (gray)",        len(common))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Height A: {height_a}  (orange = only in A)",
                                            f"Height B: {height_b}  (green = only in B)"],
                            horizontal_spacing=0.08)

        cmap_a = comparison_color_map(set_a, set_b)
        cmap_b = {e: 'limegreen' if e in only_b else 'gray' for e in set_b}

        for col_idx, (G, pos, heights, cmap) in enumerate(
                [(G_a, pos_a, heights_a, cmap_a),
                 (G_b, pos_b, heights_b, cmap_b)], start=1):
            ts = build_graph_traces(G, pos, heights,
                                     arc_min=arc_min, arc_max=arc_max,
                                     edge_opacity=edge_opacity,
                                     color_map=cmap,
                                     fig=fig, row=1, col=col_idx,
                                     showscale=(col_idx==2))
            for t in ts:
                fig.add_trace(t, row=1, col=col_idx)

        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor='x')
        fig.update_layout(
            height=560, showlegend=False,
            title_text=f"{topo}  |  gray = same directed edge in both",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='monospace'),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_height_fn:
            st.subheader("Height function plots")
            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown(f"**{height_a}**")
                st.plotly_chart(make_height_plot(height_a), use_container_width=True)
            with hc2:
                st.markdown(f"**{height_b}**")
                st.plotly_chart(make_height_plot(height_b), use_container_width=True)

        if run_pph:
            st.subheader("PPH Barcode Comparison")
            col1, col2 = st.columns(2)
            for col, cfg, label in [(col1, cfg_a, height_a), (col2, cfg_b, height_b)]:
                with col:
                    st.markdown(f"**{label}**")
                    with st.spinner(f"Computing PPH for {label}..."):
                        result = analyze(cfg)
                    st.markdown(
                        f"{result.n_bars} H₁ bars · "
                        f"max death = {result.max_death/np.pi:.3f}π"
                    )
                    st.plotly_chart(make_barcode_figure(result, height_px=350),
                                    use_container_width=True)
                    with st.expander("Raw barcode"):
                        for b, d in result.barcode:
                            d_str = f"{d/np.pi:.4f}π" if np.isfinite(d) else "∞"
                            st.text(f"  [{b/np.pi:.4f}π,  {d_str}]")

    except Exception as e:
        import traceback
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())
# =============================================================================

elif page == "PPH Barcode":
    st.title("PPH Barcode Explorer")

    st.sidebar.header("Topology")
    topo, k, bridge, radii = sidebar_topology()

    st.sidebar.header("Sampling")
    total_n, n_per_circle, allocation, sampling, double_edges = sidebar_sampling()

    st.sidebar.header("Height")
    height_var     = st.sidebar.selectbox("Height function", HEIGHT_OPTIONS, format_func=lambda x: HEIGHT_DESCRIPTIONS.get(x, x))
    show_height_fn = st.sidebar.checkbox("Show height function plot", value=False)

    st.sidebar.header("Distance")
    distance = sidebar_distance()

    try:
        cfg = make_cfg(topo, k, bridge, radii, height_var, distance,
                       total_n, n_per_circle, allocation, sampling, double_edges)

        col_graph, col_bar = st.columns([1, 1])

        with col_graph:
            st.subheader("Graph")
            with st.spinner("Building graph..."):
                G, pos = build_graph(cfg)
            heights = [G.nodes[n]['height'] for n in G.nodes()]
            st.caption(f"{G.number_of_nodes()} nodes · {G.number_of_edges()} edges")
            traces = build_graph_traces(G, pos, heights)
            fig_g  = make_graph_figure(traces, title=topo)
            fig_g.update_layout(height=450)
            st.plotly_chart(fig_g, use_container_width=True)

            if show_height_fn:
                st.subheader(f"Height: {height_var}")
                st.plotly_chart(make_height_plot(height_var), use_container_width=True)

        with col_bar:
            st.subheader("Barcode")
            with st.spinner("Computing PPH..."):
                result = analyze(cfg)
            st.markdown(
                f"**{result.n_bars} H₁ bars** · "
                f"max death = {result.max_death/np.pi:.3f}π"
            )
            if result.barcode:
                st.plotly_chart(make_barcode_figure(result, height_px=450),
                                use_container_width=True)
                with st.expander("Raw barcode"):
                    for b, d in result.barcode:
                        d_str = f"{d/np.pi:.4f}π" if np.isfinite(d) else "∞"
                        st.text(f"  [{b/np.pi:.4f}π,  {d_str}]")
            else:
                st.info("No H₁ bars found.")

    except Exception as e:
        st.error(f"Error: {e}")
