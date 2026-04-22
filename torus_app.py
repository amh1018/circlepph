"""
Torus PPH Explorer — Streamlit app
===================================
Run with:
    streamlit run torus_app.py

Features:
- 3D Torus, Edge Comparison, Height Comparison pages
- Flat (θ,φ) or 3D torus view on every page
- Arrow or line edges
- Hoverable edge midpoints and nodes
- Edge set diff with orientation on Height Comparison
- Height function heatmap
- π/2 reference lines on persistence diagrams
- n up to 20 for more sampling points
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

from torus_pph import Config, build_graph, analyze, height as torus_height

st.set_page_config(page_title="Torus PPH Explorer", layout="wide")

# =============================================================================
# Constants
# =============================================================================

HEIGHT_OPTIONS = [
    "sin_theta", "sin_phi", "cos_theta", "cos_phi",
    "sin_plus_sin", "sin_minus_sin", "sin_times_sin",
    "saddle", "diagonal", "antidiagonal",
    "z_coord", "x_coord", "y_coord",
    "sin2_theta", "sin2_phi",
    "sin_plus_sin2", "sin_plus_cos2", "sin_cos",
    "triangle", "square", "sawtooth_theta",
    "damped", "fractal3", "sigmoid_periodic",
    "sin_cubed", "double_min", "triple_max",'sawtooth_sum', 'sawtooth_product', 'sawtooth_diag',
]
SAMPLING_OPTIONS = ["grid", "jittered", "random", "clustered", "chebyshev"]

# =============================================================================
# Global display toggles
# =============================================================================

page = st.sidebar.radio("Page", ["3D Torus", "Edge Comparison", "Height Comparison"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Display options**")
show_edge_markers = st.sidebar.checkbox("Edge midpoint hover markers", value=True)
show_node_markers = st.sidebar.checkbox("Node markers", value=True)
use_arrows        = st.sidebar.checkbox("Arrows on edges (slow for large n)", value=False)

# =============================================================================
# Core helpers
# =============================================================================

def to_xyz(theta, phi, R, r):
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def is_wrap(points, u, v):
    tu, pu = points[u]
    tv, pv = points[v]
    dt = min(abs(tu - tv), 2 * np.pi - abs(tu - tv))
    dp = min(abs(pu - pv), 2 * np.pi - abs(pu - pv))
    return dt > np.pi or dp > np.pi


def weight_color(w, w_min, w_max):
    t = (w - w_min) / (w_max - w_min) if w_max > w_min else 0.5
    return pc.sample_colorscale('plasma', float(np.clip(t, 0, 1)))[0]


def edge_hover_text(u, v, w, points, heights):
    tu, pu = points[u]
    tv, pv = points[v]
    return (
        f"<b>Edge {u} → {v}</b><br>"
        f"weight: {w:.4f}<br><br>"
        f"<b>Source node {u}</b><br>"
        f"  θ={tu/np.pi:.3f}π  φ={pu/np.pi:.3f}π<br>"
        f"  height={heights[u]:.4f}<br><br>"
        f"<b>Target node {v}</b><br>"
        f"  θ={tv/np.pi:.3f}π  φ={pv/np.pi:.3f}π<br>"
        f"  height={heights[v]:.4f}"
    )


def node_hover_text(i, points, heights):
    t, p = points[i]
    return (
        f"<b>Node {i}</b><br>"
        f"θ={t/np.pi:.3f}π  φ={p/np.pi:.3f}π<br>"
        f"height={heights[i]:.4f}"
    )


def flat_xy(points):
    return [(t / np.pi, p / np.pi) for t, p in points]


# =============================================================================
# Barcode figure (reused on all pages)
# =============================================================================

def make_barcode_figure(result, height_400=True):
    """Persistence diagram with π/2 dashed reference lines."""
    finite   = [(b, d) for b, d in result.barcode if np.isfinite(d)]
    infinite = [b for b, d in result.barcode if not np.isfinite(d)]
    fig = go.Figure()

    if finite:
        births, deaths = zip(*finite)
        fig.add_trace(go.Scatter(
            x=list(births), y=list(deaths), mode='markers',
            marker=dict(size=10, color='#3498db',
                        line=dict(color='#1a5276', width=1)),
            name=f'finite ({len(finite)})',
            hovertemplate='birth: %{x:.4f}<br>death: %{y:.4f}<extra></extra>',
        ))

    all_vals = ([b for b, d in finite] + [d for b, d in finite] + infinite) \
               if (finite or infinite) else [0, np.pi]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.1 or 0.2

    if infinite:
        pin_y = hi + pad * 0.5
        fig.add_trace(go.Scatter(
            x=infinite, y=[pin_y]*len(infinite), mode='markers',
            marker=dict(size=12, symbol='triangle-up', color='#e74c3c',
                        line=dict(color='#922b21', width=1)),
            name=f'infinite ({len(infinite)})',
            hovertemplate='birth: %{x:.4f}<br>death: ∞<extra></extra>',
        ))

    # diagonal
    fig.add_trace(go.Scatter(
        x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
        mode='lines', line=dict(color='gray', dash='dash', width=1),
        showlegend=False, hoverinfo='skip',
    ))

    # π/2 reference lines
    half_pi = np.pi / 2
    fig.add_vline(x=half_pi,
                  line=dict(color='rgba(180,80,80,0.55)', dash='dot', width=1.5),
                  annotation_text='π/2', annotation_position='top',
                  annotation_font=dict(size=11, color='rgba(180,80,80,0.9)'))
    fig.add_hline(y=half_pi,
                  line=dict(color='rgba(180,80,80,0.55)', dash='dot', width=1.5),
                  annotation_text='π/2', annotation_position='right',
                  annotation_font=dict(size=11, color='rgba(180,80,80,0.9)'))

    fig.update_layout(
        xaxis_title='Birth', yaxis_title='Death',
        height=400 if height_400 else 350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='left', x=0,
        ),
        margin=dict(t=60),
    )
    return fig


# =============================================================================
# Height function heatmap
# =============================================================================

def make_height_heatmap(height_variant, R=3.0, r=1.0):
    """2-D contour heatmap of the height function over (θ, φ)."""
    theta = np.linspace(0, 2 * np.pi, 200)
    phi   = np.linspace(0, 2 * np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    Z = np.vectorize(lambda t, p: torus_height(t, p, height_variant, R, r))(TH, PH)

    fig = go.Figure(go.Contour(
        x=theta / np.pi, y=phi / np.pi, z=Z,
        colorscale='RdBu_r',
        contours=dict(showlabels=False),
        colorbar=dict(title='f(θ,φ)', thickness=14),
        hovertemplate='θ=%{x:.2f}π  φ=%{y:.2f}π<br>height=%{z:.4f}<extra></extra>',
    ))
    fig.update_layout(
        xaxis_title='θ / π', yaxis_title='φ / π',
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


# =============================================================================
# 2-D flat edge/node trace builders
# =============================================================================

def make_flat_edge_traces(edges_to_draw, coords, w_min, w_max,
                           edge_opacity, fig=None, row=None, col=None,
                           color_override=None):
    """Line traces or arrow annotations (if use_arrows). Returns list of traces."""
    if not edges_to_draw:
        return []

    if use_arrows and fig is not None:
        for u, v, w in edges_to_draw:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            c = color_override or weight_color(w, w_min, w_max)
            ann = dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=1.5, arrowcolor=c, opacity=edge_opacity,
            )
            if row is not None:
                fig.add_annotation(**ann, row=row, col=col)
            else:
                fig.add_annotation(**ann)
        return []

    N = 20
    w_range = w_max - w_min if w_max > w_min else 1.0
    buckets = [[] for _ in range(N)]
    for u, v, w in edges_to_draw:
        b = min(int((w - w_min) / w_range * N), N - 1)
        buckets[b].append((u, v, w))

    traces = []
    for bucket in buckets:
        if not bucket:
            continue
        ex, ey = [], []
        rep_w = bucket[len(bucket) // 2][2]
        c = color_override or weight_color(rep_w, w_min, w_max)
        for u, v, _ in bucket:
            x0, y0 = coords[u]; x1, y1 = coords[v]
            ex += [x0, x1, None]; ey += [y0, y1, None]
        traces.append(go.Scatter(
            x=ex, y=ey, mode='lines',
            line=dict(color=c, width=1.5), opacity=edge_opacity,
            hoverinfo='skip', showlegend=False,
        ))
    return traces


def make_flat_node_trace(points, heights, showscale=True):
    h_arr  = np.array(heights)
    coords = flat_xy(points)
    texts  = [node_hover_text(i, points, heights) for i in range(len(heights))]
    return go.Scatter(
        x=[c[0] for c in coords], y=[c[1] for c in coords],
        mode='markers',
        marker=dict(
            size=10, color=h_arr.tolist(), colorscale='RdBu',
            cmin=float(h_arr.min()), cmax=float(h_arr.max()),
            showscale=showscale,
            colorbar=dict(title='Height', thickness=14) if showscale else None,
            line=dict(color='white', width=0.5),
        ),
        hovertemplate='%{text}<extra></extra>',
        text=texts, showlegend=False,
    )


def make_flat_midpoint_trace(edges_to_draw, coords, points, heights):
    mx, my, htexts = [], [], []
    for u, v, w in edges_to_draw:
        x0, y0 = coords[u]; x1, y1 = coords[v]
        mx.append((x0+x1)/2); my.append((y0+y1)/2)
        htexts.append(edge_hover_text(u, v, w, points, heights))
    return go.Scatter(
        x=mx, y=my, mode='markers',
        marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(width=0)),
        hovertemplate='%{text}<extra></extra>',
        text=htexts, showlegend=False, name='edge info',
    )


# =============================================================================
# 3-D torus trace builders
# =============================================================================

def make_torus_surface(R, r, opacity=0.10):
    u = np.linspace(0, 2*np.pi, 80); v = np.linspace(0, 2*np.pi, 40)
    U, V = np.meshgrid(u, v)
    return go.Surface(
        x=(R + r*np.cos(V))*np.cos(U),
        y=(R + r*np.cos(V))*np.sin(U),
        z=r*np.sin(V),
        colorscale=[[0,'lightgrey'],[1,'lightgrey']],
        showscale=False, opacity=opacity,
        hoverinfo='skip', lighting=dict(ambient=0.9, diffuse=0.4),
    )


def make_3d_edge_traces(edges_to_draw, node_xyz, w_min, w_max, edge_opacity):
    if not edges_to_draw:
        return []
    N = 20
    w_range = w_max - w_min if w_max > w_min else 1.0
    buckets = [[] for _ in range(N)]
    for u, v, w in edges_to_draw:
        b = min(int((w - w_min) / w_range * N), N-1)
        buckets[b].append((u, v, w))

    traces = []
    cone_x, cone_y, cone_z, cone_u, cone_v, cone_w = [], [], [], [], [], []

    for bucket in buckets:
        if not bucket:
            continue
        ex, ey, ez = [], [], []
        rep_w = bucket[len(bucket)//2][2]
        col = weight_color(rep_w, w_min, w_max)
        for u, v, w in bucket:
            x0,y0,z0 = node_xyz[u]; x1,y1,z1 = node_xyz[v]
            ex += [x0,x1,None]; ey += [y0,y1,None]; ez += [z0,z1,None]
            if use_arrows:
                dx,dy,dz = x1-x0, y1-y0, z1-z0
                ln = max(np.sqrt(dx**2+dy**2+dz**2), 1e-9)
                s = 0.12
                cone_x.append(x1); cone_y.append(y1); cone_z.append(z1)
                cone_u.append(dx/ln*s); cone_v.append(dy/ln*s); cone_w.append(dz/ln*s)
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color=col, width=2),
            opacity=edge_opacity, hoverinfo='skip', showlegend=False,
        ))

    if use_arrows and cone_x:
        traces.append(go.Cone(
            x=cone_x, y=cone_y, z=cone_z,
            u=cone_u, v=cone_v, w=cone_w,
            colorscale=[[0,'gray'],[1,'gray']],
            showscale=False, sizemode='absolute', sizeref=0.08,
            hoverinfo='skip', showlegend=False,
        ))
    return traces


def make_3d_node_trace(points, heights, node_xyz):
    h_arr = np.array(heights)
    xs, ys, zs = zip(*node_xyz)
    texts = [node_hover_text(i, points, heights) for i in range(len(heights))]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode='markers',
        marker=dict(size=6, color=h_arr.tolist(), colorscale='RdBu',
                    colorbar=dict(title='Height', x=-0.12, thickness=14, len=0.6),
                    line=dict(color='white', width=0.5)),
        hovertemplate='%{text}<extra></extra>',
        text=texts, name='nodes',
    )


def make_3d_midpoint_trace(edges_to_draw, node_xyz, points, heights):
    mx, my, mz, htexts = [], [], [], []
    for u, v, w in edges_to_draw:
        x0,y0,z0 = node_xyz[u]; x1,y1,z1 = node_xyz[v]
        mx.append((x0+x1)/2); my.append((y0+y1)/2); mz.append((z0+z1)/2)
        htexts.append(edge_hover_text(u, v, w, points, heights))
    return go.Scatter3d(
        x=mx, y=my, z=mz, mode='markers',
        marker=dict(size=8, color='rgba(0,0,0,0)'),
        hovertemplate='%{text}<extra></extra>',
        text=htexts, showlegend=False, name='edge info',
    )


def scene_layout():
    return dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)',
        camera=dict(eye=dict(x=1.6, y=1.2, z=0.8)),
        aspectmode='data',
    )


def base_layout(height=540, title=''):
    return dict(
        height=height, title_text=title,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        hoverlabel=dict(bgcolor='white', font_size=12, font_family='monospace'),
    )


# =============================================================================
# Reusable flat-panel builder
# =============================================================================

def add_flat_panel(fig, G, points, heights, coords,
                   edges_to_draw, w_min, w_max,
                   edge_opacity, row, col,
                   color_override=None, showscale=True):
    """Add all flat traces for one panel to fig. Midpoints added last."""
    line_ts = make_flat_edge_traces(
        edges_to_draw, coords, w_min, w_max, edge_opacity,
        fig=fig, row=row, col=col, color_override=color_override)
    for t in line_ts:
        fig.add_trace(t, row=row, col=col)
    if show_node_markers:
        nt = make_flat_node_trace(points, heights, showscale=showscale)
        fig.add_trace(nt, row=row, col=col)
    if show_edge_markers and edges_to_draw:
        mt = make_flat_midpoint_trace(edges_to_draw, coords, points, heights)
        fig.add_trace(mt, row=row, col=col)


def add_flat_comparison_panel(fig, G, points, heights, coords,
                               edges_here, edges_ref, unique_color,
                               edge_opacity, row, col, showscale=True):
    """Add comparison-coloured flat panel (orange/green/gray logic)."""
    only_here = edges_here - edges_ref
    comm      = edges_here & edges_ref

    wrap, comm_list, only_list = [], [], []
    for u, v in edges_here:
        x0,y0 = coords[u]; x1,y1 = coords[v]
        w = G.edges[u,v]['weight']
        if abs(x1-x0) > 1.0 or abs(y1-y0) > 1.0:
            wrap.append((u,v,w))
        elif (u,v) in comm:
            comm_list.append((u,v,w))
        else:
            only_list.append((u,v,w))

    all_w = [G.edges[u,v]['weight'] for u,v in edges_here]
    wn = min(all_w) if all_w else 0.0
    wx = max(all_w) if all_w else 1.0

    if wrap:
        ex,ey=[],[]
        for u,v,_ in wrap:
            ex+=[coords[u][0],coords[v][0],None]; ey+=[coords[u][1],coords[v][1],None]
        fig.add_trace(go.Scatter(x=ex,y=ey,mode='lines',
                                 line=dict(color='lightgray',width=0.5),
                                 opacity=0.3,hoverinfo='skip',showlegend=False),
                      row=row, col=col)

    for edge_list, col_str in [(comm_list, 'gray'), (only_list, unique_color)]:
        ts = make_flat_edge_traces(edge_list, coords, wn, wx, edge_opacity,
                                   fig=fig, row=row, col=col,
                                   color_override=col_str)
        for t in ts:
            fig.add_trace(t, row=row, col=col)

    if show_node_markers:
        fig.add_trace(make_flat_node_trace(points, heights, showscale=showscale),
                      row=row, col=col)

    non_wrap = comm_list + only_list
    if show_edge_markers and non_wrap:
        fig.add_trace(make_flat_midpoint_trace(non_wrap, coords, points, heights),
                      row=row, col=col)


# =============================================================================
# PAGE 1 — 3D Torus
# =============================================================================

if page == "3D Torus":
    st.title("3D Torus Explorer")
    st.caption("Hover over edge midpoints for vertex info. Hover nodes for coordinates.")

    st.sidebar.header("Graph parameters")
    n            = st.sidebar.slider("n (grid side, n×n points)", 2, 20, 5)
    R            = st.sidebar.slider("R (major radius)", 1.5, 6.0, 3.0, step=0.5)
    r            = st.sidebar.slider("r (minor radius)", 0.3, 2.0, 1.0, step=0.1)
    height_var   = st.sidebar.selectbox("Height function", HEIGHT_OPTIONS)
    sampling     = st.sidebar.selectbox("Sampling", SAMPLING_OPTIONS)
    double_edges = st.sidebar.checkbox("Double edges (ties)", value=True)

    st.sidebar.header("Edge filter")
    arc_min = st.sidebar.slider("Min edge length", 0.0, 6.0, 0.0, step=0.05)
    arc_max = st.sidebar.slider("Max edge length", 0.1, 10.0, 10.0, step=0.05)

    st.sidebar.header("Display")
    show_surface    = st.sidebar.checkbox("Show torus surface", value=True)
    surface_opacity = st.sidebar.slider("Surface opacity", 0.0, 0.5, 0.10, step=0.01)
    edge_opacity    = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.6, step=0.05)
    show_height_fn  = st.sidebar.checkbox("Show height function heatmap", value=False)
    run_pph         = st.sidebar.checkbox("Compute PPH barcode (slow for n > 7)", value=False)

    cfg = Config(n=n, height_variant=height_var, sampling=sampling,
                 double_edges=double_edges, R=R, r=r)
    with st.spinner("Building graph..."):
        G, heights, points = build_graph(cfg)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{G.number_of_nodes()} nodes** · **{G.number_of_edges()} edges**")

    node_xyz = [to_xyz(t, p, R, r) for t, p in points]
    edges_to_draw = [
        (u, v, G.edges[u,v]['weight']) for u,v in G.edges()
        if arc_min <= G.edges[u,v]['weight'] <= arc_max and not is_wrap(points,u,v)
    ]
    all_w = [w for _,_,w in edges_to_draw]
    w_min = min(all_w) if all_w else 0.0
    w_max = max(all_w) if all_w else 1.0

    data = []
    if show_surface:
        surf = make_torus_surface(R, r)
        surf.opacity = surface_opacity
        data.append(surf)
    data.extend(make_3d_edge_traces(edges_to_draw, node_xyz, w_min, w_max, edge_opacity))
    if all_w:
        data.append(go.Scatter3d(
            x=[None],y=[None],z=[None], mode='markers',
            marker=dict(colorscale='plasma', color=[w_min,w_max],
                        colorbar=dict(title='Edge weight',x=1.05,thickness=14,len=0.6),
                        showscale=True, size=0),
            hoverinfo='skip', showlegend=False,
        ))
    if show_node_markers:
        data.append(make_3d_node_trace(points, heights, node_xyz))
    if show_edge_markers and edges_to_draw:
        data.append(make_3d_midpoint_trace(edges_to_draw, node_xyz, points, heights))

    fig = go.Figure(data=data)
    fig.update_layout(scene=scene_layout(), margin=dict(l=0,r=0,t=20,b=0),
                      height=600, showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                      hoverlabel=dict(bgcolor='white',font_size=12,font_family='monospace'))
    st.plotly_chart(fig, use_container_width=True)

    if show_height_fn:
        st.subheader(f"Height function: {height_var}")
        st.plotly_chart(make_height_heatmap(height_var, R, r), use_container_width=True)

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

    st.caption(f"n={n}, R={R}, r={r}, height={height_var!r}, "
               f"sampling={sampling!r}, double_edges={double_edges}")


# =============================================================================
# PAGE 2 — Edge Comparison
# =============================================================================

elif page == "Edge Comparison":
    st.title("Edge Comparison")
    st.markdown("Compare edges at two arc-length thresholds. **Hover edge midpoints** for info.")

    st.sidebar.header("Graph parameters")
    n            = st.sidebar.slider("n (grid side, n×n points)", 2, 20, 5)
    R            = st.sidebar.slider("R (major radius)", 1.5, 6.0, 3.0, step=0.5)
    r_minor      = st.sidebar.slider("r (minor radius)", 0.3, 2.0, 1.0, step=0.1)
    height_var   = st.sidebar.selectbox("Height function", HEIGHT_OPTIONS)
    sampling     = st.sidebar.selectbox("Sampling", SAMPLING_OPTIONS)
    double_edges = st.sidebar.checkbox("Double edges (ties)", value=True)

    st.sidebar.header("Arc lengths to compare")
    arc_a = st.sidebar.slider("Arc A (× π)", 0.0, 2.0, 0.4, step=0.05) * np.pi
    arc_b = st.sidebar.slider("Arc B (× π)", 0.0, 2.0, 0.8, step=0.05) * np.pi
    eps   = 1e-6

    st.sidebar.header("View")
    view_3d      = st.sidebar.checkbox("3D torus view", value=False)
    edge_opacity = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.7, step=0.05)
    show_surface = st.sidebar.checkbox("Show torus surface", value=True)
    show_height_fn = st.sidebar.checkbox("Show height function heatmap", value=False)

    cfg = Config(n=n, height_variant=height_var, sampling=sampling,
                 double_edges=double_edges, R=R, r=r_minor)
    with st.spinner("Building graph..."):
        G, heights, points = build_graph(cfg)

    h_arr   = np.array(heights)
    edges_a = {(u,v) for u,v in G.edges() if abs(G.edges[u,v]['weight']-arc_a) < eps}
    edges_b = {(u,v) for u,v in G.edges() if abs(G.edges[u,v]['weight']-arc_b) < eps}
    only_a  = edges_a - edges_b
    only_b  = edges_b - edges_a
    common  = edges_a & edges_b

    c1,c2,c3 = st.columns(3)
    c1.metric(f"Only at {arc_a/np.pi:.2g}π", len(only_a), help="orange")
    c2.metric(f"Only at {arc_b/np.pi:.2g}π", len(only_b), help="green")
    c3.metric("Both arcs", len(common), help="gray")

    if view_3d:
        node_xyz = [to_xyz(t,p,R,r_minor) for t,p in points]
        data = []
        if show_surface:
            data.append(make_torus_surface(R, r_minor))
        for edge_set, color in [(common,'gray'),(only_a,'orange'),(only_b,'limegreen')]:
            edges = [(u,v,G.edges[u,v]['weight']) for u,v in edge_set
                     if not is_wrap(points,u,v)]
            if not edges: continue
            all_w=[w for _,_,w in edges]
            ts = make_3d_edge_traces(edges, node_xyz, min(all_w), max(all_w), edge_opacity)
            if not use_arrows:
                for t in ts:
                    if hasattr(t,'line'): t.line.color = color
            data.extend(ts)
        all_cmp = [(u,v,G.edges[u,v]['weight']) for u,v in (edges_a|edges_b)
                   if not is_wrap(points,u,v)]
        if show_node_markers:
            data.append(make_3d_node_trace(points, heights, node_xyz))
        if show_edge_markers and all_cmp:
            data.append(make_3d_midpoint_trace(all_cmp, node_xyz, points, heights))
        fig = go.Figure(data=data)
        fig.update_layout(
            scene=scene_layout(),
            title_text=(f"height={height_var!r}, n={n}  |  "
                        f"orange={arc_a/np.pi:.2g}π only, "
                        f"green={arc_b/np.pi:.2g}π only, gray=both"),
            margin=dict(l=0,r=0,t=60,b=0), height=580,
            showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
            hoverlabel=dict(bgcolor='white',font_size=12,font_family='monospace'),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        coords = flat_xy(points)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Arc = {arc_a/np.pi:.2g}π  (orange = unique)",
                                            f"Arc = {arc_b/np.pi:.2g}π  (green = unique)"],
                            horizontal_spacing=0.08)
        add_flat_comparison_panel(fig, G, points, heights, coords,
                                   edges_a, edges_b, 'orange',
                                   edge_opacity, row=1, col=1, showscale=False)
        add_flat_comparison_panel(fig, G, points, heights, coords,
                                   edges_b, edges_a, 'limegreen',
                                   edge_opacity, row=1, col=2, showscale=False)
        fig.add_trace(go.Scatter(
            x=[None],y=[None], mode='markers',
            marker=dict(size=0, color=[float(h_arr.min()),float(h_arr.max())],
                        colorscale='RdBu',
                        colorbar=dict(title='Height',thickness=14,len=0.5,x=1.02),
                        showscale=True),
            showlegend=False,
        ), row=1, col=2)
        fig.update_xaxes(title_text='θ / π', range=[-0.05,2.05],
                         showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(title_text='φ / π', range=[-0.05,2.05],
                         showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_layout(**base_layout(540,
            f"height={height_var!r}, n={n}  |  gray = both arcs"))
        st.plotly_chart(fig, use_container_width=True)

    if show_height_fn:
        st.subheader(f"Height function: {height_var}")
        st.plotly_chart(make_height_heatmap(height_var, R, r_minor), use_container_width=True)

    st.caption(f"n={n}, R={R}, r={r_minor}, height={height_var!r}, "
               f"sampling={sampling!r}, double_edges={double_edges}")


# =============================================================================
# PAGE 3 — Height Comparison
# =============================================================================

elif page == "Height Comparison":
    st.title("Height Function Comparison")
    st.markdown(
        "Same graph, two height functions. "
        "**Orange** = edges only in A, **green** = only in B, **gray** = both (orientation-aware). "
        "Hover edge midpoints for info."
    )

    st.sidebar.header("Graph parameters")
    n            = st.sidebar.slider("n (grid side, n×n points)", 2, 20, 5)
    R            = st.sidebar.slider("R (major radius)", 1.5, 6.0, 3.0, step=0.5)
    r_minor      = st.sidebar.slider("r (minor radius)", 0.3, 2.0, 1.0, step=0.1)
    sampling     = st.sidebar.selectbox("Sampling", SAMPLING_OPTIONS)
    double_edges = st.sidebar.checkbox("Double edges (ties)", value=True)

    st.sidebar.header("Height functions")
    height_a = st.sidebar.selectbox("Height A", HEIGHT_OPTIONS, index=0)
    height_b = st.sidebar.selectbox("Height B", HEIGHT_OPTIONS, index=4)

    st.sidebar.header("Edge filter")
    arc_min = st.sidebar.slider("Min edge length", 0.0, 6.0, 0.0, step=0.05)
    arc_max = st.sidebar.slider("Max edge length", 0.1, 10.0, 10.0, step=0.05)

    st.sidebar.header("View")
    view_3d        = st.sidebar.checkbox("3D torus view", value=False)
    edge_opacity   = st.sidebar.slider("Edge opacity", 0.1, 1.0, 0.6, step=0.05)
    show_surface   = st.sidebar.checkbox("Show torus surface", value=True)
    show_height_fn = st.sidebar.checkbox("Show height function heatmaps", value=False)
    run_pph        = st.sidebar.checkbox("Compute & compare PPH barcodes", value=False)

    cfg_a = Config(n=n, height_variant=height_a, sampling=sampling,
                   double_edges=double_edges, R=R, r=r_minor)
    cfg_b = Config(n=n, height_variant=height_b, sampling=sampling,
                   double_edges=double_edges, R=R, r=r_minor)

    with st.spinner("Building graphs..."):
        G_a, heights_a, points_a = build_graph(cfg_a)
        G_b, heights_b, points_b = build_graph(cfg_b)

    # Directed edge sets (orientation-aware)
    def filtered_edges(G):
        return {(u,v) for u,v in G.edges()
                if arc_min <= G.edges[u,v]['weight'] <= arc_max}

    set_a = filtered_edges(G_a)
    set_b = filtered_edges(G_b)
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

    # We show the same node positions (points are identical since same n/sampling/seed)
    # but different edge directions from each height function
    coords_a = flat_xy(points_a)
    coords_b = flat_xy(points_b)

    if view_3d:
        col1, col2 = st.columns(2)
        for col, G, heights, points, label, edge_set, ref_set, uniq_col in [
            (col1, G_a, heights_a, points_a, f"Height A: {height_a}", set_a, set_b, 'orange'),
            (col2, G_b, heights_b, points_b, f"Height B: {height_b}", set_b, set_a, 'limegreen'),
        ]:
            node_xyz = [to_xyz(t,p,R,r_minor) for t,p in points]
            only_here = edge_set - ref_set
            comm_here = edge_set & ref_set

            data = []
            if show_surface:
                data.append(make_torus_surface(R, r_minor))

            for eset, color in [(comm_here,'gray'),(only_here, uniq_col)]:
                edges = [(u,v,G.edges[u,v]['weight']) for u,v in eset
                         if not is_wrap(points,u,v)
                         and arc_min <= G.edges[u,v]['weight'] <= arc_max]
                if not edges: continue
                all_w=[w for _,_,w in edges]
                ts = make_3d_edge_traces(edges, node_xyz, min(all_w), max(all_w), edge_opacity)
                if not use_arrows:
                    for t in ts:
                        if hasattr(t,'line'): t.line.color = color
                data.extend(ts)

            all_drawn = [(u,v,G.edges[u,v]['weight']) for u,v in edge_set
                         if not is_wrap(points,u,v)
                         and arc_min <= G.edges[u,v]['weight'] <= arc_max]
            if show_node_markers:
                data.append(make_3d_node_trace(points, heights, node_xyz))
            if show_edge_markers and all_drawn:
                data.append(make_3d_midpoint_trace(all_drawn, node_xyz, points, heights))

            fig = go.Figure(data=data)
            fig.update_layout(
                scene=scene_layout(), title_text=label,
                margin=dict(l=0,r=0,t=40,b=0), height=500,
                showlegend=False, paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(bgcolor='white',font_size=11,font_family='monospace'),
            )
            with col:
                st.plotly_chart(fig, use_container_width=True)
    else:
        h_arr_a = np.array(heights_a)
        h_arr_b = np.array(heights_b)
        all_h = np.concatenate([h_arr_a, h_arr_b])
        h_min, h_max = float(all_h.min()), float(all_h.max())

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f"Height A: {height_a}  (orange = only in A)",
                                            f"Height B: {height_b}  (green = only in B)"],
                            horizontal_spacing=0.08)

        # Panel A: show set_a edges, coloured by membership vs set_b
        add_flat_comparison_panel(fig, G_a, points_a, heights_a, coords_a,
                                   set_a, set_b, 'orange',
                                   edge_opacity, row=1, col=1, showscale=False)
        # Panel B: show set_b edges, coloured by membership vs set_a
        add_flat_comparison_panel(fig, G_b, points_b, heights_b, coords_b,
                                   set_b, set_a, 'limegreen',
                                   edge_opacity, row=1, col=2, showscale=False)

        # Shared colorbar
        fig.add_trace(go.Scatter(
            x=[None],y=[None], mode='markers',
            marker=dict(size=0, color=[h_min, h_max], colorscale='RdBu',
                        colorbar=dict(title='Height',thickness=14,len=0.5,x=1.02),
                        showscale=True),
            showlegend=False,
        ), row=1, col=2)

        fig.update_xaxes(title_text='θ / π', range=[-0.05,2.05],
                         showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_yaxes(title_text='φ / π', range=[-0.05,2.05],
                         showgrid=True, gridcolor='rgba(128,128,128,0.15)')
        fig.update_layout(**base_layout(540,
            f"n={n}, sampling={sampling!r}  |  gray = same directed edge in both"))
        st.plotly_chart(fig, use_container_width=True)

    if show_height_fn:
        st.subheader("Height function heatmaps")
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            st.markdown(f"**{height_a}**")
            st.plotly_chart(make_height_heatmap(height_a, R, r_minor),
                            use_container_width=True)
        with hcol2:
            st.markdown(f"**{height_b}**")
            st.plotly_chart(make_height_heatmap(height_b, R, r_minor),
                            use_container_width=True)

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
                st.plotly_chart(make_barcode_figure(result, height_400=False),
                                use_container_width=True)
                with st.expander("Raw barcode"):
                    for b, d in result.barcode:
                        d_str = f"{d/np.pi:.4f}π" if np.isfinite(d) else "∞"
                        st.text(f"  [{b/np.pi:.4f}π,  {d_str}]")

    st.caption(f"n={n}, R={R}, r={r_minor}, sampling={sampling!r}, "
               f"double_edges={double_edges}, arc ∈ [{arc_min:.2f}, {arc_max:.2f}]")
