"""Plot death times of the main cycle for 4 configurations on T²,
plus heatmaps of the height functions for context."""

import numpy as np
import matplotlib.pyplot as plt
from torus_pph import batch_analyze, MOD4_COLORS, height, plot_height_function

n_range = range(2, 13)

# Compute all 4 configurations
configs = {
    ('sin_theta',    False): batch_analyze(n_range, height_variant='sin_theta',    double_edges=False),
    ('sin_theta',    True):  batch_analyze(n_range, height_variant='sin_theta',    double_edges=True),
    ('sin_plus_sin', False): batch_analyze(n_range, height_variant='sin_plus_sin', double_edges=False),
    ('sin_plus_sin', True):  batch_analyze(n_range, height_variant='sin_plus_sin', double_edges=True),
}

titles = {
    ('sin_theta',    False): 'f=sin(θ), no double edges',
    ('sin_theta',    True):  'f=sin(θ), double edges',
    ('sin_plus_sin', False): 'f=sin(θ)+sin(φ), no double edges',
    ('sin_plus_sin', True):  'f=sin(θ)+sin(φ), double edges',
}

# --- Figure 1: Height function heatmaps ---
fig0, axes0 = plt.subplots(1, 2, figsize=(13, 5))

theta = np.linspace(0, 2 * np.pi, 300)
phi = np.linspace(0, 2 * np.pi, 300)
TH, PH = np.meshgrid(theta, phi)

for ax, variant, label in [
    (axes0[0], 'sin_theta',    'f(θ,φ) = sin(θ)'),
    (axes0[1], 'sin_plus_sin', 'f(θ,φ) = sin(θ) + sin(φ)'),
]:
    Z = np.vectorize(lambda t, p: height(t, p, variant))(TH, PH)
    im = ax.contourf(TH / np.pi, PH / np.pi, Z, levels=40, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='f(θ, φ)')
    ax.set_xlabel('θ / π')
    ax.set_ylabel('φ / π')
    ax.set_title(label, fontweight='bold')

fig0.suptitle('Height Functions on T²', fontweight='bold', fontsize=13)
fig0.tight_layout()
fig0.savefig('torus_height_functions.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: torus_height_functions.png")

# --- Figure 2: ALL death times (every bar), 2×2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
panel_keys = [
    ('sin_theta',    False), ('sin_theta',    True),
    ('sin_plus_sin', False), ('sin_plus_sin', True),
]

for ax, key in zip(axes.flat, panel_keys):
    results = configs[key]
    for r in results:
        deaths_raw = [bar[1] / np.pi for bar in r.barcode]

        if not deaths_raw:
            continue

        finite_deaths = [d for d in deaths_raw if np.isfinite(d)]
        all_infinite = len(finite_deaths) == 0

        max_death = max(finite_deaths) if finite_deaths else 1.0

        is_infinite = [not np.isfinite(d) for d in deaths_raw]
        deaths_raw = [d if np.isfinite(d) else max_death for d in deaths_raw]

        # Group by multiplicity
        deaths_sorted = sorted(deaths_raw)
        groups = []
        cur_val, cur_count = deaths_sorted[0], 1
        for d in deaths_sorted[1:]:
            if abs(d - cur_val) < 1e-8:
                cur_count += 1
            else:
                groups.append((cur_val, cur_count))
                cur_val, cur_count = d, 1
        groups.append((cur_val, cur_count))

        mod4 = r.n % 4
        for death, count in groups:
            size = 40 * count
            marker = 'X' if all_infinite else 'o'
            edge_color = 'red' if all_infinite else ('black' if count > 1 else 'none')
            edge_width = 1.5 if all_infinite else (0.8 if count > 1 else 0)

            ax.scatter(r.n, death, c=MOD4_COLORS[mod4], s=size, alpha=0.7,
                       marker=marker, edgecolors=edge_color, linewidths=edge_width)

            if count > 1 or all_infinite:
                label_text = f"{count}*∞" if all_infinite else str(count)
                ax.annotate(label_text, (r.n, death), fontsize=7, fontweight='bold',
                            ha='center', va='center', color='white')

    ax.set_title(titles[key], fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

axes[1, 0].set_xlabel('n (grid side length, n×n points)')
axes[1, 1].set_xlabel('n (grid side length, n×n points)')
axes[0, 0].set_ylabel('Death time (units of π)')
axes[1, 0].set_ylabel('Death time (units of π)')

# Shared legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=MOD4_COLORS[i], markersize=10,
           label=f'n ≡ {i} (mod 4)') for i in range(4)]
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=12, markeredgecolor='black', markeredgewidth=0.8,
               label='×k (multiplicity)'))
handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
               markersize=12, markeredgecolor='red', markeredgewidth=1.5,
               label='All infinite'))
fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('All H₁ Death Times on T²', fontweight='bold', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig('torus_main_cycle_death_times.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: torus_main_cycle_death_times.png")
