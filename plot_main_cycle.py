"""Plot death times of the main cycle for 4 configurations,
plus the height functions for context."""

import importlib
import numpy as np
import matplotlib.pyplot as plt

mod = importlib.import_module("circle_pph-2")
batch_analyze = mod.batch_analyze
MOD4_COLORS = mod.MOD4_COLORS
height = mod.height

n_range = range(3, 51)

# Compute all 4 configurations
configs = {
    ('standard', False): batch_analyze(n_range, height_variant='standard', double_edges=False),
    ('standard', True):  batch_analyze(n_range, height_variant='standard', double_edges=True),
    ('double_max', False): batch_analyze(n_range, height_variant='double_max', double_edges=False),
    ('double_max', True):  batch_analyze(n_range, height_variant='double_max', double_edges=True),
}

titles = {
    ('standard', False):   'Standard f, no double edges',
    ('standard', True):    'Standard f, double edges',
    ('double_max', False): 'Modified f, no double edges',
    ('double_max', True):  'Modified f, double edges',
}

# --- Figure 1: Height functions ---
fig0, ax0 = plt.subplots(figsize=(8, 4))
theta = np.linspace(0, 2*np.pi, 500)
h_std = [height(t, 'standard') for t in theta]
h_dmax = [height(t, 'double_max') for t in theta]
ax0.plot(theta/np.pi, h_std, label='Standard f = sin(θ)', lw=2)
ax0.plot(theta/np.pi, h_dmax, label='Modified f (double_max)', lw=2, ls='--')
ax0.axvline(x=0.5, color='gray', ls=':', alpha=0.5, label='θ = π/2')
ax0.set_xlabel('θ / π')
ax0.set_ylabel('f(θ)')
ax0.set_title('Height Functions on S¹', fontweight='bold')
ax0.legend()
ax0.grid(True, alpha=0.3)
fig0.tight_layout()
fig0.savefig('height_functions.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: height_functions.png")

# --- Figure 2: ALL death times (every bar), 2x2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
panel_keys = [
    ('standard', False), ('standard', True),
    ('double_max', False), ('double_max', True),
]

for ax, key in zip(axes.flat, panel_keys):
    results = configs[key]
    for r in results:
        # Extract all deaths (keep infinite)
        deaths_raw = [bar[1]/np.pi for bar in r.barcode]
        
        if not deaths_raw:
            continue
        
        # Find max finite death time, use as cap for infinite deaths
        finite_deaths = [d for d in deaths_raw if np.isfinite(d)]
        all_infinite = len(finite_deaths) == 0
        
        if finite_deaths:
            max_death = max(finite_deaths)
        else:
            max_death = 1.0  # fallback if all are infinite
        
        # Track which deaths are infinite before replacing them
        is_infinite = [not np.isfinite(d) for d in deaths_raw]
        deaths_raw = [d if np.isfinite(d) else max_death for d in deaths_raw]
        
        # Count multiplicities (group deaths within 1e-8)
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

        for death, count in groups:
            # Scale marker size by multiplicity
            size = 40 * count
            marker = 'X' if all_infinite else 'o'  # Star marker for all infinite
            edge_color = 'red' if all_infinite else ('black' if count > 1 else 'none')
            edge_width = 1.5 if all_infinite else (0.8 if count > 1 else 0)
            
            ax.scatter(r.n, death, c=MOD4_COLORS[r.mod4], s=size, alpha=0.7,
                       marker=marker, edgecolors=edge_color, linewidths=edge_width)
            
            if count > 1 or all_infinite:
                label_text = f"{count}*∞" if all_infinite else str(count)
                ax.annotate(label_text, (r.n, death), fontsize=7, fontweight='bold',
                            ha='center', va='center', color='white')

    ax.axhline(y=0.5, color='black', ls='--', lw=1.5, label='π/2')
    ax.set_title(titles[key], fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

axes[1, 0].set_xlabel('n (sample size)')
axes[1, 1].set_xlabel('n (sample size)')
axes[0, 0].set_ylabel('Death time (units of π)')
axes[1, 0].set_ylabel('Death time (units of π)')

# Shared legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=MOD4_COLORS[i], markersize=10,
           label=f'n ≡ {i} (mod 4)') for i in range(4)]
handles.append(plt.Line2D([0], [0], color='black', ls='--', lw=1.5, label='π/2'))
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=12, markeredgecolor='black', markeredgewidth=0.8,
               label='×k (multiplicity)'))
handles.append(plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
               markersize=12, markeredgecolor='red', markeredgewidth=1.5,
               label='All infinite'))
fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('All H₁ Death Times on S¹', fontweight='bold', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig('main_cycle_death_times.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: main_cycle_death_times.png")