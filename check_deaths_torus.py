"""Check death times for n=3,4,5 on T²: sin_theta vs sin_plus_sin."""
import numpy as np
from torus_pph import analyze_simple

for n in [3, 4, 5]:
    print(f"=== n = {n} ({n}×{n} = {n*n} pts) ===")
    for variant in ['sin_theta', 'sin_plus_sin', 'constant']:
        for de in [False, True]:
            r = analyze_simple(n, double_edges=de, height_variant=variant)
            bars = []
            for b in r.barcode:
                birth = round(b[0] / np.pi, 4)
                death = round(b[1] / np.pi, 4) if np.isfinite(b[1]) else "inf"
                bars.append(f"[{birth}pi, {death}pi]")
            de_label = "DE" if de else "no DE"
            print(f"  {variant:12s}  {de_label:5s}  bars={r.n_bars}  edges={r.n_edges}"
                  f"  {', '.join(bars) if bars else '(none)'}")
    print()
