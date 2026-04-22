"""Check death times for n=6,8: constant vs double_max."""
import importlib
import numpy as np

mod = importlib.import_module("circle_pph_2")

for n in [6, 8]:
    print(f"=== n = {n} ===")
    for variant in ['constant', 'double_max']:
        for de in [False, True]:
            r = mod.analyze_simple(n, double_edges=de, height_variant=variant)
            bars = []
            for b in r.barcode:
                birth = round(b[0] / np.pi, 4)
                death = round(b[1] / np.pi, 4) if np.isfinite(b[1]) else "inf"
                bars.append(f"[{birth}pi, {death}pi]")
            de_label = "DE" if de else "no DE"
            print(f"  {variant:10s}  {de_label:5s}  bars={r.n_bars}  edges={r.n_edges}  {', '.join(bars) if bars else '(none)'}")
    print()
