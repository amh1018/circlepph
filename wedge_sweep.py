"""
Systematic sweep over wedge_pph parameter combinations.

Produces:
  - wedge_sweep_results.csv
  - wedge_sweep_report.md
"""
from __future__ import annotations

import csv
import math
import statistics
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

import wedge_pph as wp


# ---------------------------------------------------------------------------
# Constants: variant enumerations (mirror TopologyVariant / HeightVariant /
# DistanceVariant in wedge_pph but keep a local copy for clarity).
# ---------------------------------------------------------------------------

TOPOLOGIES = [
    'wedge2', 'wedge3', 'wedge_k', 'theta', 'lollipop', 'eyeglasses',
    'figure8_asymmetric', 'chain', 'necklace', 'necklace_full',
]

HEIGHT_VARIANTS = [
    # local (per-circle angle)
    'standard', 'cos', 'sin2', 'cos2', 'abs_sin', 'triangle', 'sawtooth', 'square',
    # local R² embedding-based
    'x_coord', 'y_coord', 'radial_dist', 'angle_from_top', 'dist_from_right',
    # local modulated
    'dip', 'double_max', 'plateau', 'squeeze', 'shallow', 'damped',
    # global
    'global_x', 'global_y', 'global_r', 'global_angle', 'global_sawtooth_x',
    'global_sin_x', 'global_sin_y', 'global_sin_r', 'global_diagonal',
    'global_saddle', 'global_ripple', 'global_dist_max',
    # 'constant' deliberately excluded (trivial)
]

DISTANCE_VARIANTS = [
    'within_only', 'euclidean', 'geodesic', 'arc_sum', 'global_arc',
]

FIXED_KW = dict(
    double_edges=True,
    total_n=40,
    allocation='uniform',
    sampling='uniform',
    random_seed=0,
)


# ---------------------------------------------------------------------------
# Build the list of topology-configuration "base" dicts to enumerate.
# ---------------------------------------------------------------------------

def topology_base_configs() -> list[dict]:
    """Return the topology-specific parameter dicts to sweep over."""
    bases: list[dict] = []
    for topo in TOPOLOGIES:
        if topo in ('wedge_k', 'chain', 'necklace', 'necklace_full'):
            for k in (2, 3, 4):
                bases.append({'topology': topo, 'k': k})
        elif topo == 'lollipop':
            for b in (0, 2, 4):
                bases.append({'topology': topo, 'bridge_length': b})
        elif topo == 'eyeglasses':
            for b in (0, 2):
                bases.append({'topology': topo, 'bridge_length': b})
        elif topo == 'figure8_asymmetric':
            for radii in ([1.0, 1.0], [1.0, 2.0]):
                bases.append({'topology': topo, 'radii': radii})
        else:
            bases.append({'topology': topo})
    return bases


def expand_jobs() -> list[dict]:
    """Every (topology-config × height × distance) job to run."""
    jobs: list[dict] = []
    for base in topology_base_configs():
        for h in HEIGHT_VARIANTS:
            for d in DISTANCE_VARIANTS:
                job = dict(base)
                job['height_variant'] = h
                job['distance'] = d
                job.update(FIXED_KW)
                jobs.append(job)
    return jobs


# ---------------------------------------------------------------------------
# Worker: run analyze() on a single config dict.
# ---------------------------------------------------------------------------

def _run_one(job: dict) -> dict:
    """Run a single analyze() and return a flat record."""
    record = {
        'topology': job['topology'],
        'k': job.get('k'),
        'bridge_length': job.get('bridge_length'),
        'radii': job.get('radii'),
        'height_variant': job['height_variant'],
        'distance': job['distance'],
        'n_bars': None,
        'max_death_over_pi': None,
        'has_infinite': None,
        'barcode': None,
        'error': None,
    }
    try:
        cfg = wp.Config(**job)
        res = wp.analyze(cfg)
        barcode = list(res.barcode)
        pi = math.pi
        record['n_bars'] = res.n_bars
        record['max_death_over_pi'] = (
            res.max_death / pi if res.n_bars else 0.0
        )
        record['has_infinite'] = any(not math.isfinite(d) for (_, d) in barcode)
        record['barcode'] = [
            (round(b / pi, 3),
             round(d / pi, 3) if math.isfinite(d) else float('inf'))
            for (b, d) in barcode
        ]
    except Exception as exc:  # noqa: BLE001
        record['error'] = f"{type(exc).__name__}: {exc}"
        record['barcode'] = []
        record['n_bars'] = 0
        record['max_death_over_pi'] = 0.0
        record['has_infinite'] = False
        traceback.print_exc()
    return record


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def run_sweep(jobs: list[dict], max_workers: int | None = None) -> list[dict]:
    """Run all jobs in a process pool, printing progress."""
    results: list[dict] = []
    n = len(jobs)
    print(f"Dispatching {n} configurations across process pool …")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            rec = fut.result()
            results.append(rec)
            tag = (
                f"{rec['topology']}"
                + (f"/k={rec['k']}" if rec['k'] is not None else "")
                + (f"/bridge={rec['bridge_length']}"
                   if rec['bridge_length'] is not None else "")
                + (f"/radii={rec['radii']}" if rec['radii'] is not None else "")
                + f" | h={rec['height_variant']} d={rec['distance']}"
            )
            if rec['error']:
                print(f"[{done}/{n}] ERROR  {tag}: {rec['error']}")
            else:
                print(
                    f"[{done}/{n}] OK     {tag}  "
                    f"n_bars={rec['n_bars']} "
                    f"max_death/π={rec['max_death_over_pi']:.3f}"
                )
    return results


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------

def _barcode_str(barcode) -> str:
    if not barcode:
        return ""
    parts = []
    for b, d in barcode:
        d_s = "inf" if d == float('inf') else f"{d:.3f}"
        parts.append(f"({b:.3f},{d_s})")
    return ";".join(parts)


def write_csv(results: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([
            'topology', 'k', 'bridge_length', 'radii',
            'height_variant', 'distance',
            'n_bars', 'max_death_over_pi', 'has_infinite', 'barcode_str',
        ])
        for r in results:
            w.writerow([
                r['topology'],
                '' if r['k'] is None else r['k'],
                '' if r['bridge_length'] is None else r['bridge_length'],
                '' if r['radii'] is None else str(r['radii']),
                r['height_variant'],
                r['distance'],
                r['n_bars'] if r['n_bars'] is not None else '',
                ('' if r['max_death_over_pi'] is None
                 else f"{r['max_death_over_pi']:.6f}"),
                '' if r['has_infinite'] is None else str(r['has_infinite']),
                _barcode_str(r['barcode'] or []),
            ])
    print(f"Wrote {path}")


def _topology_tag(r: dict) -> str:
    tag = r['topology']
    if r['k'] is not None:
        tag += f"(k={r['k']})"
    if r['bridge_length'] is not None:
        tag += f"(bridge={r['bridge_length']})"
    if r['radii'] is not None:
        tag += f"(radii={r['radii']})"
    return tag


def _config_line(r: dict) -> str:
    bits = [f"topology={r['topology']}"]
    if r['k'] is not None:
        bits.append(f"k={r['k']}")
    if r['bridge_length'] is not None:
        bits.append(f"bridge_length={r['bridge_length']}")
    if r['radii'] is not None:
        bits.append(f"radii={r['radii']}")
    bits.append(f"height_variant={r['height_variant']}")
    bits.append(f"distance={r['distance']}")
    return ", ".join(bits)


def write_report(results: list[dict], path: str) -> None:
    valid = [r for r in results if r['error'] is None]

    lines: list[str] = []
    lines.append("# Wedge PPH parameter sweep\n")
    lines.append(
        f"Fixed parameters: double_edges=True, total_n=40, "
        f"allocation='uniform', sampling='uniform'.\n"
    )
    lines.append(
        f"Total configurations run: {len(results)}  "
        f"(successful: {len(valid)}, errored: {len(results) - len(valid)}).\n"
    )

    # -------------------------------------------------------------------
    # Summary table: rows=topology tag, cols=distance, cell=median n_bars
    # across all height variants for that (topology config, distance).
    # -------------------------------------------------------------------
    lines.append("## Summary table\n")
    lines.append(
        "Median `n_bars` across all height variants, by topology "
        "configuration (rows) and distance function (columns).\n"
    )
    topo_tags: list[str] = []
    seen = set()
    for r in valid:
        t = _topology_tag(r)
        if t not in seen:
            seen.add(t)
            topo_tags.append(t)

    header = "| topology | " + " | ".join(DISTANCE_VARIANTS) + " |"
    sep = "|" + "---|" * (1 + len(DISTANCE_VARIANTS))
    lines.append(header)
    lines.append(sep)
    for t in topo_tags:
        row = [t]
        for d in DISTANCE_VARIANTS:
            cell = [r['n_bars'] for r in valid
                    if _topology_tag(r) == t and r['distance'] == d]
            if cell:
                row.append(f"{statistics.median(cell):.1f}")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # -------------------------------------------------------------------
    # Interesting configurations
    # -------------------------------------------------------------------
    lines.append("## Interesting configurations\n")
    lines.append("### Top 10 by n_bars\n")
    top_nbars = sorted(valid, key=lambda r: r['n_bars'], reverse=True)[:10]
    for i, r in enumerate(top_nbars, 1):
        lines.append(
            f"{i}. **n_bars={r['n_bars']}**, max_death/π="
            f"{r['max_death_over_pi']:.3f}"
            + (" (has ∞)" if r['has_infinite'] else "")
        )
        lines.append(f"   - {_config_line(r)}")
        lines.append(f"   - barcode: {_barcode_str(r['barcode'])}")
    lines.append("")

    lines.append("### Top 10 by max_death/π\n")
    top_death = sorted(
        valid,
        key=lambda r: (r['max_death_over_pi'] or 0.0),
        reverse=True,
    )[:10]
    for i, r in enumerate(top_death, 1):
        lines.append(
            f"{i}. **max_death/π={r['max_death_over_pi']:.3f}**, "
            f"n_bars={r['n_bars']}"
            + (" (has ∞)" if r['has_infinite'] else "")
        )
        lines.append(f"   - {_config_line(r)}")
        lines.append(f"   - barcode: {_barcode_str(r['barcode'])}")
    lines.append("")

    # -------------------------------------------------------------------
    # Effect of height variant: wedge2, distance=within_only
    # -------------------------------------------------------------------
    lines.append("## Effect of height variant\n")
    lines.append("For `topology=wedge2`, `distance=within_only`, all height "
                 "variants (sorted by `n_bars` descending):\n")
    subset = [r for r in valid
              if r['topology'] == 'wedge2' and r['distance'] == 'within_only']
    subset.sort(key=lambda r: r['n_bars'], reverse=True)
    lines.append("| height_variant | n_bars | max_death/π |")
    lines.append("|---|---|---|")
    for r in subset:
        lines.append(
            f"| {r['height_variant']} | {r['n_bars']} | "
            f"{r['max_death_over_pi']:.3f} |"
        )
    lines.append("")

    # -------------------------------------------------------------------
    # Effect of distance function: wedge3, height=standard
    # -------------------------------------------------------------------
    lines.append("## Effect of distance function\n")
    lines.append("For `topology=wedge3`, `height_variant=standard`:\n")
    subset = [r for r in valid
              if r['topology'] == 'wedge3' and r['height_variant'] == 'standard']
    # preserve the DISTANCE_VARIANTS ordering
    subset.sort(key=lambda r: DISTANCE_VARIANTS.index(r['distance']))
    lines.append("| distance | n_bars | max_death/π |")
    lines.append("|---|---|---|")
    for r in subset:
        lines.append(
            f"| {r['distance']} | {r['n_bars']} | "
            f"{r['max_death_over_pi']:.3f} |"
        )
    lines.append("")

    # -------------------------------------------------------------------
    # Full results per topology
    # -------------------------------------------------------------------
    lines.append("## Full results per topology\n")
    for topo in TOPOLOGIES:
        topo_results = [r for r in valid if r['topology'] == topo]
        if not topo_results:
            continue
        lines.append(f"### {topo}\n")

        most_bars = max(topo_results, key=lambda r: r['n_bars'])
        lines.append(
            f"**Configuration with most H1 bars:** n_bars="
            f"{most_bars['n_bars']}"
        )
        lines.append(f"- {_config_line(most_bars)}")
        lines.append(f"- barcode: {_barcode_str(most_bars['barcode'])}")
        lines.append("")

        def _longest_bar(r):
            bars = r['barcode'] or []
            best = 0.0
            best_bar = None
            for b, d in bars:
                if d == float('inf'):
                    continue
                life = d - b
                if life > best:
                    best = life
                    best_bar = (b, d)
            return best, best_bar

        longest = max(topo_results, key=lambda r: _longest_bar(r)[0])
        life, bar = _longest_bar(longest)
        if bar is not None:
            lines.append(
                f"**Configuration with longest-lived bar:** "
                f"life={life:.3f}π, (birth,death)/π=({bar[0]:.3f},{bar[1]:.3f})"
            )
        else:
            lines.append(
                "**Configuration with longest-lived bar:** (none finite)"
            )
        lines.append(f"- {_config_line(longest)}")
        lines.append(f"- barcode: {_barcode_str(longest['barcode'])}")
        lines.append("")

        lines.append(
            "Top 5 height variants (by `n_bars`) with `distance=within_only`:\n"
        )
        lines.append("| height_variant | n_bars | max_death/π |")
        lines.append("|---|---|---|")
        wo = [r for r in topo_results if r['distance'] == 'within_only']
        wo.sort(key=lambda r: r['n_bars'], reverse=True)
        for r in wo[:5]:
            lines.append(
                f"| {r['height_variant']} | {r['n_bars']} | "
                f"{r['max_death_over_pi']:.3f} |"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # Errors section (if any)
    # -------------------------------------------------------------------
    errs = [r for r in results if r['error'] is not None]
    if errs:
        lines.append("## Errors\n")
        lines.append(f"{len(errs)} configurations raised an exception.\n")
        lines.append("| topology | height_variant | distance | error |")
        lines.append("|---|---|---|---|")
        for r in errs:
            lines.append(
                f"| {_topology_tag(r)} | {r['height_variant']} | "
                f"{r['distance']} | `{r['error']}` |"
            )
        lines.append("")

    Path(path).write_text("\n".join(lines))
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

def main() -> None:
    jobs = expand_jobs()
    print(f"Total configurations to sweep: {len(jobs)}")
    results = run_sweep(jobs)
    # Sort for stable output
    results.sort(key=lambda r: (
        r['topology'],
        r['k'] if r['k'] is not None else -1,
        r['bridge_length'] if r['bridge_length'] is not None else -1,
        str(r['radii']) if r['radii'] is not None else '',
        r['height_variant'],
        r['distance'],
    ))
    write_csv(results, 'wedge_sweep_results.csv')
    write_report(results, 'wedge_sweep_report.md')


if __name__ == '__main__':
    main()
