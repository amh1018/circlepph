# circlepph

Persistent path homology on circles, tori, and wedge/glued topologies.

## Modules

- `circle_pph_2.py` — PPH on S¹
- `torus_pph.py` — PPH on T²
- `wedge_pph.py` — wedges, theta graphs, lollipops, eyeglasses, necklaces
- `wedge_sweep.py` — parameter sweeps over wedge topologies
- `torus_app.py`, `wedge_app.py` — interactive apps

Results are cached to `.homology_cache/` (circle/wedge) and `.pph_cache/` (torus)
using an MD5 hash of the `Config` dataclass as the key, so repeated runs with
the same parameters return instantly.

## Setup

Requires Python 3.12+ and the `grpphati` PPH library (which pulls in a Rust
backend, LoPHAT).

```bash
# Clone
git clone <your-repo-url> circlepph
cd circlepph

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Syncing across multiple machines

Use git. Caches and generated figures are excluded via `.gitignore` — each
machine rebuilds its own `.homology_cache/` on demand.

```bash
# Start of a session
git pull

# End of a session
git add -A
git commit -m "…"
git push
```

### Notebooks

Jupyter notebook outputs create huge, noisy diffs (base64-encoded images).
Either clear outputs manually before committing:

```bash
jupyter nbconvert --clear-output --inplace *.ipynb
```

or install `nbstripout` once per clone to do it automatically:

```bash
pip install nbstripout
nbstripout --install
```

## Usage

```bash
# Quick test
python circle_pph_2.py --quick
python torus_pph.py --quick

# Single analysis
python torus_pph.py --n 6 --height sin_plus_sin

# Full sweep
python wedge_sweep.py
```
