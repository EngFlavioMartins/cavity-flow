"""Add the LaTeX colour bar to the cavity render.

Usage: python compose_cavity.py <raw png> <output stem> [omega scale, default 2.5]
The bar follows the tanh compression of render_cavity.py, so its ticks
are placed at their true vorticity values.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlay import compose  # noqa: E402

raw, stem = Path(sys.argv[1]), Path(sys.argv[2])
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5
frac = lambda w: 0.5 + 0.5 * math.tanh(w / scale)
compose([(raw, None)], stem, bar=dict(name="signed_dark", vmin=-1, vmax=1, title=r"$\omega L/U$",
                                     ticks=[(frac(-5), "$-$5"), (frac(0), "0"), (frac(5), "5")]))
