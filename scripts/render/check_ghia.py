"""Compare the steady Re = 1000 centreline velocities with Ghia, Ghia & Shin (1982)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "re1000" / "steady.npz"
d = np.load(path)
n = int(d["grid"])
s = (np.arange(n) + 0.5) / n
u_mid = 0.5 * (d["ucc"][n // 2 - 1, :] + d["ucc"][n // 2, :])   # u(x = 0.5, y)
v_mid = 0.5 * (d["vcc"][:, n // 2 - 1] + d["vcc"][:, n // 2])   # v(x, y = 0.5)

# Ghia et al. (1982), J. Comput. Phys. 48, 387, Tables I and II, Re = 1000 (subset)
ghia_u = [(0.0547, -0.18109), (0.1719, -0.38289), (0.5000, -0.06080), (0.7344, 0.18719), (0.9531, 0.47244)]
ghia_v = [(0.1563, 0.37095), (0.5000, 0.02526), (0.8047, -0.31966), (0.9063, -0.51550)]
print(f"grid {n}, t = {float(d['t']):.1f}, last max|du/dt| = {d['history'][-1, 1]:.2e}")
for name, prof, ref in (("u(0.5, y)", u_mid, ghia_u), ("v(x, 0.5)", v_mid, ghia_v)):
    for pos, val in ref:
        here = np.interp(pos, s, prof)
        print(f"{name} at {pos:.4f}: {here:+.4f}  Ghia {val:+.4f}  diff {here - val:+.4f}")
