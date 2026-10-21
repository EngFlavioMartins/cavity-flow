"""Run the repository's cavity kernels at Re = 1000 to steady state for the website render.

Uses the same solverLibs kernels as solve_flow(), without the legacy one-second
pause at every write, and stores only the final cell-centre fields.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from solverLibs.functions import (CorretorStep, bc, initializeFields,  # noqa: E402
                                  initializeGrid, predictorStep,
                                  solvePoissonEquation_2dDCT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--reynolds", type=float, default=1000)
    parser.add_argument("--courant", type=float, default=0.25)
    parser.add_argument("--t-end", type=float, default=120.0, help="maximum time in lid-transit units")
    parser.add_argument("--tol", type=float, default=1e-7, help="steady-state tolerance on max |du/dt|")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "re1000" / "steady.npz")
    args = parser.parse_args()

    n, re = args.grid, args.reynolds
    if args.courant > min(0.5, re / (4 * n)):
        parser.error("Courant number exceeds the convective/diffusive safety bound")
    x, y, dx, dy = initializeGrid(n, n, 1.0, 1.0)
    dt = args.courant * dx
    u, v, _, _, p = initializeFields(n, n)

    history = []
    check_every = max(1, int(round(0.5 / dt)))  # every half time unit
    u_prev = None
    t0 = time.time()
    steps = int(np.ceil(args.t_end / dt))
    for step in range(1, steps + 1):
        u, v = bc(u, v, 0, 0, 1, 0, 0, 0, 0, 0)
        rhs = predictorStep(u, v, dx, dy, dt, re)
        p = solvePoissonEquation_2dDCT(rhs, n, n, dx, dy)
        u, v = CorretorStep(p, u, v, dx, dy)
        if step % check_every == 0:
            if not np.isfinite(u).all():
                raise RuntimeError("non-finite velocity; reduce the Courant number")
            if u_prev is not None:
                rate = np.abs(u - u_prev).max() / (check_every * dt)
                history.append((step * dt, rate))
                print(f"t = {step * dt:7.2f}   max|du/dt| = {rate:.3e}   ({time.time() - t0:.0f} s)", flush=True)
                if rate < args.tol:
                    break
            u_prev = u.copy()

    u, v = bc(u, v, 0, 0, 1, 0, 0, 0, 0, 0)
    ucc = (u[0:-1, 1:-1] + u[1:, 1:-1]) / 2
    vcc = (v[1:-1, 0:-1] + v[1:-1, 1:]) / 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Face velocities are kept so vorticity can be formed on cell corners.
    np.savez_compressed(args.output, u_face=u, v_face=v, ucc=ucc, vcc=vcc, dx=dx, dt=dt,
                        reynolds=re, grid=n, t=step * dt, history=np.array(history))
    print(f"Wrote {args.output} at t = {step * dt:.2f}")


if __name__ == "__main__":
    main()
