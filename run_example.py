"""Run the lid-driven cavity to steady state and plot it (headless; MAIN.py is the window-based interface).

Uses the solverLibs kernels of solve_flow() (predictor, DCT pressure solve, corrector) without
the legacy one-second pause at every write, and stops when max |du/dt| drops below --tol or at
--t-end. Writes outputs/quickstart/velocity.svg and steady.npz.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from solverLibs.functions import (CorretorStep, bc, initializeFields, initializeGrid,  # noqa: E402
                                  predictorStep, solvePoissonEquation_2dDCT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--reynolds", type=float, default=100)
    parser.add_argument("--courant", type=float, default=0.5)
    parser.add_argument("--t-end", type=float, default=30.0, help="maximum time in lid-transit units")
    parser.add_argument("--tol", type=float, default=1e-5, help="steady-state tolerance on max |du/dt|")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many steps instead (a quick check)")
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    args = parser.parse_args()
    n, re = args.grid, args.reynolds
    if n < 4 or not np.isfinite([re, args.courant]).all() or min(re, args.courant) <= 0:
        parser.error("grid >= 4 and finite positive flow parameters are required")
    # dx = dy = 1/grid, dt = Co dx; explicit convective/diffusive restriction
    if args.courant > min(0.5, re / (4 * n)):
        parser.error(f"Courant number exceeds the convective/diffusive safety bound {min(0.5, re / (4 * n)):.3g}")

    x, y, dx, dy = initializeGrid(n, n, 1.0, 1.0)
    dt = args.courant * dx
    u, v, _, _, p = initializeFields(n, n)
    steps = args.steps or int(np.ceil(args.t_end / dt))
    check = max(1, int(round(0.5 / dt)))
    u_prev, rate = None, np.inf
    for step in range(1, steps + 1):
        u, v = bc(u, v, 0, 0, 1, 0, 0, 0, 0, 0)
        p = solvePoissonEquation_2dDCT(predictorStep(u, v, dx, dy, dt, re), n, n, dx, dy)
        u, v = CorretorStep(p, u, v, dx, dy)
        if step % check == 0:
            if not np.isfinite(u).all():
                raise RuntimeError("non-finite velocity; reduce the Courant number")
            if u_prev is not None:
                rate = np.abs(u - u_prev).max() / (check * dt)
                if rate < args.tol:
                    break
            u_prev = u.copy()
    u, v = bc(u, v, 0, 0, 1, 0, 0, 0, 0, 0)
    ucc = ((u[0:-1, 1:-1] + u[1:, 1:-1]) / 2).T
    vcc = ((v[1:-1, 0:-1] + v[1:-1, 1:]) / 2).T

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "steady.npz", ucc=ucc, vcc=vcc, reynolds=re, grid=n, t=step * dt)
    xc = (np.arange(n) + 0.5) / n
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    field = ax.contourf(xc, xc, np.hypot(ucc, vcc), 20, cmap="viridis")
    ax.streamplot(xc, xc, ucc, vcc, color="white", linewidth=0.6, density=1.2)
    ax.set(xlabel="x/L", ylabel="y/L", title=f"Lid-driven cavity · Re = {re:g} · t = {step * dt:.1f}", aspect="equal")
    fig.colorbar(field, ax=ax, label="speed / lid speed")
    fig.savefig(args.output / "velocity.svg")
    plt.close("all")
    state = "steady" if rate < args.tol else f"max |du/dt| = {rate:.1e}"
    print(f"Re = {re:g}, {n} x {n} cells: {step} steps to t = {step * dt:.1f} ({state}).")
    print(f"Figure: {(args.output / 'velocity.svg').resolve()}")


if __name__ == "__main__":
    main()
