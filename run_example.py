"""Headless wrapper around the unchanged legacy cavity driver."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from solver_lidDrivenCavityFlow import solve_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=24)
    parser.add_argument("--steps", type=int, default=31)
    parser.add_argument("--reynolds", type=float, default=100)
    parser.add_argument("--courant", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    args = parser.parse_args()
    if args.grid < 4 or args.steps < 1 or not np.isfinite([args.reynolds, args.courant]).all() or min(args.reynolds, args.courant) <= 0:
        parser.error("grid >= 4, steps >= 1 and finite positive flow parameters are required")
    # dx=dy=1/grid, dt=Co*dx. Necessary explicit diffusion restriction.
    if args.courant > min(0.5, args.reynolds/(4*args.grid)):
        parser.error("Courant number exceeds the convective/diffusive safety bound")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("outputs_*.npz")):
        parser.error("output already contains snapshots; choose a fresh output directory")
    solve_flow(1, 1, args.grid, args.grid, args.steps, args.reynolds, args.courant, str(args.output.resolve()))
    snapshots = sorted(args.output.glob("outputs_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    with np.load(snapshots[-1], allow_pickle=False) as data:
        u, v = data["ucc"].T, data["vcc"].T
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise RuntimeError("non-finite velocity; reduce the time step")
    x = (np.arange(args.grid)+0.5)/args.grid
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    field = ax.contourf(x, x, np.hypot(u, v), 20, cmap="viridis")
    ax.streamplot(x, x, u, v, color="white", linewidth=0.6, density=0.9)
    ax.set(xlabel="x/L", ylabel="y/L", title=f"Lid-driven cavity · Re = {args.reynolds:g}", aspect="equal")
    fig.colorbar(field, ax=ax, label="speed / lid speed")
    fig.savefig(args.output / "velocity.svg")
    plt.close("all")
    print(f"Wrote {len(snapshots)} snapshots and velocity.svg to {args.output.resolve()}")


if __name__ == "__main__":
    main()
