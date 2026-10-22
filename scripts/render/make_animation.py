"""Animation for the README: the Re = 1000 cavity spinning up from rest to its steady vortex.

Runs the repository's solverLibs kernels (as run_cavity_re1000.py, on a 128 x 128 grid) and
draws snapshots that are dense early, when the lid vortex forms and travels into the cavity,
and sparse later, when the flow settles. Same look as the still figure: line integral
convolution of the flow direction, coloured by vorticity with signed_dark (teal clockwise,
red anticlockwise, zero fading into the black). A slowly moving ripple in the convolution
kernel makes the texture drift along the streamlines.
Writes docs/assets/animation.webp.  Run:  python scripts/render/make_animation.py
"""
import sys
import time
from pathlib import Path

import numba
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from gif import save_gif  # noqa: E402
from render_cavity import LID, WALL, sample, signed_dark  # noqa: E402
from solverLibs.functions import (CorretorStep, bc, initializeFields,  # noqa: E402
                                  initializeGrid, predictorStep,
                                  solvePoissonEquation_2dDCT)

N, RE, CO, T_END, FRAMES = 128, 1000.0, 0.25, 36.0, 110
SIDE = 520                      # cavity size in pixels; the GIF is a little larger with the walls
OMEGA_SCALE = 2.5


@numba.njit(parallel=True, cache=True)
def lic(ux, uy, noise, length, h, phase, ripple):
    """LIC with a travelling ripple in the kernel (ripple = 0 gives plain LIC)."""
    ny, nx = noise.shape
    out = np.zeros_like(noise)
    for j in numba.prange(ny):
        for i in range(nx):
            acc = 0.0
            wsum = 0.0
            for direction in (1.0, -1.0):
                x = i + 0.5
                y = j + 0.5
                for k in range(length):
                    if direction < 0 and k == 0:
                        continue
                    ix = int(x)
                    iy = int(y)
                    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                        break
                    w = (0.5 + 0.5 * np.cos(np.pi * k / length)) * \
                        (1.0 + ripple * np.cos(2 * np.pi * direction * k / length - phase))
                    acc += w * noise[iy, ix]
                    wsum += w
                    xm = x + 0.5 * direction * h * ux[iy, ix]
                    ym = y + 0.5 * direction * h * uy[iy, ix]
                    ixm = int(xm)
                    iym = int(ym)
                    if ixm < 0 or iym < 0 or ixm >= nx or iym >= ny:
                        break
                    x += direction * h * ux[iym, ixm]
                    y += direction * h * uy[iym, ixm]
            out[j, i] = acc / wsum if wsum > 0 else 0.0
    return out


def snapshots():
    """Integrate from rest; return (t, u_face, v_face) at times dense near t = 0."""
    x, y, dx, dy = initializeGrid(N, N, 1.0, 1.0)
    dt = CO * dx
    u, v, _, _, p = initializeFields(N, N)
    targets = T_END * (np.arange(1, FRAMES + 1) / FRAMES) ** 1.8
    out, step, t0 = [], 0, time.time()
    for target in targets:
        while step * dt < target:
            u, v = bc(u, v, 0, 0, 1, 0, 0, 0, 0, 0)
            p = solvePoissonEquation_2dDCT(predictorStep(u, v, dx, dy, dt, RE), N, N, dx, dy)
            u, v = CorretorStep(p, u, v, dx, dy)
            step += 1
        u, v = bc(u, v, 0, 0, 1, 0, 0, 0, 0, 0)
        out.append((step * dt, u.copy(), v.copy()))
    print(f"{step} steps to t = {step * dt:.1f} in {time.time() - t0:.0f} s", flush=True)
    return out, dx


def frame(u, v, dx, noise, phase):
    ucc = (u[0:-1, 1:-1] + u[1:, 1:-1]) / 2
    vcc = (v[1:-1, 0:-1] + v[1:-1, 1:]) / 2
    wz = (v[1:, :] - v[:-1, :]) / dx - (u[:, 1:] - u[:, :-1]) / dx
    wcc = 0.25 * (wz[:-1, :-1] + wz[1:, :-1] + wz[:-1, 1:] + wz[1:, 1:])
    us, vs, w = sample(ucc, N, SIDE), sample(vcc, N, SIDE), sample(wcc, N, SIDE)
    speed = np.hypot(us, vs) + 1e-12
    ux, uy = us / speed, -vs / speed
    tex = lic(ux, uy, noise, 22, 0.7, phase, 0.6)
    tex = lic(ux, uy, tex, 8, 0.7, 0.0, 0.0)
    lo, hi = np.percentile(tex, [1, 99])
    tex = np.clip((tex - lo) / (hi - lo + 1e-12), 0, 1)
    rgb = signed_dark(0.5 + 0.5 * np.tanh(w / OMEGA_SCALE)) * (0.45 + 0.55 * tex)[..., None]

    wall, lid, m = 3, 8, 14
    canvas = Image.new("RGB", (SIDE + 2 * m, SIDE + 2 * m + lid), (0, 0, 0))
    x0, y0 = m, m + lid
    canvas.paste(Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)), (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x0 - wall, y0, x0 + SIDE + wall - 1, y0 + SIDE + wall - 1], outline=WALL, width=wall)
    draw.rectangle([x0 - wall, y0 - lid, x0 + SIDE + wall - 1, y0 - 1], fill=LID)
    return canvas


def main():
    states, dx = snapshots()
    rng = np.random.default_rng(7)
    noise = np.kron(rng.random((SIDE // 4, SIDE // 4)), np.ones((4, 4)))
    frames = [frame(u, v, dx, noise, 2 * np.pi * k / 12) for k, (_, u, v) in enumerate(states)]
    _, u, v = states[-1]                 # dwell on the steady state, texture still drifting
    frames += [frame(u, v, dx, noise, 2 * np.pi * k / 12) for k in range(len(states), len(states) + 24)]
    save_gif(frames, ROOT / "docs" / "assets" / "animation.webp", fps=20, width=480)


if __name__ == "__main__":
    main()
