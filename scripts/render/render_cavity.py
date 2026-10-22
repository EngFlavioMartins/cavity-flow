"""Line-integral-convolution render of the steady Re = 1000 cavity flow, OpenONDA style.

Reads the output of run_cavity_re1000.py. Texture: LIC of the unit velocity
direction, so slow corner eddies show as clearly as the primary vortex.
Colour: vorticity with signed_dark (teal clockwise, red anticlockwise, zero
fading into the black), compressed with tanh so the thin lid boundary layer
does not swamp the interior. Writes the raw image; compose_cavity.py adds
the LaTeX colour bar.
"""
import argparse
from pathlib import Path

import numba
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from onda_style import STOPS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BLACK = (0, 0, 0)
WALL = (96, 104, 108)     # OpenONDA surface grey
LID = (163, 163, 163)     # OpenONDA light grey for the moving lid


def signed_dark(t):
    """signed_dark from onda_style, vectorised over t in [0, 1]."""
    stops = np.array([[int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] for c in STOPS["signed_dark"]])
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


@numba.njit(parallel=True, cache=True)
def lic(ux, uy, noise, length, h):
    """LIC on a pixel grid: ux, uy are unit vectors (row = y, col = x)."""
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
                        continue  # centre sample counted once
                    ix = int(x)
                    iy = int(y)
                    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                        break
                    w = 0.5 + 0.5 * np.cos(np.pi * k / length)  # Hann kernel
                    acc += w * noise[iy, ix]
                    wsum += w
                    vx = ux[iy, ix]
                    vy = uy[iy, ix]
                    # midpoint (RK2) step
                    xm = x + 0.5 * direction * h * vx
                    ym = y + 0.5 * direction * h * vy
                    ixm = int(xm)
                    iym = int(ym)
                    if ixm < 0 or iym < 0 or ixm >= nx or iym >= ny:
                        break
                    x += direction * h * ux[iym, ixm]
                    y += direction * h * uy[iym, ixm]
            out[j, i] = acc / wsum if wsum > 0 else 0.0
    return out


def sample(field, n_cells, px):
    """Bicubic resample of a cell-centred (x, y)-indexed field to px × px, row 0 at the top."""
    s = (np.arange(px) + 0.5) / px * n_cells - 0.5
    yy, xx = np.meshgrid(s[::-1], s, indexing="ij")
    return map_coordinates(field, [xx, yy], order=3, mode="nearest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "re1000" / "steady.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "render" / "cavity-flow-raw.png")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--omega-scale", type=float, default=2.5, help="tanh compression scale for vorticity")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    d = np.load(args.input)
    n, dx = int(d["grid"]), float(d["dx"])
    uf, vf = d["u_face"], d["v_face"]
    # vorticity at cell corners, then averaged to cell centres
    wz = (vf[1:, :] - vf[:-1, :]) / dx - (uf[:, 1:] - uf[:, :-1]) / dx  # (n+1, n+1)
    wcc = 0.25 * (wz[:-1, :-1] + wz[1:, :-1] + wz[:-1, 1:] + wz[1:, 1:])

    width = args.width
    height = width * 2 // 3                       # 3:2 frame
    side = int(height * 0.84)                     # cavity square with a generous margin
    ss = 2                                        # supersampling for the texture
    px = side * ss

    u = sample(d["ucc"], n, px)
    v = sample(d["vcc"], n, px)
    w = sample(wcc, n, px)
    speed = np.hypot(u, v) + 1e-12
    ux, uy = u / speed, -v / speed                # image rows grow downward

    rng = np.random.default_rng(args.seed)
    noise = rng.random((px // 2, px // 2)).astype(np.float64)
    noise = np.kron(noise, np.ones((2, 2)))       # 2-px grains survive downsampling
    tex = lic(ux, uy, noise, 28 * ss, 0.7)
    tex = lic(ux, uy, tex, 14 * ss, 0.7)          # second pass smooths the grain
    lo, hi = np.percentile(tex, [1, 99])
    tex = np.clip((tex - lo) / (hi - lo), 0, 1)

    c = signed_dark(0.5 + 0.5 * np.tanh(w / args.omega_scale))
    shade = 0.45 + 0.55 * tex
    rgb = np.clip(c * shade[..., None], 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((side, side), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), BLACK)
    x0, y0 = int(0.12 * width), (height - side) // 2 + int(height * 0.02)
    canvas.paste(img, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    wall = max(2, side // 260)
    draw.rectangle([x0 - wall, y0, x0 + side + wall - 1, y0 + side + wall - 1], outline=WALL, width=wall)
    lid = max(6, side // 70)
    draw.rectangle([x0 - wall, y0 - lid, x0 + side + wall - 1, y0 - 1], fill=LID)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
