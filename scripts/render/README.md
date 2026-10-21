# Website render

These scripts make the cavity image on [engflaviomartins.github.io](https://engflaviomartins.github.io/).

```bash
pip install numpy scipy numba cmcrameri pillow
python scripts/render/run_cavity_re1000.py     # 192 × 192 grid, Re = 1000, about 4 minutes
python scripts/render/check_ghia.py            # centreline velocities against Ghia et al. (1982)
python scripts/render/render_cavity_lic.py     # writes outputs/render/cavity-flow.png and .webp
```

The run uses the repository's `solverLibs` kernels, without the legacy one-second pause at each write. It stops at t = 120 lid-transit times, or earlier if max |du/dt| falls below 1e-7.

In the image, the texture is a line-integral convolution of the flow direction, and the colour is vorticity (Crameri `vik`, blue clockwise, red anticlockwise). The colour scale is compressed with tanh, so the thin boundary layer under the lid does not swamp the interior. The dark grey bar marks the moving lid.
