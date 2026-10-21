#!/bin/sh
# Make the website figure in outputs/render/ (PNG, WebP and PDF).
# Needs Python with the packages in the README, ParaView's pvpython and LaTeX.
# Set PVPYTHON if pvpython is not on your PATH, e.g.
#   PVPYTHON=/Applications/ParaView-6.1.0.app/Contents/bin/pvpython sh scripts/render/make_figure.sh
set -e
cd "$(dirname "$0")/../.."
PV=${PVPYTHON:-pvpython}
R=scripts/render
O=outputs/render
mkdir -p $O
python $R/run_cavity_re1000.py        # about 4 minutes
python $R/check_ghia.py               # compare with Ghia et al. (1982)
python $R/render_cavity.py
python $R/compose_cavity.py $O/cavity-flow-raw.png $O/cavity-flow
