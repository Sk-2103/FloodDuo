#!/bin/bash
# Quick check: regenerates 3 of the 16 figures to confirm a setup works.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/3] Figure 4 — LORO IoU distribution across models"
python scripts/fig_boxplot.py

echo "[2/3] Figure 15 — accuracy vs. training compute"
python scripts/fig_efficiency.py

echo "[3/3] Figure 7-style gallery — FloodDuo predictions vs. U-Net/DOFA/TerraMind"
python scripts/fig_make.py --dataset floodplanet --regions Ghana Nigeria US-Oklahoma --mode overall

echo "Done — see outputs/manuscript_figs/ (Fig_boxplot_LORO_IoU, Fig_efficiency, Gallery_floodplanet_overall_*)."
