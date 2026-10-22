# Figures

Only the architecture diagram (`Figure_01_overall_architecture_of_floodduo.png`)
is bundled here — it's the one embedded in the main [README](../README.md).

The other 15 figures from the paper are not shipped as image files (no point
carrying ~20 MB of PNGs nobody will open outside the paper itself), but every
one of them is fully reproducible from `runs/` + `outputs/manuscript_figs/`
with the matching script:

Rows marked ★ are the 3 "major figures" `scripts/make_key_figures.sh` runs as
a quick end-to-end check — not all 16, just enough to confirm a setup works.

| # | Figure | Script |
|---|---|---|
| 2-3 | Earth-Adapter / post-hoc adapter schematics | hand-drawn |
| 4 ★ | LORO IoU distribution, 8 models | `fig_boxplot.py` |
| 5 | Per-region precision-recall | `fig_pr_scatter.py` |
| 6 | Global map of cross-region performance | `fig_geo_map.py` |
| 7 ★ | FloodPlanet prediction gallery | `fig_make.py` |
| 8 | UFO prediction gallery | `fig_make.py` |
| 9 | Boundary-resolved error analysis | `boundary_analysis.py` → `fig_boundary.py` |
| 10 | Fine-scale detail examples | `fig_make.py` |
| 11 | Encoder representation analysis | `representation_stats.py` + `representation_rerun.py` → `fig_representation.py` |
| 12 | Encoder complementarity | `fig_complementarity_render.py` |
| 13 | Disagreement map D as a reliability signal | `disagreement_full_analysis.py` → `fig_disagreement.py` |
| 14 | Learned per-pixel fusion routing | `fig_gate.py` |
| 15 ★ | Accuracy vs. training compute | `fig_efficiency.py` |
| 16 | Failure modes + disagreement maps | `failure_disagreement.py` → `fig_failure.py` |

See the main README's [Reproducing figures](../README.md#reproducing-figures)
section for the full command sequence, and the paper for every caption.
