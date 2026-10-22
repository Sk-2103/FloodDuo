# Results

All numbers are **leave-one-region-out (LORO)**: for each of the 19 FloodPlanet
regions and 14 UFO flood-event sites, a model is trained from scratch on every
*other* region and evaluated on the held-out one, with no validation-based
checkpoint selection. Region-mean IoU is the unweighted mean of per-region IoU
(not micro-averaged over pixels), so every region counts equally regardless of
its tile count. See the paper (Section 2.3) for the full protocol and
`scripts/show_loro.py` to regenerate every table below from `runs/`.

## Headline comparison

| Model | FloodPlanet IoU | UFO IoU |
|---|---|---|
| **FloodDuo (ours)** | **0.728** | **0.819** |
| TerraMind-L (frozen, UPerNet) | 0.700 | 0.789 |
| Prithvi-EO-2 (frozen, UPerNet) | 0.678 | 0.774 |
| DOFA-L (frozen, UPerNet) | 0.687 | 0.766 |
| Clay v1.5-L (frozen, UPerNet) | 0.675 | 0.773 |
| CROMA-L (frozen, UPerNet) | 0.672 | 0.715 |
| SSL4EO-S12 DINO (ViT-S, frozen, UPerNet) | 0.650 | 0.767 |
| U-Net (from scratch) | 0.648 | 0.782 |

FloodDuo beats U-Net by +8.0 / +3.7 percentage points (pp) and the strongest
single-encoder foundation model (TerraMind) by +2.8 / +3.0 pp on FloodPlanet /
UFO (paired Wilcoxon, significant on both).

## Matched single-encoder controls

DINOv3-L-only and Clay-v1.5-only, each trained under FloodDuo's *exact*
adaptation (Earth-Adapter + ADAC + PPAdapter) and LinkNet decoder, so any gap
reflects the encoder rather than added capacity or a different decoder:

| Model | FloodPlanet IoU | UFO IoU |
|---|---|---|
| DINOv3-L only (spatial) | 0.668 | 0.751 |
| Clay v1.5 only (spectral) | 0.702 | 0.799 |
| **FloodDuo (dual)** | **0.728** | **0.819** |

Dropping DINOv3-L costs −4.0 pp / −2.0 pp; dropping Clay costs more on both
benchmarks. FloodDuo exceeds the DINOv3-only control significantly on both
benchmarks (p < 0.001) and the Clay-only control at p = 0.07 / 0.09.

## Disagreement-gated fusion ablation

| Fusion | FloodPlanet IoU | UFO IoU |
|---|---|---|
| Scalar-gate concat | 0.701 | 0.813 |
| **Disagreement-gated (ours)** | **0.728** | **0.819** |

Conditioning the per-pixel fusion gate on the branch-disagreement map D (a
Jensen-Shannon divergence between the two branches' auxiliary predictions,
Figure 13) adds +2.7 / +0.6 pp over a plain learned scalar gate per branch.

## Random split vs. leave-one-region-out

The identical FloodDuo model (same seed, epoch budget, learning rate,
normalization — only the train/test split protocol differs), evaluated with
the same region-mean IoU:

| Protocol | FloodPlanet IoU | UFO IoU |
|---|---|---|
| Random split (regions shared between train/test) | 0.796 | 0.875 |
| **Leave-one-region-out (ours)** | **0.728** | **0.819** |
| Overstatement | +6.8 pp | +5.6 pp |

A conventional random split overstates cross-region transferability by 5-7 pp
on the same model — the motivation for reporting LORO throughout.

## Boundary-resolved error analysis (Figure 9)

Boundary-F1 at four tolerances (meters), 95% region-bootstrap CI omitted here
for brevity (see the paper / regenerate with `boundary_analysis.py`):

| Dataset | Model | τ=3m | τ=6m | τ=9m | τ=12m |
|---|---|---|---|---|---|
| FloodPlanet | FloodDuo | 0.345 | 0.448 | 0.517 | 0.552 |
| FloodPlanet | TerraMind | 0.303 | 0.408 | 0.488 | 0.534 |
| FloodPlanet | U-Net | 0.293 | 0.378 | 0.434 | 0.466 |
| UFO | FloodDuo | 0.499 | 0.642 | 0.723 | 0.761 |
| UFO | TerraMind | 0.424 | 0.567 | 0.663 | 0.705 |
| UFO | U-Net | 0.478 | 0.598 | 0.663 | 0.694 |

Narrow-feature (≤2 px wide) recall: FloodDuo 0.559 / U-Net 0.526 / TerraMind
0.526 on FloodPlanet; FloodDuo 0.537 / U-Net 0.570 / TerraMind 0.497 on UFO.

## Disagreement map D as a reliability signal (Figure 13)

- Error association: cell error rate rises monotonically with D (Spearman
  ρ = 0.52 on FloodPlanet, 0.59 on UFO, across equal-count deciles).
- Region-level: per-region mean D correlates negatively with held-out IoU
  (Pearson r = −0.41 FloodPlanet, −0.66 UFO).
- A monotonic D→error mapping fit leave-one-region-out (never using the
  scored region) is well calibrated: ECE = 0.007 FloodPlanet, 0.001 UFO.
- Blind spot: ~2/3 of total error mass falls where both branches agree on the
  wrong class, which D cannot surface by construction (Figure 16).

## Efficiency (Figure 15)

Training compute at 512×512, forward + backward, frozen backbones:
FloodDuo ≈ 13.9 TFLOPs/step, about 12× U-Net's. ~65M trainable parameters
against ~614M frozen (~679M total).

## Reproducing these tables

```bash
python scripts/show_loro.py --budget 80          # headline + comparative table
python scripts/boundary_analysis.py              # Figure 9 statistics
python scripts/disagreement_full_analysis.py     # D calibration (Figure 13)
python scripts/failure_disagreement.py           # D blind-spot audit + Figure 16 maps
```

All read from `runs/<experiment>/<dataset>/<region>/result.json`, produced by
the LORO training runs described in the main [README](README.md).
