# Results — High-Resolution Flood Segmentation

Detailed numbers for all experiments. CLAUDE.md keeps only major findings +
how-to-run context; this file is the full record. Per-tile/per-fold raw JSON
lives in `runs/<name>/...` on the SSD.

Protocol reminders: strict per-dataset independence; norm = FP dataset-stats /
UFO per-image; threshold 0.5; full-tile 1024² eval; LORO = pool all splits,
hold out one region, train rest, NO val selection (leak-free), eval last model.

---

## Random-split test (full-tile 1024², best-by-own-val)

| Dataset | Model | val IoU | test IoU | test F1 | test P | test R |
|---|---|---|---|---|---|---|
| FP | UNet | 0.627 | 0.724 | 0.840 | 0.831 | 0.849 |
| FP | DINOv3+DOFA | 0.683 | 0.796 | 0.886 | 0.886 | 0.887 |
| FP | DINOv3+Clay | 0.675 | 0.802 | 0.890 | 0.886 | 0.895 |
| FP | **arch5** | 0.698 | **0.810** | — | — | — |
| UFO | UNet | 0.857 | 0.865 | 0.927 | 0.925 | 0.930 |
| UFO | DINOv3+Clay | 0.893 | 0.880 | 0.936 | 0.960 | 0.913 |
| UFO | **DINOv3+DOFA / arch5** | 0.887 | **0.886** | 0.939 | 0.949 | 0.930 |

Both FM variants beat UNet everywhere (FP +7–8 pp, UFO +1.5–2 pp).
FP test (0.80) ≫ FP val (0.68) — FP val split is harder; don't tune on test.
UFO val≈test (~0.88).

### Normalization experiment (`runs/normalize_results/`)
Per-image vs fixed dataset stats. UFO benefits from per-image (UNet +6.3 pp,
DualSeg +0.9 pp test IoU — radiometrically varied urban scenes); FP: DualSeg
unchanged, UNet hurt (−2.1 pp). → standing norm rule (FP dataset / UFO per-image).
DualSeg robust to the choice; UNet sensitive.

### LR sweep (`runs/lr_sweep/`, arch5, 30-ep, val IoU)
FP: 1e-4 .671 / 2e-4 .670 / 3e-4 .670 / **5e-4 .678** / 1e-3 .673.
UFO: flat (.872–.874, spread 0.17 pp = noise). → lr=5e-4 both datasets.

---

## Ablations (FP random-split test unless noted)

- **Dual encoder confirmed:** Clay-only (no DINOv3) FP 0.762 (−4.0 pp),
  UFO 0.860 (−2.0 pp). DINOv3 branch earns its place on both.

### Architecture search (FP test; baseline stem model 0.8023)
| exp | in-block | post-hoc | decoder | val | test IoU |
|---|---|---|---|---|---|
| baseline | — | ADAC | Fine+stem | 0.683 | 0.802 |
| arch1 | LoRA-8 | ADAC+PPAd | UPerNet | 0.691 | 0.804 |
| arch2 | LoRA-8 | ADAC+PPAd | LinkNet | 0.690 | 0.807 |
| arch3 | LoRA-8 | +HFDA | UPerNet | 0.697 | 0.789 |
| arch4 | Earth-Adapter | ADAC+PPAd | UPerNet | 0.706 | 0.809 |
| **arch5** | **Earth-Adapter** | ADAC+PPAd | **LinkNet** | 0.698 | **0.810** |
| arch6 | Earth-Adapter | ADAC+PPAd | DPT | 0.683 | 0.797 |
| arch7 | Earth-Adapter | STRIP+PPAd | UPerNet | 0.681 | 0.803 |
| arch8 | Earth-Adapter | WAVELET+PPAd | UPerNet | 0.676 | 0.794 |

- CHAMPION arch5 = EA + ADAC+PPAd + LinkNet. Decoder: LinkNet > UPerNet > DPT.
  Local-adapter: ADAC > Strip > Wavelet. Earth-Adapter > LoRA.
- **Rejected:** fine-grid DOFA (`dofa_upsample:2`, FP 0.776 — Clay's win is
  native patch-8 pretraining, not grid geometry); HFDA (best-val-worst-test,
  selection noise); M2F decoder (degenerate for binary); NDWI-guided adapter.

---

## Leave-one-region-out (LORO)

`src/loro_fold.py`, `scripts/run_loro*.sh`. Region from filename token 2
(FP 19 regions, UFO 14). Results in `runs/loro*/.../result.json`.

### 40-ep, lr 3e-4 (PRIMARY for paper)
| Dataset | arch5 | UNet | Δ |
|---|---|---|---|
| FP region-mean | **0.713** | 0.637 | +7.6 pp |
| UFO region-mean | **0.807** | 0.773 | +3.4 pp |

arch5 swiraug variant (FP cross-region champion): FP mean 0.722 (+0.9 pp over
arch5 baseline, median +4.2 pp). aug-only inconsistent; UFO aug neutral.

### 80-ep, lr 5e-4 (robustness + comparative-set budget)
| Dataset | arch5 | DOFA-L dual | DOFA-L single | UNet |
|---|---|---|---|---|
| FP mean | **0.701** | 0.696 | 0.691 | 0.648 |
| UFO mean | **0.813** | 0.809 | 0.772 | 0.782 |

- DOFA-L ≈ arch5 (within 0.5 pp) — spectral FM choice negligible at matched cap.
- Single-DOFA-L (no DINOv2): FP −0.5 pp (dispensable), UFO −3.7 pp (drops below
  UNet — DINOv2 needed for UFO's spatial detail).

### Comparative single-FM set (`runs/loro_comparative/<v>/`, 80 ep)
Single frozen encoder + UPerNet, no adapters. Region-mean IoU:

| variant | FP mean | UFO mean |
|---|---|---|
| clay_noada | 0.675 | 0.773 |
| dofa_noada | 0.687 | 0.766 |
| prithvi | 0.678 | 0.774 |
| terramind | **0.700** | 0.789 |
| croma | 0.672 | 0.715 |
| ssl4eo_dino* | 0.650 | 0.767 |

(*ssl4eo_dino is ViT-**S**/16 — the only checkpoint available — NOT capacity-
matched to the ViT-L FMs; expect it low partly from size. Uses NIR natively
via the S2 B8 slot.)

Full per-region tables (arch5 + all frozen no-adapter FMs + UNet) are generated
on demand — they are NOT duplicated here to avoid drift:
`python scripts/show_loro.py --budget 80`.

CROMA caveat: 2D-ALiBi attention tied to a fixed patch grid (pretrained 120px),
so it can't run our 512/1024 patch-8 tiles — input resized to fixed 256px
(32² tokens) grid (`src/models/croma_encoder.py`, `croma_size`; ALiBi
vectorised). Coarse grid → weakest spectral FM; FP +2.4 pp over UNet but UFO
−6.7 pp BELOW UNet (NSW collapses 0.586 vs ~0.88).

### LORO conclusions
- FM dual-encoder advantage over UNet robust across budgets (+5–8 pp FP, +3 pp UFO).
- Spectral FM choice near-negligible cross-region *when the FM ingests the
  resolution* — CROMA is the exception (capped at 256px → fails UFO).
- DINOv2 dispensable on FP, needed on UFO (spatial fine detail).
- ssl4eo_dino (ViT-S) is the weakest FM (FP 0.650 ≈ UNet; UFO 0.767 < UNet) —
  capacity matters: a small ViT-S FM ≈ scratch UNet, the ViT-L FMs clearly beat it.
- LORO ≈ 7–9 pp below random-split → regional domain shift is the main gap;
  failure mode = missed water (low recall) in unseen regions.
- Failure regions: FP US-Kansas/Spain/Bangladesh/Bolivia (largest, 27%);
  UFO worst are tiny regions (SLC/BNA/BEI, 2–6 tiles, noisy).

---

## Data-side experiments

### Spectral aug + SWIR distillation
- `spectral_augment` (train p=0.8): per-band gain ±20%, offset ±0.02, gamma
  e^±0.25 — simulates cross-region radiometry. `train.spectral_aug`.
- SWIR distillation: PS lacks SWIR (no new info from VNIR — TerraMind generation
  prototype abandoned). Cross-sensor distill from paired `floodPlanet_Sen2`:
  aux head regresses MNDWI=(B3−B11)/(B3+B11) from S2, L1 masked by pair
  availability, head discarded at inference (model stays 4-band).
  `model.swir_aux_head`, `train.swir_aux`/`w_swir`. FP-only.
- Result: swiraug = FP cross-region champion (see 40-ep LORO above).

### External zero-shot: Rio Grande Valley (39 tiles)
`scripts/eval_external.py`. Severe collapse 0.80 in-domain → ~0.40. Per-image
labels inundated agricultural fields — appearance gap absent from training.
Reinforces the data-side direction over more architecture work.
