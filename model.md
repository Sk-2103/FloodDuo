# FloodDualSeg — Current Model Architecture

Status: 2026-06-25. Code: `src/models/`. Full result tables in results.md.
- **arch5** (DINOv3+Clay, Earth-Adapter in-block, ADAC+PPAd post-hoc, NO stem,
  LinkNet decoder): FP test IoU **0.8100**; cross-region LORO FP 0.701 / UFO 0.813.
- **arch6 (current leading variant)** = arch5 + **disagreement-aware fusion**
  (per-branch aux heads → disagreement map D → D-gated per-pixel fusion,
  replacing arch5's scalar-gate concat). Cross-region LORO **FP 0.728
  (+2.7 pp over arch5, wins 15/19 regions)**; UFO running. Config
  `configs/arch6_fp.yaml` (`model.fusion_opts`). Ablation ladder v0–v3 in
  `configs/disag_v*_fp.yaml` (v0 = arch5, bit-for-bit). See §5b.

## CHAMPION CONFIG (arch5, configs/arch5_ea_linknet_fp.yaml)
- In-block: **Earth-Adapter** (`earth_adapter.py`, arXiv:2504.06220) — per
  block, parallel to FFN in BOTH backbones: spatial + DFT low-freq + DFT
  high-freq low-rank experts (bottleneck 32, rho 0.25), softmax router,
  zero-init scale; patch tokens only. Requires `grad_ckpt: true` (CkptBlock
  wrappers; fp32 FFT activations OOM otherwise). Beat LoRA-8 (+0.5 pp).
- Post-hoc per tap: **ADAC + PPAdapter** (pyramid-pooled context). Tested
  and rejected as ADAC replacements: StripAdapter (0.8027), WaveletAdapter
  (0.7939), HFDA (0.7890 w/ LoRA). ADAC remains best local adapter.
- Decoder: **LinkNetDecoder** (`decoders_v2.py`) — Reassemble pyramid +
  additive-skip sequential decoding. Ranking (2× replicated):
  LinkNet > UPerNet > DPT. No PPA stem (stem-free ≥ stem).
- 64.3M trainable / 679M total; ~3 s/step bs8 512² (grad ckpt), 15.6 GB.

## Overview

Dual frozen foundation-model encoders + trainable adapters/fusion/decoder.
Input: raw PlanetScope reflectance `(B, 4, H, W)` (B,G,R,NIR @ 3 m, values
~[0, 0.5]). Output: binary water logits `(B, 1, H, W)`.

Diagram = arch6 path (stem-free, LinkNet, disagreement-gated fusion). arch5 =
identical minus the aux-heads/D and using a scalar-gate MultiDepthFusion (§5).

```
                       (B, 4, 512, 512) raw reflectance
                          │
            ┌─────────────┴──────────────┐
            │ RGB [R,G,B]                 │ 4ch (B,G,R,NIR)
            ▼                             ▼
   ┌──────────────────┐         ┌──────────────────┐
   │ DINOv3-L  FROZEN │         │ Clay v1.5-L FROZEN│
   │ patch 16         │         │ patch 8, GSD-aware│
   │ +Earth-Adapter   │         │ +Earth-Adapter    │   (EA in-block, trainable)
   └────────┬─────────┘         └────────┬─────────┘
            │ taps 5,11,17,23            │ taps 5,11,17,23
            ▼                            ▼
     4×(B,1024,32,32)             4×(B,1024,64,64)
            │ [ADAC+PPAd ×4]             │ [ADAC+PPAd ×4]   ← trainable post-hoc
            │                            │
            │ deepest tap                │ deepest tap
            ▼                            ▼
      ┌───────────┐                ┌───────────┐
      │ aux head  │                │ aux head  │   (BranchAuxHead, deep-sup)
      └─────┬─────┘                └─────┬─────┘
       aux_dino (B,1,H,W)           aux_clay (B,1,H,W)
            └──────────┬─────────────────┘
                       ▼  fp32, @32², detach
              D = JSD(p_dino, p_clay)   (label-free disagreement / OOD map)
                       │
            ┌──────────┴───────────────────────────┐
            │   GatedMultiDepthFusion (per depth)   │ ◄── feats_a, feats_b
            │   proj a,b→256 ; gate(cat[a,b,D])     │
            │   →softmax→ per-pixel w_dino,w_clay ;  │
            │   fused = w_dino·a + w_clay·b → 3×3    │
            └──────────────────┬────────────────────┘
                       4×(B, 256, 64, 64)
                               │
                       LinkNetDecoder  (Reassemble pyramid + additive skips)
                               ▼
                       (B, 1, 512, 512) water logits

  Losses: main BCE+Dice+boundary on logits
        + 0.3·(BCE+Dice) on aux_dino and aux_clay  (independent, no coupling)
```

## Components

### 1. DINOv3 branch (`encoders.py: DinoV3Encoder`) — semantic/spatial
- `timm vit_large_patch16_dinov3.lvd1689m`, `dynamic_img_size=True`, FROZEN
  (runs under `no_grad`).
- Input prep (inside module): RGB = bands [R,G,B]; norm per `norm_mode`:
  - `dataset` (FP rule): TCI scaling `clip(refl/0.3, 0, 1)` → ImageNet mean/std.
  - `per_image` (UFO rule): per-band per-image z-score (no TCI/ImageNet).
- Features via `forward_intermediates`, taps at blocks **(5, 11, 17, 23)** of
  24 → 4 maps `(B, 1024, H/16, W/16)`.

### 2. Spectral branch — two interchangeable encoders (`spectral:` config)
**DOFA-B** (`encoders.py: DofaEncoder`) — vendored `models_dwv.py`, ckpt
`pretrained/DOFA_ViT_base_e100.pth`, FROZEN.
- Wavelength-conditioned patch embed; PS wavelengths µm [0.490, 0.565, 0.665,
  0.865]. Patch 16. Taps **(2, 5, 8, 11)** of 12 → 4× `(B, 768, H/16, W/16)`.
- Norm: z-score with combined train stats (`dataset`) or per-image.
- Pos-embed bicubic-interpolated for non-224 inputs.
- `upsample: 2` variant (input 2× → 1/8 grid) was tried and REJECTED
  (FP 0.776 < 0.796 baseline — interpolated pixels, OOD scale).

**Clay v1.5-L** (`clay_encoder.py: ClayEncoder`) — vendored `clay/{backbone,
factory,utils}.py`, ckpt `pretrained/clay-v1.5.ckpt`,
FROZEN.
- Wavelength-conditioned DynamicEmbedding + **GSD-aware** sincos pos-enc
  (gsd=3 m), patch 8, **full-res input** (`clay_downsample: 1`) → 64×64 grid
  at 512 (finer than DINOv3). Time/latlon metadata encodings zeroed.
- MAE masking bypassed: patch-embed → pos-enc → transformer layers iterated
  manually; taps **(5, 11, 17, 23)** of 24 → 4× `(B, 1024, H/8, W/8)`.

### 3. ADAC adapters (`adapters.py`) — trainable, applied POST-HOC
- On each tapped map: parallel depth-wise 3×3 convs, dilations (1,2,3),
  summed → GroupNorm → GELU → 1×1 conv → zero-init gamma gate → residual.
- Post-hoc (outside the backbone) so backbones stay fully frozen/no_grad.
  In-block adapters = untested ablation.

### 4. PPA DetailStem (`ppa.py`) — trainable, full-res local detail
- Conv stride-2 → PPA → conv stride-2 → PPA. Skips: f2 `(64, H/2)`,
  f4 `(128, H/4)`.
- PPA (adapted from HCF-Net): skip 1×1 + [serial 2×conv3×3 ‖ local patch-attn
  (16² grid gate) ‖ global patch-attn (4² grid gate)] → channel attn (avg+max
  MLP) → spatial attn (7×7) → residual + GN + GELU.

### 5. MultiDepthFusion (`fusion.py`) — trainable [arch5 / ladder v0]
- Per depth: 1×1 proj both branches → 256 + GN; learnable scalar gate per
  branch; if grids differ, coarser upsampled (bilinear) to finer; concat →
  1×1 conv → 3×3 conv (GN+GELU).
- Limitation: the gate is **one scalar per branch per depth** — same trust
  ratio at every pixel of every image. arch6 (§5b) replaces this.

### 5b. Disagreement-aware fusion (`fusion.py`, `heads.py`) — arch6 [ladder v2]
Gated by `model.fusion_opts` (default all-false = arch5; v0/v1 unaffected).
Dependency chain: `gated_fusion → disagreement → aux_heads`.

- **Per-branch aux heads** (`heads.BranchAuxHead`, `aux_heads:true`): a tiny
  head (1×1→GN→GELU→1×1→1 logit) on each branch's deepest adapted tap predicts
  water INDEPENDENTLY (`aux_dino` from DINOv3, `aux_clay` from Clay), upsampled
  to full res. Deep-supervised (BCE+Dice vs GT, weight `aux_loss_weight` 0.3),
  with NO coupling between the two heads — so their divergence stays an emergent
  signal, measured in PREDICTION space (not unaligned feature space) and
  PRE-fusion.
- **Disagreement map D** (`heads.disagreement_map`, `disagreement:true`):
  D = Bernoulli **Jensen–Shannon divergence** between the two branches' water
  probs at the coarse 32² grid; bounded [0, ln2]. Also exposes entropy H of the
  mean. **Computed in fp32** (the JSD log/division underflow in bf16 — `1-1e-6`
  rounds to 1.0 → NaN) and **detached** (a label-free OOD/uncertainty signal
  into the gate, NOT a gradient path → keeps heads independent + stable).
- **D-gated fusion** (`GatedMultiDepthFusion`, `gated_fusion:true`): per depth,
  gate = conv(`concat[proj_dino, proj_clay, D]`) → softmax → per-pixel weights
  `w_dino, w_clay`; `fused = w_dino·proj_dino + w_clay·proj_clay` → 3×3 conv
  (GN+GELU). `spectral_bias` (0.5) warm-starts trust toward Clay where D is high
  (learnable). Gate runs fp32, final conv zero-init (clean softmax prior at start).
  D tells the gate THAT the experts disagree, not WHO is right — policy is learned.
- **Cross-branch co-attention** (`CrossAttnStack`, `cross_attn:true`, ladder v3,
  default OFF): staged co-attention at 32² (Clay pooled to 32² for attention
  only; native 64² kept via residual), 4 heads, zero-init scale (no-op at init),
  +34M params. Riskiest module.
- D is returned from the model; `src.eval` dumps D maps + per-tile mean_D
  (`dmaps_<split>/`); `scripts/disagreement_diag.py {pixel,region}` checks D
  calibration (error-vs-D, ECE) and region OOD (mean-D vs IoU-drop).

### 6. FineDecoder (`decoder.py`) — trainable
- Residual merge of the 4 depth maps (deepest first, conv each).
- Stride-agnostic: target sizes derived from stem skips. Stages:
  →1/8 (192) → 1/4 (128) ⊕ f4 → 1/2 (64) ⊕ f2 → head conv(32)→1ch @ full res.

## Loss / training (src/losses.py, src/train.py)
- BCE + Dice + boundary-band BCE (band = morphological gradient of GT, 3 px,
  via max-pool); weights 1/1/1.
- arch6 adds **deep-supervision aux loss** = (BCE+Dice) on `aux_dino` and
  `aux_clay` independently, ×`aux_loss_weight` (0.3); logged as `aux`/`meanD`.
  Train loops (train.py, loro_fold.py) skip any non-finite-loss batch (guard).
- AdamW lr 3e-4 (adapters/stem/fusion/decoder only), wd 0.01, cosine+warmup,
  bf16 autocast, grad-clip 1.0, bs 8, 512² random crops + flips/rot90,
  60 epochs. Full-tile 1024² eval.
- Trainable params: 15.67M (DOFA variant) / 17.80M (Clay variant); totals
  430M / 632M.
- Norm rule (user-decided): FP = `dataset`, UFO = `per_image`.

## Known weaknesses / architecture work candidates
1. **Cross-region recall collapse** (LORO: US-Kansas R 0.52, Spain, Bangladesh;
   UFO SLC/DKA precision≫recall): unseen water appearance gets missed.
   Candidates: stronger spectral augmentation (band jitter, gain/offset),
   region-style randomization, NDWI as explicit input/aux supervision.
2. **Checkpoint selection noise** on small val sets (Clay-UFO: best val
   0.893 → test 0.880): consider EMA weights or val+swa.
3. **Adapters are post-hoc only** — backbone never sees adapted features;
   in-block ADAC (cheap LoRA-style) untested.
4. **Fusion is per-depth concat** — no cross-branch attention; tokens never
   interact across branches. → ADDRESSED by arch6 (§5b): D-gated per-pixel
   fusion; optional cross-attention (ladder v3, default off).
5. **Spectral branch disagreement**: Clay wins FP, DOFA wins UFO — a gated
   mixture or distill-both setup could capture both. → arch6 turns branch
   disagreement into a per-pixel routing signal (D); aux heads make it explicit.
6. **No TTA / ensembling** in eval yet (cheap +).
7. PPA stem is shallow (2 stages); water edges at 3 m may benefit from a
   1/1-resolution stage or frequency-domain branch (SFFNet-style).
