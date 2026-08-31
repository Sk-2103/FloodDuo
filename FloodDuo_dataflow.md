# FloodDuo (arch6) — Data Flow (for the schematic)

A step-by-step, plain-language description of how data moves through FloodDuo.
Use this as the reference when drawing the architecture figure.

---

## 0. One-sentence summary
Two **frozen** foundation-model encoders (one spatial, one spectral) each see the
PlanetScope tile, are lightly **adapted**, and are **fused per-pixel** using a
**disagreement map** (how much the two encoders disagree) before a decoder
produces the flood mask.

---

## 1. Input
- One PlanetScope tile: **4 bands** (Blue, Green, Red, NIR), e.g. 512×512 in
  training, 1024×1024 at test.
- It is sent to **both** encoders:
  - **DINOv3-L** gets the **RGB** channels (spatial branch).
  - **Clay v1.5-L** gets **all 4 bands** (spectral branch).

## 2. The two encoders (both FROZEN)
- Each is a ViT (DINOv3 patch-16, Clay patch-8). Their pretrained weights are
  **never updated**.
- **Earth-Adapter (EA)** is the only thing trainable *inside* the encoder: it
  sits **in parallel with the FFN of every transformer block** (so EA acts at
  **every layer**, all 24 blocks). This is the *in-block* adapter.

## 3. Tap 4 stages per encoder
- From each encoder we read out (“tap”) the feature map at **4 depths: blocks
  6, 12, 18, 24**.
- So we get **4 maps from DINOv3** and **4 maps from Clay** = 8 feature maps.

## 4. ADAC + PPA — post-hoc, on ALL 4 stages
- Each of the 4 tapped maps (per branch) is refined by two **post-hoc** adapters
  applied *outside* the transformer:
  - **ADAC** (atrous depth-wise convs) → re-adds local/boundary detail.
  - **PPA** (pyramid pooling) → re-adds multi-scale context.
- Result: **4 adapted maps per branch** (still 8 total). These adapted maps are
  what everything downstream uses.
- ⚠️ **EA ≠ ADAC/PPA in placement:** EA is *inside* every block; ADAC+PPA are
  *outside*, on the 4 tapped maps.

## 5. Auxiliary heads + disagreement map D — DEEPEST stage only
- Take **only the deepest adapted stage (block 24)** of each branch.
- A tiny **aux head** on each turns it into an independent water-probability map:
  - DINOv3 stage-24 → **p_dino**
  - Clay stage-24 → **p_clay**
- **Disagreement map D = Jensen–Shannon divergence(p_dino, p_clay)** on a 32×32
  grid, bounded [0, ln 2]. High D = the two encoders disagree here.
- ⚠️ **Only stage 24 makes D.** Stages 6/12/18 do **not** feed the aux heads or D.
- D is computed once, in fp32, and **detached** (the aux heads learn only from
  their own deep-supervision loss; gradient does not flow back through D).

## 6. Disagreement-gated fusion — at EACH of the 4 depths
For every depth k ∈ {1,2,3,4}:
1. Project DINOv3 map_k and Clay map_k to 256 channels (DINOv3 32² is upsampled
   to Clay’s 64²).
2. A small **gate** network takes `concat[ DINOv3_k , Clay_k , D ]` → softmax →
   two per-pixel weights `(w_spatial, w_spectral)`.
3. `fused_k = w_spatial · DINOv3_k + w_spectral · Clay_k` → 3×3 conv.
- ⚠️ **The same D is fed into all 4 fusion blocks** (resized to each grid).
- Output: **4 fused maps** (256 ch, 64×64).

## 7. Decoder → output
- A **LinkNet decoder** takes the 4 fused maps, upsamples coarse→fine, and
  outputs the full-resolution **binary flood mask**.

## 8. Losses (training only)
- Main loss on the mask: **BCE + Dice + boundary-weighted BCE**.
- **Deep supervision**: each aux head (p_dino, p_clay) also gets its own
  (BCE+Dice) loss vs. the ground truth, weight 0.3, **independent** (no term
  couples the two heads) — this is what keeps D a meaningful, emergent signal.

---

## Arrow diagram (what connects to what)

Read it as: **every stage → its fusion block** (horizontal arrows). **Only the
stage-24 row has the extra "└─► aux head" branch.** Stages 6/12/18 have NO arrow
to the aux heads or D.

```
DINOv3 branch  (frozen; EA inside every block)
  stage 6   ─ADAC+PPA─────────────────────────────────────────►  fusion depth-1
  stage 12  ─ADAC+PPA─────────────────────────────────────────►  fusion depth-2
  stage 18  ─ADAC+PPA─────────────────────────────────────────►  fusion depth-3
  stage 24  ─ADAC+PPA──┬──────────────────────────────────────►  fusion depth-4
                       └──► aux head ─► p_dino ──┐
                                                 │
                                                 ├─►  D = JSD(p_dino, p_clay)
                                                 │
                       ┌──► aux head ─► p_clay ──┘
  stage 24  ─ADAC+PPA──┴──────────────────────────────────────►  fusion depth-4
  stage 18  ─ADAC+PPA─────────────────────────────────────────►  fusion depth-3
  stage 12  ─ADAC+PPA─────────────────────────────────────────►  fusion depth-2
  stage 6   ─ADAC+PPA─────────────────────────────────────────►  fusion depth-1
Clay branch  (frozen; EA inside every block)

      D ───►  fed as a gating input into ALL FOUR fusion blocks (depth-1 … depth-4)

      fusion depth-1 … depth-4  →  f1 … f4  →  LinkNet decoder  →  binary flood mask
```

Key reading of the arrows:
- **stages 6, 12, 18** → straight to fusion (no branch).
- **stage 24 (both branches)** → fusion **and** → aux head → p → **D**.
- **D** (one map) → gates **all four** fusion blocks.

## Count sheet (how many of each)
| Thing | Count |
|---|---|
| Encoders (frozen) | 2 (DINOv3 spatial, Clay spectral) |
| EA adapters | in **every** ViT block (both encoders) |
| Tapped stages per encoder | 4 (blocks 6, 12, 18, 24) |
| ADAC+PPA blocks | 4 per encoder (one per tap) |
| Aux heads | 2 (one per branch, **deepest stage only**) |
| Disagreement maps D | **1** (from stage 24), reused by all fusion blocks |
| Gated-fusion blocks | 4 (one per depth) |
| Decoder | 1 LinkNet |

## Tensor shapes (at 512² input)
| Signal | Shape |
|---|---|
| DINOv3 tap (each of 4) | (1024, 32, 32) |
| Clay tap (each of 4) | (1024, 64, 64) |
| aux p_dino / p_clay | (1, H, W) → compared at 32×32 |
| D | (1, 32, 32), values in [0, ln 2] |
| fused map (each of 4) | (256, 64, 64) |
| output logits | (1, 512, 512) |

---

## Extended arrow map — ONE ViT block (where EA lives) + the tap

A transformer block is **two residual sub-layers**: attention, then FFN. EA is a
trainable branch **in parallel with the FFN** — it does *not* replace anything,
it adds a small frequency-aware correction. The frozen weights (LN, attention,
FFN) are untouched; **only EA's experts + router + α are trained inside the block.**

```
 x_in  (tokens: B, N, 1024)
   │
   ├──────────────────────────────────────────────┐  (residual)
   ▼                                                │
 LayerNorm-1 ─► Multi-Head Self-Attention ─────────►(+)──► a   (B, N, 1024)
                                                            │
   ┌────────────────────────────────────────────────────── ┤  (residual)
   ▼                                                         │
 LayerNorm-2 ─► z ─┬─────────────► FFN (MLP) ──────────────┐│
                   │                                        ▼▼
                   └──► Earth-Adapter(z) ─► α·Δ ──────────►(+)──► x_out  (B, N, 1024)
                              (α zero-init: at start EA = 0, block = original)
```

Earth-Adapter internals (the `EA(z)` box above) — patch tokens only:

```
 z (patch tokens, reshaped to 32×32 or 64×64 grid)
   │
   ├─ 2D FFT → split by a circular mask (radius ρ=0.25)
   │     ├─ inside  → iFFT → LF tokens ─► low-freq expert  (1024→32→1024)
   │     └─ outside → iFFT → HF tokens ─► high-freq expert (1024→32→1024)
   ├──────────────────────────────────► spatial expert   (1024→32→1024)   (raw z)
   │
   router: mean over tokens → Linear(1024→3) → softmax → (g_spatial, g_lf, g_hf)
   │
   Δ = g_spatial·spatial + g_lf·LF + g_hf·HF
   EA(z) = α · Δ          (α scalar, initialised to 0)
```

**The tap (post-hoc, only at blocks 6/12/18/24):** the block's *output* `x_out`
is what gets read out, reshaped to a 2D map, then refined OUTSIDE the block:

```
 x_out (B, N, 1024)  ─ drop prefix tokens, reshape ─► map (1024, H, W)
        │
        ├─► ADAC  (atrous DW-conv, residual: map + γ·adac(map))
        │
        └─► PPA   (pyramid pooling, residual: ··· + γ·ppa(···))
                       │
                       ▼
                 adapted tap (1024, H, W)   → goes to fusion (Section 6)
```

⚠️ **Placement contrast:**
- **EA** = inside **every** block (all 24), parallel to the FFN, trained.
- **ADAC + PPA** = outside, applied **only to the 4 tapped blocks' outputs**.
- Blocks that are *not* tapped still run EA, but their output is not read out.

## Extended arrow map — ONE disagreement-gated fusion block (depth k)

Runs identically for k = 1…4. Inputs are the depth-k adapted maps + the shared D.

```
 DINOv3_k (1024, 32×32) ─ 1×1 conv → GN ─► a (256, 32×32) ─ bilinear ↑ to 64² ─► a (256, 64×64) ┐
 Clay_k   (1024, 64×64) ─ 1×1 conv → GN ─► b (256, 64×64) ──────────────────────────────────────┤
 D        (1,    32×32) ─ bilinear ↑ to 64² ───────────────► D (1, 64×64) ───────────────────────┤
                                                                                                  ▼
                                                      concat[ a , b , D ]  = (513, 64×64)
                                                                  │
                                              gate:  1×1 conv→256 → GELU → 1×1 conv→2   (+ spectral-bias)
                                                                  │
                                                      softmax over the 2 channels
                                                                  ▼
                                               (w_spatial , w_spectral)   (2, 64×64),  sum=1 per pixel
                                                                  │
                                          fused = w_spatial · a  +  w_spectral · b      (256, 64×64)
                                                                  │
                                                        3×3 conv → GN → GELU
                                                                  ▼
                                                            f_k   (256, 64×64)
```
→ produces **f1, f2, f3, f4** (one per depth), each (256, 64×64).

## Extended arrow map — LinkNet decoder (f1…f4 → mask)

Step 1 — **Reassemble**: turn the four same-size fused maps into a feature
**pyramid** (shallowest fused map → finest/largest level; deepest → coarsest):

```
 f1 (256,64²) ─ ConvTranspose 2× ───────────────► p0 (96, 128×128)   [finest]
 f2 (256,64²) ─ 1×1 conv → GN ───────────────────► p1 (192, 64×64)
 f3 (256,64²) ─ conv stride-2 ───────────────────► p2 (384, 32×32)
 f4 (256,64²) ─ conv stride-2 ×2 ────────────────► p3 (768, 16×16)    [coarsest]
```

Step 2 — **decode coarse → fine with ADDITIVE skips** (each LinkNetBlock =
1×1 reduce → transposed-conv 2× upsample → 1×1 expand, GN+GELU):

```
 p3 (768,16²) ─► dec3 (768→384, ↑2×) ─► x (384, 32×32)
                                          x + p2 ─► dec2 (384→192, ↑2×) ─► x (192, 64×64)
                                                       x + p1 ─► dec1 (192→96, ↑2×) ─► x (96, 128×128)
                                                                    x + p0 ─► final (96→48→32→1, ↑2×) ─► (1, 256×256)
                                                                                  │
                                                                    bilinear ↑ to input size ─► logits (1, 512×512)
                                                                                  │
                                                                    sigmoid ≥ 0.5 ─► binary flood mask
```
("+" = add the same-resolution pyramid level as a skip; each `pX` is bilinearly
resized to match `x` before adding.)

Shapes recap (512² input): fused f1…f4 = (256,64²) → pyramid p0..p3 =
(96,128²)/(192,64²)/(384,32²)/(768,16²) → decoder → logits (1,512²).

## The 3 things people draw wrong (double-check your figure)
1. **EA is inside the ViT blocks; ADAC+PPA are outside** (on the 4 tapped maps).
2. **D comes ONLY from stage 24** (deepest), via the two aux heads — stages
   6/12/18 go to fusion only.
3. **D fans out to all 4 fusion blocks** (computed once, used four times).
