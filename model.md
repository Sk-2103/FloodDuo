# Architecture

Two **frozen** vision foundation-model encoders, each adapted with lightweight
trainable modules, merged by a disagreement-gated fusion, decoded to a
full-resolution water mask. See Figures 1-3 for the diagrams this document
describes, and `src/models/flood_model.py` (`FloodDualSeg`) for the code.

```
                 (B, 4, 512, 512) raw PlanetScope reflectance
                          |
            +-------------+--------------+
            | RGB [R,G,B]                | all 4 bands (B,G,R,NIR)
            v                             v
   +------------------+         +------------------+
   | DINOv3-L  FROZEN |         | Clay v1.5-L FROZEN|
   | patch 16         |         | patch 8, GSD-aware|
   | +Earth-Adapter   |         | +Earth-Adapter    |  (in-block, trainable)
   +--------+---------+         +--------+---------+
            | taps 5,11,17,23            | taps 5,11,17,23
            v                            v
     4x(B,1024,32,32)             4x(B,1024,64,64)
            | ADAC + PPAdapter x4         | ADAC + PPAdapter x4   (post-hoc, trainable)
            |                            |
            | deepest tap                | deepest tap
            v                            v
      +-----------+                +-----------+
      | aux head  |                | aux head  |   (independent, deep-supervised)
      +-----+-----+                +-----+-----+
       aux_dino (B,1,H,W)           aux_clay (B,1,H,W)
            +-----------+------------------+
                        v  fp32, 32x32, detached
               D = JSD(p_dino, p_clay)   (label-free branch-disagreement map)
                        |
            +-----------+----------------------------------------+
            |   Disagreement-gated fusion, per tapped depth       |
            |   project both branches to 256ch; gate = softmax(   |
            |   conv(concat[proj_dino, proj_clay, D])) -> per-pixel|
            |   w_dino, w_clay; fused = w_dino.a + w_clay.b -> 3x3 |
            +-----------+----------------------------------------+
                 4x(B, 256, 64, 64)
                        |
                 LinkNet decoder (reassemble pyramid + additive skips)
                        v
                 (B, 1, 512, 512) water logits

  Loss: BCE + Dice + boundary-band BCE on the fused output
      + 0.3 x (BCE + Dice) on aux_dino and aux_clay independently
```

## Components

### Spatial branch — DINOv3-L (`src/models/encoders.py: DinoV3Encoder`)
`timm` `vit_large_patch16_dinov3.lvd1689m`, frozen. Input is the R, G, B
bands, normalized per the dataset's `norm_mode` (reflectance clipped to
[0, 0.3] and rescaled to ImageNet's range for the `dataset` rule, or a
per-image z-score for the `per_image` rule). Features are taken via
`forward_intermediates` at blocks (5, 11, 17, 23) of 24, giving four
`(B, 1024, H/16, W/16)` maps.

### Spectral branch — Clay v1.5-L (`src/models/clay_encoder.py: ClayEncoder`)
Wavelength-conditioned patch embedding (PlanetScope band centers 0.490,
0.565, 0.665, 0.865 µm) with GSD-aware positional encoding (3 m), frozen,
run at full resolution (patch 8, no downsampling) so its 64×64 grid at
input 512 is finer than DINOv3's 32×32. Taps at the same relative depths
(5, 11, 17, 23 of 24).

Both branches can be swapped for DOFA, Prithvi-EO-2, TerraMind, CROMA or
SSL4EO-S12 DINO (`spectral:` config key) as single-encoder, frozen,
no-adapter comparative baselines (`configs/{dofa_noada,prithvi,terramind,
croma,ssl4eo_dino}_{fp,ufo}.yaml`) — these use a plain UPerNet decoder
instead of the adapters/fusion/LinkNet pipeline below.

### Earth-Adapter — in-block, trainable (`src/models/earth_adapter.py`)
Added in parallel to the FFN of every frozen ViT block in both backbones.
Splits patch tokens with a 2D FFT into low- and high-frequency bands,
processes the unfiltered/low/high tokens with three bottleneck experts
(rank 32, bottleneck ratio 0.25), and combines them with a softmax router
and a zero-initialized scale: `FFN(x) + alpha * Delta`. Outperformed an
equivalent-budget LoRA adapter in the architecture search.

### ADAC + PPAdapter — post-hoc, trainable (`src/models/adapters.py`, `ppa.py`)
Applied to each of the four tapped feature maps, outside the frozen
backbone:
- **ADAC** (Atrous Depth-Wise Convolution Adapter): three parallel
  depth-wise convolutions (dilations 1, 2, 3), fused by a 1×1 conv, added
  as a zero-initialized gated residual — re-injects local detail the
  frozen backbone's large receptive field smooths away.
- **PPAdapter** (Pyramid Pooling Adapter): average pooling at scales
  1/2/3/6, concatenated, fused by a 1×1 conv, same zero-initialized gated
  residual — adds multi-scale context.

### Disagreement-gated fusion (`src/models/heads.py`, `fusion.py`)
1. **Auxiliary heads**: a small head on each branch's deepest adapted tap
   predicts water independently (no coupling between the two heads),
   deep-supervised against the ground truth.
2. **Disagreement map D**: the Bernoulli Jensen-Shannon divergence between
   the two branches' auxiliary water probabilities, computed in fp32 at
   the coarse 32×32 grid (bf16 underflows the JSD log near 1.0) and
   detached before it reaches the gate — a label-free signal of where the
   two branches disagree, not a gradient path.
3. **Gated fusion**: at each of the four tapped depths, both branches are
   projected to 256 channels; a small conv on `concat[proj_a, proj_b, D]`
   produces a per-pixel softmax over the two branches, replacing a single
   learned scalar gate per branch with a spatially varying one.

### Decoder — LinkNet (`src/models/decoders_v2.py: LinkNetDecoder`)
Reassembles the four fused depth maps with additive skip connections to
full resolution. Outperformed UPerNet and DPT decoders in the architecture
search at matched adapter configuration.

## Training

AdamW, lr 5e-4 (only the adapters/fusion/decoder/aux-heads are trainable;
both backbones stay frozen throughout), cosine schedule with warmup, bf16
autocast, gradient clip 1.0, batch size 8, 512×512 random crops with
flips/90° rotations, 80 epochs, full 1024×1024 evaluation. Loss is
BCE + Dice + a boundary-band-weighted BCE (3-pixel band) on the fused
output, plus 0.3×(BCE + Dice) on each auxiliary head independently.
Normalization: FloodPlanet uses fixed training-set statistics, UFO uses
per-image statistics (see `src/paths.py` / `configs/band_stats.json`).

**Parameters:** ~65M trainable (adapters, fusion, decoder, aux heads) /
~614M frozen (both ViT-L backbones) / ~679M total.
