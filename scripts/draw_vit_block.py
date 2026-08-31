"""
Modified ViT block diagram showing:
  - Frozen MHSA (unmodified)
  - FFNWithEA: original FFN (frozen) + EarthAdapter (trainable) in parallel
  - EarthAdapter internals: DFT split → 3 experts → softmax router → ×α
  - Post-hoc ADAC + PPAdapter outside the block (at tap depths)

Saved to:
  outputs/viz/vit_block.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("outputs/viz")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

# ── Palette ───────────────────────────────────────────────────────────────────
C = dict(
    frozen_f="#F2F2F2", frozen_e="#888888",   # frozen components
    ln_f="#FFF8DC",     ln_e="#B8860B",        # LayerNorm
    attn_f="#E8E8F8",   attn_e="#5555AA",      # MHSA (frozen)
    ffn_f="#F0F0F0",    ffn_e="#666666",        # FFN (frozen)
    ea_f="#EDE8FC",     ea_e="#6E33A8",         # EarthAdapter (trainable)
    dft_f="#D8F0FC",    dft_e="#1A6EA8",        # DFT split
    exp_f="#E0F5E8",    exp_e="#2A7A40",        # experts
    router_f="#FEF0DC", router_e="#C06820",     # router
    alpha_f="#FCE8EC",  alpha_e="#B02040",      # alpha scale
    add_f="#FFFFFF",    add_e="#444444",         # add circles
    adac_f="#FFFADC",   adac_e="#B8860B",       # ADAC+PPAd (post-hoc)
    block_border="#AAAAAA",
    arr="#444444",
    trainable_tag="#6E33A8",
    frozen_tag="#888888",
)

W, H = 13.0, 18.0
fig, ax = plt.subplots(figsize=(W, H), facecolor="white")
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_facecolor("white"); ax.axis("off")

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.2, r=0.12, zorder=3, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)

def arr(x1, y1, x2, y2, col=C["arr"], lw=1.4, ms=9,
        style="arc3,rad=0.0", zorder=6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=col,
        connectionstyle=style, zorder=zorder))

def line(x1, y1, x2, y2, col=C["arr"], lw=1.2, ls="-", zorder=5):
    ax.plot([x1, x2], [y1, y2], color=col, lw=lw, ls=ls, zorder=zorder)

def lbl(x, y, s, fs=8, ha="center", va="center", col="#222",
        bold=False, rot=0, zorder=7):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=col,
            fontweight="bold" if bold else "normal", rotation=rot,
            zorder=zorder, fontfamily="DejaVu Sans", linespacing=1.35)

def add_circle(cx, cy, r=0.22, col=C["add_e"], zorder=6):
    c = plt.Circle((cx, cy), r, fc=C["add_f"], ec=col, lw=1.3, zorder=zorder)
    ax.add_patch(c)
    lbl(cx, cy, "+", fs=9, bold=True, col=col, zorder=zorder+1)

def tag(x, y, txt, col):
    """Small italic tag label."""
    ax.text(x, y, txt, fontsize=6.5, ha="left", va="center",
            color=col, fontstyle="italic", zorder=8)

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
MID   = W / 2          # 6.5 — centre of figure
BX0   = 0.4            # block left edge
BX1   = W - 0.4        # block right edge
BW    = BX1 - BX0      # block width

# Y positions (bottom → top)
Y_OUT_BOTTOM  = 1.0    # arrow out of block to ADAC
Y_BLOCK_BOT   = 2.6    # block outer border bottom
Y_EA_BOT      = 2.8    # FFNWithEA zone bottom
Y_FFN_BOT     = 3.2    # inner FFN bottom
Y_FFN_TOP     = 5.8    # inner FFN top
Y_EA_ADD      = 6.5    # ⊕ where FFN + EA delta merge
Y_LN2         = 7.2    # LN2 box bottom
Y_ADD1        = 8.2    # first residual add
Y_MHSA_BOT    = 8.8    # MHSA bottom
Y_MHSA_TOP    = 11.2   # MHSA top
Y_LN1         = 11.6   # LN1 bottom
Y_BLOCK_TOP   = 15.8   # block outer border top
Y_IN          = 16.3   # input token flow top

# X split for two-column FFNWithEA zone
FFN_XM  = 4.8          # left column centre (FFN path)
EA_XL   = 5.5          # right column left edge (EA path)
EA_XR   = BX1 - 0.15  # right column right edge

# ─────────────────────────────────────────────────────────────────────────────
# 0.  OUTER BLOCK BORDER
# ─────────────────────────────────────────────────────────────────────────────
rbox(BX0, Y_BLOCK_BOT, BW, Y_BLOCK_TOP - Y_BLOCK_BOT,
     fc="#FAFAFA", ec=C["block_border"], lw=1.8, r=0.3, zorder=1)
lbl(BX0 + 0.18, (Y_BLOCK_BOT + Y_BLOCK_TOP)/2,
    "Transformer Block  (×24, all blocks modified)", fs=8, bold=True,
    col="#555", rot=90, zorder=2)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  INPUT TOKENS  (top)
# ─────────────────────────────────────────────────────────────────────────────
lbl(MID, Y_IN + 0.15, "Token sequence  x  (B, N + prefix, D)", fs=9, bold=True)
arr(MID, Y_IN - 0.1, MID, Y_BLOCK_TOP, col=C["arr"], lw=1.6, ms=10)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MHSA  (frozen, unmodified)
# ─────────────────────────────────────────────────────────────────────────────
# LN1
LN1_W = 3.2; LN1_H = 0.55
LN1_X = MID - LN1_W/2
rbox(LN1_X, Y_LN1, LN1_W, LN1_H, fc=C["ln_f"], ec=C["ln_e"], lw=1.2)
lbl(MID, Y_LN1 + LN1_H/2, "Layer Norm", fs=8)
arr(MID, Y_BLOCK_TOP, MID, Y_LN1 + LN1_H, col=C["arr"], lw=1.4, ms=9)

# MHSA box
MHSA_W = 5.5; MHSA_H = Y_MHSA_TOP - Y_MHSA_BOT
MHSA_X = MID - MHSA_W/2
rbox(MHSA_X, Y_MHSA_BOT, MHSA_W, MHSA_H, fc=C["attn_f"], ec=C["attn_e"], lw=1.5)
lbl(MID, (Y_MHSA_BOT + Y_MHSA_TOP)/2 + 0.30,
    "Multi-Head Self-Attention", fs=9, bold=True, col=C["attn_e"])
lbl(MID, (Y_MHSA_BOT + Y_MHSA_TOP)/2 - 0.10,
    "Q = K = V = LN(x)", fs=8, col="#555")
lbl(MID, (Y_MHSA_BOT + Y_MHSA_TOP)/2 - 0.50,
    "Attn(Q, K, V)  →  projection", fs=8, col="#555")
arr(MID, Y_LN1, MID, Y_MHSA_TOP, col=C["arr"], lw=1.4, ms=9)

# frozen tag on MHSA
tag(MHSA_X + MHSA_W + 0.08, (Y_MHSA_BOT + Y_MHSA_TOP)/2,
    "❄ frozen", C["frozen_tag"])

# first residual add  (x + MHSA output)
ADD1_Y = Y_ADD1
add_circle(MID, ADD1_Y)
arr(MID, Y_MHSA_BOT, MID, ADD1_Y + 0.22, col=C["arr"], lw=1.4, ms=9)
# skip connection around MHSA
SKP_X = MHSA_X - 0.45
line(SKP_X, Y_BLOCK_TOP - 0.15, SKP_X, ADD1_Y, col="#888", lw=1.1, ls="--")
line(SKP_X, Y_BLOCK_TOP - 0.15, MID - 0.02, Y_BLOCK_TOP - 0.15,
     col="#888", lw=1.1, ls="--")
ax.add_patch(FancyArrowPatch((SKP_X, ADD1_Y), (MID - 0.22, ADD1_Y),
    arrowstyle="-|>", mutation_scale=8, linewidth=1.1,
    color="#888", linestyle="dashed", zorder=5))
lbl(SKP_X - 0.20, (Y_BLOCK_TOP - 0.15 + ADD1_Y)/2, "skip", fs=6.5,
    col="#999", rot=90)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LN2  (between MHSA output and FFNWithEA)
# ─────────────────────────────────────────────────────────────────────────────
LN2_W = 3.2; LN2_H = 0.55
LN2_X = MID - LN2_W/2
rbox(LN2_X, Y_LN2, LN2_W, LN2_H, fc=C["ln_f"], ec=C["ln_e"], lw=1.2)
lbl(MID, Y_LN2 + LN2_H/2, "Layer Norm", fs=8)
arr(MID, ADD1_Y - 0.22, MID, Y_LN2 + LN2_H, col=C["arr"], lw=1.4, ms=9)

# LN2 output splits into two columns
LN2_OUT_Y = Y_LN2
# left column centre (FFN): FFN_XM
# right column centre (EA): (EA_XL + EA_XR)/2
EA_XM = (EA_XL + EA_XR) / 2
arr(MID, LN2_OUT_Y, FFN_XM, Y_FFN_TOP + 0.05, col=C["ffn_e"],
    lw=1.2, ms=8, style="arc3,rad=0.15")
arr(MID, LN2_OUT_Y, EA_XM, Y_EA_BOT + (Y_EA_ADD - Y_EA_BOT) - 0.1,
    col=C["ea_e"], lw=1.2, ms=8, style="arc3,rad=-0.15")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  LEFT PATH: FFN (frozen)
# ─────────────────────────────────────────────────────────────────────────────
FFN_W = 3.0; FFN_H = Y_FFN_TOP - Y_FFN_BOT
FFN_X = FFN_XM - FFN_W/2

rbox(FFN_X, Y_FFN_BOT, FFN_W, FFN_H, fc=C["ffn_f"], ec=C["ffn_e"], lw=1.4, ls="--")
lbl(FFN_XM, Y_FFN_BOT + FFN_H/2 + 0.40, "FFN", fs=9, bold=True, col=C["ffn_e"])
lbl(FFN_XM, Y_FFN_BOT + FFN_H/2 + 0.00, "Linear(D, 4D)", fs=7.5, col="#666")
lbl(FFN_XM, Y_FFN_BOT + FFN_H/2 - 0.35, "GELU", fs=7.5, col="#666")
lbl(FFN_XM, Y_FFN_BOT + FFN_H/2 - 0.70, "Linear(4D, D)", fs=7.5, col="#666")
tag(FFN_X + FFN_W + 0.06, Y_FFN_BOT + FFN_H/2, "❄ frozen", C["frozen_tag"])
# FFN output arrow (down towards ⊕)
arr(FFN_XM, Y_FFN_BOT, FFN_XM, Y_EA_ADD + 0.22, col=C["ffn_e"], lw=1.2, ms=8)
lbl(FFN_XM - 0.55, (Y_FFN_BOT + Y_EA_ADD)/2, "FFN(x)", fs=7.5,
    col=C["ffn_e"], ha="right")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  RIGHT PATH: EarthAdapter (trainable)
# ─────────────────────────────────────────────────────────────────────────────
EA_BW = EA_XR - EA_XL

# — 5a. Strip prefix note ——————————————————————————————————————————————
lbl(EA_XM, Y_EA_BOT + (Y_EA_ADD - Y_EA_BOT) - 0.28,
    "patch tokens only (strip cls/prefix)", fs=7, col="#888")

# — 5b. DFT split box ——————————————————————————————————————————————————
DFT_H = 0.90; DFT_W = EA_BW - 0.20
DFT_Y = Y_EA_BOT + (Y_EA_ADD - Y_EA_BOT) - 1.55
DFT_X = EA_XL + 0.10
DFT_XM = DFT_X + DFT_W / 2
rbox(DFT_X, DFT_Y, DFT_W, DFT_H, fc=C["dft_f"], ec=C["dft_e"], lw=1.3)
lbl(DFT_XM, DFT_Y + DFT_H/2 + 0.14,
    "2-D FFT  →  shift  →  square mask (radius ρ)", fs=7.5, col=C["dft_e"])
lbl(DFT_XM, DFT_Y + DFT_H/2 - 0.20,
    "iFFT(F·mask) → LF     iFFT(F·~mask) → HF", fs=7, col="#555")

# arrows into DFT box
arr(EA_XM, Y_EA_BOT + (Y_EA_ADD - Y_EA_BOT) - 0.46,
    DFT_XM, DFT_Y + DFT_H, col=C["dft_e"], lw=1.2, ms=8)

# — 5c. Three experts in parallel ——————————————————————————————————————
EXP_H = 1.10; EXP_W = 1.60
EXP_Y = DFT_Y - 1.55
EXP_GAP = 0.22
EXP_TOTAL = 3 * EXP_W + 2 * EXP_GAP
EXP_X0 = EA_XL + (EA_BW - EXP_TOTAL) / 2

exp_data = [
    ("Spatial Expert", "x (raw tokens)", C["exp_f"], C["exp_e"]),
    ("LF Expert",      "LF tokens",      C["exp_f"], C["exp_e"]),
    ("HF Expert",      "HF tokens",      C["exp_f"], C["exp_e"]),
]
exp_xm = []
for i, (nm, inp, fc, ec) in enumerate(exp_data):
    ex = EXP_X0 + i * (EXP_W + EXP_GAP)
    exm = ex + EXP_W/2
    exp_xm.append(exm)
    rbox(ex, EXP_Y, EXP_W, EXP_H, fc=fc, ec=ec, lw=1.2)
    lbl(exm, EXP_Y + EXP_H/2 + 0.23, nm, fs=7.5, bold=True, col=ec)
    lbl(exm, EXP_Y + EXP_H/2 - 0.05, f"in: {inp}", fs=6.5, col="#555")
    lbl(exm, EXP_Y + EXP_H/2 - 0.30, "Lin(D,b)→ReLU→Lin(b,D)", fs=6.0, col="#777")
    lbl(exm, EXP_Y + EXP_H/2 - 0.52, "(b = bottleneck = 32)", fs=5.8, col="#AAA")

# arrows: DFT outputs → experts
# Spatial gets raw x (from above DFT, pass-through)
arr(EA_XM, DFT_Y, exp_xm[0], EXP_Y + EXP_H,
    col=C["exp_e"], lw=1.0, ms=7, style="arc3,rad=0.15")
lbl((EA_XM + exp_xm[0])/2 - 0.35, DFT_Y - 0.45, "x", fs=7.5, col="#555")
# LF → LF expert
arr(DFT_XM - 0.5, DFT_Y, exp_xm[1], EXP_Y + EXP_H,
    col=C["dft_e"], lw=1.0, ms=7)
lbl(DFT_XM - 0.35, DFT_Y - 0.42, "LF", fs=7.5, col=C["dft_e"])
# HF → HF expert
arr(DFT_XM + 0.5, DFT_Y, exp_xm[2], EXP_Y + EXP_H,
    col=C["dft_e"], lw=1.0, ms=7)
lbl(DFT_XM + 0.55, DFT_Y - 0.42, "HF", fs=7.5, col=C["dft_e"])

# — 5d. Router ——————————————————————————————————————————————————————————
RTR_H = 0.72; RTR_W = 2.0
RTR_Y = EXP_Y - 1.20
RTR_X = EA_XL + (EA_BW - RTR_W)/2
RTR_XM = RTR_X + RTR_W/2
rbox(RTR_X, RTR_Y, RTR_W, RTR_H, fc=C["router_f"], ec=C["router_e"], lw=1.3)
lbl(RTR_XM, RTR_Y + RTR_H/2 + 0.14, "Router", fs=8, bold=True, col=C["router_e"])
lbl(RTR_XM, RTR_Y + RTR_H/2 - 0.18,
    "Linear(D,3) on mean(x) → softmax", fs=7, col="#666")
lbl(RTR_XM, RTR_Y + RTR_H/2 - 0.44, "→ gates  g₁, g₂, g₃", fs=7.5, col=C["router_e"])

# arrow: x → router (from above DFT area, x fed separately)
arr(EA_XM, DFT_Y, RTR_XM, RTR_Y + RTR_H,
    col=C["router_e"], lw=1.0, ms=7, style="arc3,rad=-0.25")
lbl(EA_XM + 0.10, (DFT_Y + RTR_Y + RTR_H)/2, "x.mean(N)", fs=6.5,
    col=C["router_e"], ha="left")

# — 5e. Weighted sum circle ————————————————————————————————————————————
WSUM_Y = RTR_Y - 0.85
WSUM_X = RTR_XM
add_circle(WSUM_X, WSUM_Y, r=0.25)
lbl(WSUM_X + 0.40, WSUM_Y + 0.20, "Σ  gₖ · expertₖ(·)", fs=7.5, col="#333", ha="left")

# arrows: experts → weighted sum
for exm_x in exp_xm:
    arr(exm_x, EXP_Y, WSUM_X, WSUM_Y + 0.25,
        col=C["exp_e"], lw=1.0, ms=6, style="arc3,rad=0.0")
# router → weighted sum
arr(RTR_XM, RTR_Y, WSUM_X, WSUM_Y + 0.25, col=C["router_e"], lw=1.0, ms=7)
lbl(RTR_XM + 0.28, (RTR_Y + WSUM_Y)/2, "gates", fs=6.5, col=C["router_e"], ha="left")

# — 5f. α scale ————————————————————————————————————————————————————————
ALP_H = 0.55; ALP_W = 1.30
ALP_Y = WSUM_Y - 1.05
ALP_X = WSUM_X - ALP_W/2
rbox(ALP_X, ALP_Y, ALP_W, ALP_H, fc=C["alpha_f"], ec=C["alpha_e"], lw=1.3)
lbl(WSUM_X, ALP_Y + ALP_H/2 + 0.02, "× α  (zero-init)", fs=7.5,
    bold=True, col=C["alpha_e"])
arr(WSUM_X, WSUM_Y - 0.25, WSUM_X, ALP_Y + ALP_H, col=C["alpha_e"], lw=1.2, ms=8)
lbl(WSUM_X + 0.12, (WSUM_Y - 0.25 + ALP_Y + ALP_H)/2,
    "EA(x)", fs=7.5, col=C["ea_e"], ha="left")

# trainable tag on EA block
tag(EA_XR + 0.06, (Y_LN2 + ALP_Y)/2, "✓ trainable", C["trainable_tag"])

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MERGE: FFN(x) + EA(x)  →  second residual add
# ─────────────────────────────────────────────────────────────────────────────
# ⊕ circle at Y_EA_ADD, MID position
ADD2_Y = Y_EA_ADD
ADD2_X = MID
add_circle(ADD2_X, ADD2_Y)

# EA α output → ⊕ (from right column)
arr(WSUM_X, ALP_Y, ADD2_X, ADD2_Y + 0.22,
    col=C["ea_e"], lw=1.3, ms=8, style="arc3,rad=-0.15")

# FFN output already arrowed at FFN_XM → ADD2 area
# (drawn above at step 4 - arrow from FFN bottom to Y_EA_ADD)
# Add the horizontal join
line(FFN_XM, Y_EA_ADD, ADD2_X - 0.22, ADD2_Y, col=C["ffn_e"], lw=1.3)

# label the two inputs to ⊕
lbl(ADD2_X - 0.72, ADD2_Y + 0.30, "FFN(x)", fs=7.5, col=C["ffn_e"])
lbl(ADD2_X + 0.55, ADD2_Y + 0.30, "α·EA(x)", fs=7.5, col=C["ea_e"])

# second residual add (x + FFNWithEA output)
ADD3_Y = Y_EA_ADD - 0.85
add_circle(ADD2_X, ADD3_Y)
arr(ADD2_X, ADD2_Y - 0.22, ADD2_X, ADD3_Y + 0.22, col=C["arr"], lw=1.4, ms=9)

# skip around entire FFNWithEA
SKP2_X = BX1 - 0.30
line(SKP2_X, ADD1_Y - 0.22, SKP2_X, ADD3_Y, col="#888", lw=1.1, ls="--")
line(SKP2_X, ADD1_Y - 0.22, ADD2_X + 0.02, ADD1_Y - 0.22,
     col="#888", lw=1.1, ls="--")
ax.add_patch(FancyArrowPatch((SKP2_X, ADD3_Y), (ADD2_X + 0.22, ADD3_Y),
    arrowstyle="-|>", mutation_scale=8, linewidth=1.1,
    color="#888", linestyle="dashed", zorder=5))
lbl(SKP2_X + 0.20, (ADD1_Y + ADD3_Y)/2, "skip", fs=6.5, col="#999", rot=90)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  TAP OUTPUT (arrow down from block)  +  ADAC/PPAd (outside block)
# ─────────────────────────────────────────────────────────────────────────────
TAP_Y  = Y_BLOCK_BOT
arr(ADD2_X, ADD3_Y - 0.22, ADD2_X, TAP_Y, col=C["arr"], lw=1.6, ms=10)
lbl(ADD2_X + 0.45, (ADD3_Y - 0.22 + TAP_Y)/2 + 0.1,
    "output tokens", fs=7.5, col="#555", ha="left")
lbl(ADD2_X + 0.45, (ADD3_Y - 0.22 + TAP_Y)/2 - 0.2,
    "(B, N+prefix, D)", fs=7, col="#888", ha="left")

# tap note
lbl(MID, TAP_Y - 0.22,
    "↑  tapped at block depths  5 · 11 · 17 · 23  →  reshape to (B, D, H, W)",
    fs=7.5, col="#555")

# ADAC + PPAdapter boxes side by side
ADAC_W = 2.0; ADAC_H = 0.80
ADAC_GAP = 0.4
ADAC_TOTAL = 2 * ADAC_W + ADAC_GAP
ADAC_X0 = MID - ADAC_TOTAL/2
ADAC_Y = Y_OUT_BOTTOM - 0.30

adac_items = [("ADAC", "Atrous Depth-Wise Conv\nzero-gated residual  γ·y"),
              ("PPAdapter", "Pyramid Pool Context\nzero-gated residual  γ·y")]
for j, (nm, sub) in enumerate(adac_items):
    ax_ = ADAC_X0 + j * (ADAC_W + ADAC_GAP)
    rbox(ax_, ADAC_Y, ADAC_W, ADAC_H, fc=C["adac_f"], ec=C["adac_e"], lw=1.4)
    lbl(ax_ + ADAC_W/2, ADAC_Y + ADAC_H/2 + 0.18, nm, fs=8, bold=True,
        col=C["adac_e"])
    lbl(ax_ + ADAC_W/2, ADAC_Y + ADAC_H/2 - 0.18, sub, fs=6.5, col="#666")

# arrow from tap → ADAC zone
arr(MID, TAP_Y - 0.44, MID, ADAC_Y + ADAC_H, col=C["arr"], lw=1.4, ms=9)

# post-hoc brace label
ax.annotate("", xy=(ADAC_X0 - 0.06, ADAC_Y + ADAC_H + 0.05),
            xytext=(ADAC_X0 + ADAC_TOTAL + 0.06, ADAC_Y + ADAC_H + 0.05),
            arrowprops=dict(arrowstyle="<->", color="#B8860B", lw=1.0))
lbl(MID, ADAC_Y + ADAC_H + 0.22, "post-hoc  (outside block, per tap)",
    fs=7, col=C["adac_e"])
tag(ADAC_X0 + ADAC_TOTAL + 0.08, ADAC_Y + ADAC_H/2, "✓ trainable", C["trainable_tag"])

# arrow out to fusion
arr(MID, ADAC_Y, MID, 0.15, col="#333", lw=1.6, ms=10)
lbl(MID, 0.08, "→ MultiDepthFusion  (one map per tap depth)", fs=8, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# 8.  FROZEN / TRAINABLE LEGEND (top-left)
# ─────────────────────────────────────────────────────────────────────────────
LGX, LGY = 0.55, H - 0.45
legend_items = [
    (C["attn_f"], C["attn_e"],   "-",  "MHSA  (frozen, untouched)"),
    (C["ffn_f"],  C["ffn_e"],    "--", "Original FFN  (frozen)"),
    (C["ea_f"],   C["ea_e"],     "-",  "EarthAdapter  (trainable)"),
    (C["adac_f"], C["adac_e"],   "-",  "ADAC + PPAdapter  (trainable, post-hoc)"),
]
for k, (fc, ec, ls, txt) in enumerate(legend_items):
    ly = LGY - k * 0.38
    rbox(LGX, ly - 0.15, 0.38, 0.30, fc=fc, ec=ec, lw=1.0, r=0.04, ls=ls, zorder=8)
    lbl(LGX + 0.56, ly, txt, fs=7.5, ha="left", col="#333", zorder=8)

# figure title
lbl(MID, H - 0.22,
    "Modified ViT Block  (arch5: Earth-Adapter + ADAC + PPAdapter)",
    fs=10, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  SAVE
# ─────────────────────────────────────────────────────────────────────────────
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"vit_block.{fmt}",
                dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}/vit_block.{{pdf,png}}")
