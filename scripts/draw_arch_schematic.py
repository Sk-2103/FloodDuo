"""
Architecture schematic with detailed DepthFusion internals.
figsize = (27, 10) to give space for the fusion zoom-in.

Saved to:
  outputs/viz/arch_schematic.pdf/.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("outputs/viz")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

C = dict(
    dino_f="#FEF0E3", dino_e="#C0622A",
    clay_f="#E4F5EE", clay_e="#1D7A50",
    ea_f  ="#F0E8FC", ea_e  ="#6E33A8",
    adac_f="#FFFADC", adac_e="#B8860B",
    fuse_f="#EAF3FC", fuse_e="#1A6EA8",    # outer fusion box
    fi_f  ="#D4E8F8", fi_e  ="#1A6EA8",    # inner fusion sub-boxes
    gate_f="#FDE8F2", gate_e="#B02060",    # gate symbols
    cat_f ="#EEEEFF", cat_e ="#3333BB",    # concat box
    dec_f ="#E8F5E9", dec_e ="#2E7D32",
    arr   ="#404040",
    gray  ="#AAAAAA",
)

W, H = 27.0, 10.0
fig, ax = plt.subplots(figsize=(W, H), facecolor="white")
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_facecolor("white"); ax.axis("off")

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.3, r=0.10, zorder=3, alpha=1.0, ls="-"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle=ls, alpha=alpha, zorder=zorder)
    ax.add_patch(p); return p

def arr(x1, y1, x2, y2, col=C["arr"], lw=1.5, ms=10,
        style="arc3,rad=0.0", zorder=6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=col,
        connectionstyle=style, zorder=zorder))

def line(x1, y1, x2, y2, col=C["gray"], lw=1.2, ls="-", zorder=5):
    ax.plot([x1, x2], [y1, y2], color=col, lw=lw, ls=ls, zorder=zorder)

def lbl(x, y, s, fs=8, ha="center", va="center", col="#222",
        bold=False, rot=0, zorder=7):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=col,
            fontweight="bold" if bold else "normal", rotation=rot,
            zorder=zorder, fontfamily="DejaVu Sans", linespacing=1.35)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  INPUT TILE
# ═══════════════════════════════════════════════════════════════════════════════
TX, TY, TW, TH = 0.25, 3.1, 1.15, 3.8
bh = TH / 4
for i, (bc, bn) in enumerate(zip(
        ["#78C7E8", "#7ED99E", "#EF9B9B", "#C4A87A"], ["B", "G", "R", "NIR"])):
    by = TY + (3-i)*bh
    rbox(TX, by, TW, bh, fc=bc, ec="#666", lw=0.7, r=0.04)
    lbl(TX+TW/2, by+bh/2, bn, fs=8.5, bold=True)
rbox(TX-.05, TY-.05, TW+.1, TH+.1, fc="none", ec="#333", lw=2.0, r=0.14, zorder=2)
lbl(TX+TW/2, TY-.45, "PlanetScope input\n4 bands · 512×512 · 3 m GSD", fs=7.5)
lbl(TX+TW/2, TY+TH+.32, "Input", fs=10, bold=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ENCODER BARS  (DINOv2 top, Clay bottom)
# ═══════════════════════════════════════════════════════════════════════════════
DINO_Y0, DINO_YE = 6.8, 8.8
CLAY_Y0, CLAY_YE = 1.2, 3.2
ENC_H  = 2.0
DINO_YM = (DINO_Y0 + DINO_YE) / 2
CLAY_YM = (CLAY_Y0 + CLAY_YE) / 2

# Patch embeddings
PE_X0, PE_W = 2.0, 0.85
rbox(PE_X0, DINO_Y0, PE_W, ENC_H, fc=C["dino_f"], ec=C["dino_e"], lw=1.6)
lbl(PE_X0+PE_W/2, DINO_YM+.30, "Patch Embed",  fs=7)
lbl(PE_X0+PE_W/2, DINO_YM-.05, "16×16 px",     fs=6.5, col="#666")
lbl(PE_X0+PE_W/2, DINO_YM-.40, "+ pos. enc.",  fs=6,   col="#999")

rbox(PE_X0, CLAY_Y0, PE_W, ENC_H, fc=C["clay_f"], ec=C["clay_e"], lw=1.6)
lbl(PE_X0+PE_W/2, CLAY_YM+.42, "Patch Embed",   fs=7)
lbl(PE_X0+PE_W/2, CLAY_YM+.10, "8×8 px",        fs=6.5, col="#666")
lbl(PE_X0+PE_W/2, CLAY_YM-.22, "λ-conditioned", fs=6.5, col=C["clay_e"], bold=True)
lbl(PE_X0+PE_W/2, CLAY_YM-.55, "GSD pos. enc.", fs=6,   col="#999")

# Transformer blocks + EA
BX0    = PE_X0 + PE_W + 0.07
BSEG_W = 1.30
EA_W   = 0.48
SEP    = 0.05
seg_lbls = ["Blocks\n0 – 4", "Blocks\n6 – 10", "Blocks\n12 – 16", "Blocks\n18 – 22"]
ea_idx   = [5, 11, 17, 23]
tap_cx   = []
bx = BX0
for k in range(4):
    for (y0, fc, ec) in [(DINO_Y0, C["dino_f"], C["dino_e"]),
                          (CLAY_Y0, C["clay_f"], C["clay_e"])]:
        rbox(bx, y0, BSEG_W, ENC_H, fc=fc, ec=ec, lw=1.0, r=0.08)
    lbl(bx+BSEG_W/2, DINO_YM, seg_lbls[k], fs=6.5)
    lbl(bx+BSEG_W/2, CLAY_YM, seg_lbls[k], fs=6.5)
    bx += BSEG_W + SEP
    cx = bx + EA_W/2
    for y0 in (DINO_Y0, CLAY_Y0):
        rbox(bx, y0, EA_W, ENC_H, fc=C["ea_f"], ec=C["ea_e"], lw=1.8, r=0.08)
    lbl(cx, DINO_YM+.30, "EA", fs=9.5, bold=True, col=C["ea_e"])
    lbl(cx, DINO_YM-.28, f"blk {ea_idx[k]}", fs=5.5, col="#888")
    lbl(cx, CLAY_YM+.30, "EA", fs=9.5, bold=True, col=C["ea_e"])
    lbl(cx, CLAY_YM-.28, f"blk {ea_idx[k]}", fs=5.5, col="#888")
    tap_cx.append(cx)
    bx += EA_W + SEP
ENC_END_X = bx - SEP

lbl(PE_X0-.12, DINO_YM, "DINOv2-L\n(ViT-L/16\nRGB)", fs=8.5, bold=True, ha="right", col=C["dino_e"])
lbl(PE_X0-.12, CLAY_YM, "Clay v1.5\n(ViT-L/8\nBGRNIR)", fs=8.5, bold=True, ha="right", col=C["clay_e"])

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BAND-EXTRACTION ARROWS
# ═══════════════════════════════════════════════════════════════════════════════
arr(TX+TW, TY+.80*TH, PE_X0, DINO_YM, col=C["dino_e"], lw=2.0, ms=12)
lbl(1.62, 7.70, "RGB",    fs=8.5, bold=True, col=C["dino_e"])
arr(TX+TW, TY+.20*TH, PE_X0, CLAY_YM, col=C["clay_e"], lw=2.0, ms=12)
lbl(1.62, 2.25, "BGRNIR", fs=8.5, bold=True, col=C["clay_e"])

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ADAC + PPAdapter (post-hoc tap at each EA depth)
# ═══════════════════════════════════════════════════════════════════════════════
AW, AH = 0.85, 0.65
ADAC_DY = DINO_Y0 - 1.05   # bottom of DINOv2 ADAC boxes
ADAC_CY = CLAY_YE + 0.20   # bottom of Clay ADAC boxes
for tx in tap_cx:
    xl = tx - AW/2
    arr(tx, DINO_Y0, tx, ADAC_DY+AH, col="#888", lw=1.1, ms=7)
    rbox(xl, ADAC_DY, AW, AH, fc=C["adac_f"], ec=C["adac_e"], lw=1.2, r=0.08)
    lbl(tx, ADAC_DY+AH/2, "ADAC\n+PPAd.", fs=5.8, col="#6B5000")
    arr(tx, CLAY_YE, tx, ADAC_CY, col="#888", lw=1.1, ms=7)
    rbox(xl, ADAC_CY, AW, AH, fc=C["adac_f"], ec=C["adac_e"], lw=1.2, r=0.08)
    lbl(tx, ADAC_CY+AH/2, "ADAC\n+PPAd.", fs=5.8, col="#6B5000")
lbl(tap_cx[0]-AW/2-.10, ADAC_DY+AH/2, "post-\nhoc", fs=5.5, col="#999", ha="right")
lbl(tap_cx[0]-AW/2-.10, ADAC_CY+AH/2, "post-\nhoc", fs=5.5, col="#999", ha="right")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  BUNDLED ARROWS → Fusion  (one per branch, labeled with 4×)
#     Collecting horizontal line from each ADAC row → right, then single arrow
# ═══════════════════════════════════════════════════════════════════════════════
BUS_X   = ENC_END_X + 0.18   # vertical bus x
FUSE_X0 = ENC_END_X + 0.55   # left edge of fusion outer box

DINO_BUS_Y = ADAC_DY + AH/2
CLAY_BUS_Y = ADAC_CY + AH/2

# horizontal collector lines (ADAC right → bus)
for tx in tap_cx:
    line(tx+AW/2, DINO_BUS_Y, BUS_X, DINO_BUS_Y, col=C["dino_e"], lw=1.0)
    line(tx+AW/2, CLAY_BUS_Y, BUS_X, CLAY_BUS_Y, col=C["clay_e"], lw=1.0)

# bus tick marks
for col_, y_ in [(C["dino_e"], DINO_BUS_Y), (C["clay_e"], CLAY_BUS_Y)]:
    line(BUS_X, y_-.12, BUS_X, y_+.12, col=col_, lw=3.0)

# one thick arrow per branch: bus → fusion
FUSE_DINO_Y = 6.30   # where DINOv2 path enters fusion
FUSE_CLAY_Y = 3.70   # where Clay path enters fusion

arr(BUS_X, DINO_BUS_Y, FUSE_X0, FUSE_DINO_Y,
    col=C["dino_e"], lw=2.0, ms=11, style="arc3,rad=0.15")
lbl(BUS_X+.35, DINO_BUS_Y+.38,
    "fa: 4×(1024-ch, 32²)", fs=7, col=C["dino_e"], ha="left")

arr(BUS_X, CLAY_BUS_Y, FUSE_X0, FUSE_CLAY_Y,
    col=C["clay_e"], lw=2.0, ms=11, style="arc3,rad=-0.15")
lbl(BUS_X+.35, CLAY_BUS_Y-.38,
    "fb: 4×(1024-ch, 64²)", fs=7, col=C["clay_e"], ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MULTI-DEPTH FUSION  (outer box + internal mechanism)
# ═══════════════════════════════════════════════════════════════════════════════
FUSE_Y0 = 2.90
FUSE_YE = 7.50
FUSE_W  = 7.20
FUSE_H  = FUSE_YE - FUSE_Y0
FUSE_XE = FUSE_X0 + FUSE_W
FUSE_YM = (FUSE_Y0 + FUSE_YE) / 2

# outer box
rbox(FUSE_X0, FUSE_Y0, FUSE_W, FUSE_H, fc=C["fuse_f"], ec=C["fuse_e"],
     lw=2.0, r=0.18, zorder=2)

# header: "DepthFusion  ×4 depth taps  (independent weights per tap)"
lbl(FUSE_X0+FUSE_W/2, FUSE_YE-.28,
    "DepthFusion  ×4 depth taps (independent weights per tap)",
    fs=8, bold=True, col=C["fuse_e"])

# — Internal layout (two horizontal rows + merge column) ———————————————————
# DINOv2 row y-centre:
DFY = 6.15   # DINOv2 internal row
CFY = 3.85   # Clay internal row
MFY = (DFY + CFY) / 2   # merge / concat / fuse row  ≈ 5.0

# Column x positions (left edge of each sub-box)
PAD  = 0.25                  # padding from outer box left
x0   = FUSE_X0 + PAD         # start x inside fusion box
BW   = 1.10                  # sub-box width
BHr  = 0.72                  # sub-box height (row boxes)
BGw  = 0.50                  # gate box width
BCw  = 0.72                  # concat box width
BFw  = 1.10                  # fuse box width

# positions
xUPS  = x0              # upsample (DINOv2 only, dashed)
xPROJ = xUPS + BW + .18  # projection 1×1 Conv+GN
xGATE = xPROJ + BW + .18 # gate ×α
xMRG  = xGATE + BGw + .22  # merge arrows converge here (x of concat box left)
xCAT  = xMRG             # concat box
xF1   = xCAT + BCw + .22 # 1×1 fuse
xF2   = xF1  + BFw + .18 # 3×3 fuse
xOUT  = xF2  + BFw + .18 # output arrow start

# ── DINO path ──────────────────────────────────────────────────────────────
# upsample box (dashed — only needed when spatial grid differs)
rbox(xUPS, DFY-BHr/2, BW, BHr, fc="#F8F8FF", ec=C["dino_e"],
     lw=1.2, r=0.08, ls="--", zorder=4)
lbl(xUPS+BW/2, DFY+.05,  "Bilinear",  fs=6.5, col=C["dino_e"])
lbl(xUPS+BW/2, DFY-.20,  "upsample",  fs=6.5, col=C["dino_e"])
lbl(xUPS+BW/2, DFY-.44,  "32²→64²",   fs=6.0, col="#888")
lbl(xUPS+BW/2, DFY+BHr/2+.14, "(if needed)", fs=5.5, col="#AAA")
arr(FUSE_X0, DFY, xUPS, DFY, col=C["dino_e"], lw=1.5, ms=9)
arr(xUPS+BW, DFY, xPROJ, DFY, col=C["dino_e"], lw=1.5, ms=9)

# proj_a
rbox(xPROJ, DFY-BHr/2, BW, BHr, fc=C["fi_f"], ec=C["fi_e"], lw=1.2, r=0.08, zorder=4)
lbl(xPROJ+BW/2, DFY+.14, "1×1 Conv",  fs=6.5, col=C["fuse_e"])
lbl(xPROJ+BW/2, DFY-.14, "+ GN",      fs=6.5, col=C["fuse_e"])
lbl(xPROJ+BW/2, DFY-.40, "→ 256-ch",  fs=6.0, col="#555")
arr(xPROJ+BW, DFY, xGATE, DFY, col=C["dino_e"], lw=1.5, ms=9)

# gate_a
rbox(xGATE, DFY-BHr/2, BGw, BHr, fc=C["gate_f"], ec=C["gate_e"], lw=1.2, r=0.08, zorder=4)
lbl(xGATE+BGw/2, DFY+.10, "× α_a",  fs=7.0, bold=True, col=C["gate_e"])
lbl(xGATE+BGw/2, DFY-.20, "learnable", fs=5.5, col="#888")

# ── Clay path ──────────────────────────────────────────────────────────────
# note: Clay is already 64×64, no upsample
lbl(xUPS+BW/2, CFY, "64×64\n(native)", fs=6.5, col=C["clay_e"],
    bold=False, zorder=4)
arr(FUSE_X0, CFY, xPROJ, CFY, col=C["clay_e"], lw=1.5, ms=9)

# proj_b
rbox(xPROJ, CFY-BHr/2, BW, BHr, fc=C["fi_f"], ec=C["fi_e"], lw=1.2, r=0.08, zorder=4)
lbl(xPROJ+BW/2, CFY+.14, "1×1 Conv",  fs=6.5, col=C["fuse_e"])
lbl(xPROJ+BW/2, CFY-.14, "+ GN",      fs=6.5, col=C["fuse_e"])
lbl(xPROJ+BW/2, CFY-.40, "→ 256-ch",  fs=6.0, col="#555")
arr(xPROJ+BW, CFY, xGATE, CFY, col=C["clay_e"], lw=1.5, ms=9)

# gate_b
rbox(xGATE, CFY-BHr/2, BGw, BHr, fc=C["gate_f"], ec=C["gate_e"], lw=1.2, r=0.08, zorder=4)
lbl(xGATE+BGw/2, CFY+.10, "× α_b",    fs=7.0, bold=True, col=C["gate_e"])
lbl(xGATE+BGw/2, CFY-.20, "learnable", fs=5.5, col="#888")

# ── Merge: both gate outputs → concat box ─────────────────────────────────
# vertical lines from gates down/up to MFY, then horizontal to concat
gx_right = xGATE + BGw
line(gx_right, DFY, gx_right+.14, DFY, col=C["arr"])        # DINOv2 stub right
line(gx_right, CFY, gx_right+.14, CFY, col=C["arr"])        # Clay stub right
line(gx_right+.14, CFY, gx_right+.14, DFY, col=C["arr"], lw=1.2)  # vertical join
# horizontal to concat
arr(gx_right+.14, MFY, xCAT, MFY+BCw/2*.0,   # arrow into concat center
    col=C["arr"], lw=1.8, ms=10)

# concat box
rbox(xCAT, MFY-BCw/2, BCw, BCw, fc=C["cat_f"], ec=C["cat_e"], lw=1.5, r=0.08, zorder=4)
lbl(xCAT+BCw/2, MFY+.12, "Cat",    fs=7.5, bold=True, col=C["cat_e"])
lbl(xCAT+BCw/2, MFY-.15, "512-ch", fs=6.5, col=C["cat_e"])
arr(xCAT+BCw, MFY, xF1, MFY, col=C["arr"], lw=1.5, ms=9)

# ── Fuse 1×1 ────────────────────────────────────────────────────────────
BFH = 1.10
rbox(xF1, MFY-BFH/2, BFw, BFH, fc=C["fi_f"], ec=C["fi_e"], lw=1.4, r=0.08, zorder=4)
lbl(xF1+BFw/2, MFY+.28, "1×1 Conv", fs=6.5, col=C["fuse_e"])
lbl(xF1+BFw/2, MFY+.00, "+ GN",     fs=6.5, col=C["fuse_e"])
lbl(xF1+BFw/2, MFY-.28, "+ GELU",   fs=6.5, col=C["fuse_e"])
lbl(xF1+BFw/2, MFY-.52, "→ 256-ch", fs=6.0, col="#555")
arr(xF1+BFw, MFY, xF2, MFY, col=C["arr"], lw=1.5, ms=9)

# ── Fuse 3×3 ────────────────────────────────────────────────────────────
rbox(xF2, MFY-BFH/2, BFw, BFH, fc=C["fi_f"], ec=C["fi_e"], lw=1.4, r=0.08, zorder=4)
lbl(xF2+BFw/2, MFY+.28, "3×3 Conv", fs=6.5, col=C["fuse_e"])
lbl(xF2+BFw/2, MFY+.00, "+ GN",     fs=6.5, col=C["fuse_e"])
lbl(xF2+BFw/2, MFY-.28, "+ GELU",   fs=6.5, col=C["fuse_e"])
lbl(xF2+BFw/2, MFY-.52, "→ 256-ch", fs=6.0, col="#555")

# output label inside fusion
lbl(xF2+BFw+.28, MFY+.18, "256-ch", fs=7, col=C["fuse_e"], bold=True)
lbl(xF2+BFw+.28, MFY-.12, "64×64",  fs=7, col=C["fuse_e"], bold=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  LINKNET DECODER  (4 progressive upsampling blocks)
# ═══════════════════════════════════════════════════════════════════════════════
DEC_X0 = FUSE_XE + 0.42
DW, DG  = 1.05, 0.18
dec_stages = [("dec3","16²→32²"), ("dec2","32²→64²"),
               ("dec1","64²→128²"), ("final","128²→512²")]

arr(FUSE_XE, MFY, DEC_X0, 5.0, col=C["fuse_e"], lw=2.0, ms=12)
lbl(FUSE_XE+.22, MFY+.22, "fused\nfeatures", fs=6.5, col=C["fuse_e"])

dec_blks = []
dx = DEC_X0
for j, (nm, sc) in enumerate(dec_stages):
    bh_j = 0.65 + j*.46
    by_j = 5.0 - bh_j/2
    rbox(dx, by_j, DW, bh_j, fc=C["dec_f"], ec=C["dec_e"], lw=1.3, r=0.10)
    lbl(dx+DW/2, by_j+bh_j/2+.10, nm, fs=7.0, bold=True, col=C["dec_e"])
    lbl(dx+DW/2, by_j+bh_j/2-.18, sc, fs=6.0, col="#555")
    dec_blks.append((dx, by_j, DW, bh_j))
    if j > 0:
        px, pby, pdw, pbh = dec_blks[j-1]
        arr(px+pdw, pby+pbh/2, dx, by_j+bh_j/2, col=C["dec_e"], lw=1.4, ms=9)
    dx += DW + DG

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  OUTPUT MASK
# ═══════════════════════════════════════════════════════════════════════════════
OX = dx + .15; OW = 1.2; OH = 3.0; OY = 5.0 - OH/2
rbox(OX, OY, OW, OH, fc="#EAEAEA", ec="#303030", lw=1.8, r=0.10)
nr, nc = 14, 7
cw = (OW-.12)/nc; ch = (OH-.12)/nr
for ri in range(nr):
    for ci in range(nc):
        dist = abs(ci - nc/2)
        is_w = (dist < 1.5 + .5*np.sin(ri)) or (ri > 9 and ci > 4)
        ax.add_patch(mpatches.Rectangle(
            (OX+.06+ci*cw, OY+.06+ri*ch), cw-.015, ch-.015,
            facecolor="#4DA6D8" if is_w else "#1C1C1C", edgecolor="none", zorder=4))

lx_, by_, dw_, bh_ = dec_blks[-1]
arr(lx_+dw_, by_+bh_/2, OX, OY+OH/2, col="#333", lw=1.8, ms=12)
lbl(OX+OW/2, OY-.38, "Binary Flood Mask\n(512×512)", fs=8)
lbl(OX+OW/2, OY+OH+.30, "Output", fs=10, bold=True)

# water/land mini-legend
rbox(OX+.10, OY+OH+.52, .22, .20, fc="#4DA6D8", ec="none", lw=0, r=.03, zorder=8)
lbl(OX+.38, OY+OH+.62, "Water",     fs=6.5, ha="left", col="#333", zorder=8)
rbox(OX+.10, OY+OH+.76, .22, .20, fc="#1C1C1C", ec="none", lw=0, r=.03, zorder=8)
lbl(OX+.38, OY+OH+.86, "Non-water", fs=6.5, ha="left", col="#333", zorder=8)

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SECTION HEADER BRACES
# ═══════════════════════════════════════════════════════════════════════════════
def top_brace(x1, x2, y, txt, col="#555"):
    ax.annotate("", xy=(x1, y-.06), xytext=(x2, y-.06),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.0))
    lbl((x1+x2)/2, y+.10, txt, fs=7.5, col=col)

top_brace(PE_X0, ENC_END_X, 8.96,
          "Dual Frozen Encoders  (Earth-Adapter in-block)", col="#444")
top_brace(FUSE_X0, FUSE_XE, 8.20,
          "Multi-Depth Fusion  (DepthFusion ×4)", col=C["fuse_e"])
top_brace(DEC_X0, DEC_X0+len(dec_stages)*(DW+DG)-DG, 8.20,
          "LinkNet Decoder", col=C["dec_e"])

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
items = [
    (C["dino_f"], C["dino_e"], "DINOv2-L (RGB)"),
    (C["clay_f"], C["clay_e"], "Clay v1.5 (BGRNIR, λ-cond.)"),
    (C["ea_f"],   C["ea_e"],   "Earth-Adapter (in-block FFN)"),
    (C["adac_f"], C["adac_e"], "ADAC + PPAdapter (post-hoc)"),
    (C["fi_f"],   C["fi_e"],   "Fusion sub-ops (Conv+GN)"),
    (C["gate_f"], C["gate_e"], "Learnable gate (α_a, α_b)"),
    (C["cat_f"],  C["cat_e"],  "Concatenate → 512-ch"),
    (C["dec_f"],  C["dec_e"],  "LinkNet Decoder"),
]
LY = .22; LX0 = .25; LBW = .28; LBH = .24; LSEP = 3.30
for k, (fc, ec, txt) in enumerate(items):
    lx = LX0 + k*LSEP
    rbox(lx, LY, LBW, LBH, fc=fc, ec=ec, lw=1.0, r=.04, zorder=8)
    lbl(lx+LBW+.10, LY+LBH/2, txt, fs=7, ha="left", col="#333", zorder=8)

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  SAVE
# ═══════════════════════════════════════════════════════════════════════════════
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"arch_schematic.{fmt}",
                dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}/arch_schematic.{{pdf,png}}")
