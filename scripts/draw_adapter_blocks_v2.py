"""
Clean adapter block diagrams — v2.
Three subplots, straight arrows only, generous spacing.
(a) FFNWithEA  (b) ADAC  (c) PPAdapter
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

OUT = Path("outputs/viz")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

FW, FH = 21, 14
fig = plt.figure(figsize=(FW, FH), facecolor="white")
# axes: [left, bottom, width, height] in figure fraction
ax1 = fig.add_axes([0.01, 0.02, 0.40, 0.96])   # EA — widest
ax2 = fig.add_axes([0.425, 0.02, 0.265, 0.96])  # ADAC
ax3 = fig.add_axes([0.705, 0.02, 0.285, 0.96])  # PPAdapter

for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis("off")
    ax.set_facecolor("white")

# ── shared helpers ────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, fc, ec, lw=1.5, r=0.18, ls="-", z=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))

def txt(ax, x, y, s, fs=9.5, ha="center", va="center",
        bold=False, col="#222", z=5, lsp=1.35):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=col,
            fontweight="bold" if bold else "normal",
            zorder=z, linespacing=lsp)

def darr(ax, x, y1, y2, c="#333", lw=1.4, ms=11):
    """Straight downward arrow."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
            mutation_scale=ms, shrinkA=0, shrinkB=0,
            connectionstyle="arc3,rad=0"))

def rarr(ax, x1, x2, y, c="#333", lw=1.4, ms=11):
    """Straight rightward arrow."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
            mutation_scale=ms, shrinkA=0, shrinkB=0,
            connectionstyle="arc3,rad=0"))

def seg(ax, x1, y1, x2, y2, c="#444", lw=1.3, ls="-", z=4):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, ls=ls, zorder=z)

def addsym(ax, cx, cy, r=0.28, z=6):
    ax.add_patch(plt.Circle((cx, cy), r, fc="white", ec="#444",
                             lw=1.5, zorder=z))
    ax.text(cx, cy, "+", fontsize=12, ha="center", va="center",
            fontweight="bold", color="#333", zorder=z + 1)

def fork(ax, x_from, y_from, branches_x, bar_y, c="#444", lw=1.3):
    """T-fork: line from x_from down to bar_y, horizontal bar, no arrows."""
    seg(ax, x_from, y_from, x_from, bar_y, c=c, lw=lw)
    seg(ax, branches_x[0], bar_y, branches_x[-1], bar_y, c=c, lw=lw)

def join_bar(ax, branches_x, bar_y, c="#444", lw=1.3):
    seg(ax, branches_x[0], bar_y, branches_x[-1], bar_y, c=c, lw=lw)

def skip_lshape(ax, x_box_left, y_top, y_add, skip_x, add_cx=5.0,
                c="#AAA", lw=1.1):
    """Dashed skip: left from box edge, down, right into adder at add_cx."""
    seg(ax, x_box_left, y_top, skip_x, y_top, c=c, lw=lw, ls="--")
    seg(ax, skip_x, y_top, skip_x, y_add, c=c, lw=lw, ls="--")
    rarr(ax, skip_x, add_cx - 0.30, y_add, c=c, lw=lw, ms=9)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — FFNWithEA  (Earth-Adapter, in-block)
# ─────────────────────────────────────────────────────────────────────────────
ax = ax1

# Colors
EA_F, EA_E     = "#EDE8FC", "#6A33C2"
FFN_F, FFN_E   = "#F5F5F5", "#888888"
DFT_F, DFT_E   = "#D8EEFA", "#1A6EA8"
EXP_F, EXP_E   = "#E0F5E8", "#2A7A40"
RTR_F, RTR_E   = "#FEF0DC", "#B06800"
ALF_F, ALF_E   = "#FCE8EC", "#C62828"
IO_F, IO_E     = "#EDE8FC", "#6A33C2"

txt(ax, 5, 14.75, "(a)  FFNWithEA  —  Earth-Adapter", fs=12, bold=True, col=EA_E)
txt(ax, 5, 14.25, "Applied in-block at all 24 transformer layers (DINOv2 & Clay)",
    fs=8.5, col="#555")

# Input
box(ax, 5, 13.55, 7.0, 0.65, fc=IO_F, ec=IO_E)
txt(ax, 5, 13.55, "x  ∈  ℝ^(B, N, D)   [patch tokens after LN]", fs=10)

# Stem + main fork bar
FORK_Y = 12.85
seg(ax, 5, 13.22, 5, FORK_Y, c="#444")
seg(ax, 2.2, FORK_Y, 7.8, FORK_Y, c="#444")

# ── Left branch: frozen FFN ────────────────────────────────────────────────
FFN_CX, FFN_CY, FFN_W, FFN_H = 2.2, 11.0, 4.0, 3.2
darr(ax, FFN_CX, FORK_Y, FFN_CY + FFN_H/2, c=FFN_E, ms=10)
box(ax, FFN_CX, FFN_CY, FFN_W, FFN_H, fc=FFN_F, ec=FFN_E, lw=1.4, ls="--")
txt(ax, FFN_CX, 12.05, "FFN  (❄ frozen)", fs=9.5, bold=True, col=FFN_E)
txt(ax, FFN_CX, 11.60, "Linear(D, 4D)", fs=9,   col="#555")
txt(ax, FFN_CX, 11.22, "→  GELU",         fs=9,   col="#555")
txt(ax, FFN_CX, 10.84, "→  Linear(4D, D)", fs=9,  col="#555")
txt(ax, FFN_CX, 10.38, "=  FFN(x)",        fs=10,  bold=True, col=FFN_E)
FFN_BOT = FFN_CY - FFN_H/2   # = 9.4

# ── Right branch: EarthAdapter ────────────────────────────────────────────
EA_CX = 7.5

# DFT Split box
DFT_CY, DFT_H = 12.25, 0.80
darr(ax, EA_CX, FORK_Y, DFT_CY + DFT_H/2, c=DFT_E, ms=10)
box(ax, EA_CX, DFT_CY, 4.5, DFT_H, fc=DFT_F, ec=DFT_E)
txt(ax, EA_CX, DFT_CY + 0.12, "2-D FFT  →  fftshift  →  square mask  (ρ=0.25)",
    fs=9, bold=True, col=DFT_E)
txt(ax, EA_CX, DFT_CY - 0.20, "→  LF tokens     →  HF tokens",
    fs=8.5, col="#555")
DFT_BOT = DFT_CY - DFT_H/2   # = 11.85

# Expert fork
EXP_FORK_Y = DFT_BOT - 0.15   # 11.70
seg(ax, EA_CX, DFT_BOT, EA_CX, EXP_FORK_Y, c="#444")
EXP_XS = [5.3, 7.0, 8.7]
seg(ax, EXP_XS[0], EXP_FORK_Y, EXP_XS[-1], EXP_FORK_Y, c="#444")

# 3 expert boxes
EXP_CY, EXP_H, EXP_W = 10.45, 1.95, 1.55
EXP_TOP = EXP_CY + EXP_H/2   # 11.42
EXP_BOT = EXP_CY - EXP_H/2   # 9.47
EXP_LABELS = [
    ("Spatial", "Expert", "input: x", "(all tokens,", "unfiltered)"),
    ("LF", "Expert", "input: LF", "(low-freq.", "components)"),
    ("HF", "Expert", "input: HF", "(high-freq.", "components)"),
]
EXP_COLORS = ["#E8F8EC", "#D5F0DD", "#C3E8CD"]
for ex, (t1, t2, t3, t4, t5), fc in zip(EXP_XS, EXP_LABELS, EXP_COLORS):
    darr(ax, ex, EXP_FORK_Y, EXP_TOP, c=EXP_E, ms=9)
    box(ax, ex, EXP_CY, EXP_W, EXP_H, fc=fc, ec=EXP_E, lw=1.3)
    txt(ax, ex, EXP_CY + 0.68, t1, fs=9, bold=True, col=EXP_E)
    txt(ax, ex, EXP_CY + 0.38, t2, fs=9, bold=True, col=EXP_E)
    txt(ax, ex, EXP_CY + 0.02, t3, fs=7.5, col="#555")
    txt(ax, ex, EXP_CY - 0.26, t4, fs=7,   col="#777")
    txt(ax, ex, EXP_CY - 0.50, t5, fs=7,   col="#777")
    txt(ax, ex, EXP_CY - 0.82, "Lin(1024→32)\nReLU\nLin(32→1024)", fs=7.5,
        col="#555", lsp=1.3)

# Expert join bar → Router
EXP_JOIN_Y = EXP_BOT - 0.28   # 9.19
for ex in EXP_XS:
    seg(ax, ex, EXP_BOT, ex, EXP_JOIN_Y, c="#444")
join_bar(ax, EXP_XS, EXP_JOIN_Y, c="#444")
darr(ax, EA_CX, EXP_JOIN_Y, 8.78, c="#444", ms=10)  # to Router top

# Router
RTR_CY, RTR_H = 8.25, 0.90
box(ax, EA_CX, RTR_CY, 4.8, RTR_H, fc=RTR_F, ec=RTR_E)
txt(ax, EA_CX, RTR_CY + 0.22, "Router  —  which expert matters most?",
    fs=9.5, bold=True, col=RTR_E)
txt(ax, EA_CX, RTR_CY - 0.10,
    "mean over N tokens  →  Linear(1024→3)  →  softmax",
    fs=8.5, col="#444")
txt(ax, EA_CX, RTR_CY - 0.36,
    "→  [g₁, g₂, g₃]   (3 importance weights,  g₁+g₂+g₃ = 1)",
    fs=8, col=RTR_E)
RTR_BOT = RTR_CY - RTR_H/2
darr(ax, EA_CX, RTR_BOT, 7.28, c="#444", ms=10)

# Weighted sum
WS_CY, WS_H = 6.90, 0.80
box(ax, EA_CX, WS_CY, 4.8, WS_H, fc=DFT_F, ec=DFT_E, lw=1.3)
txt(ax, EA_CX, WS_CY + 0.20,
    "Gated sum:  Δ = g₁·Spatial(x) + g₂·LF + g₃·HF",
    fs=9.5, bold=True)
txt(ax, EA_CX, WS_CY - 0.18,
    "Δ  ∈  ℝ^(B, N, 1024)   (same shape as x)", fs=8, col="#555")
WS_BOT = WS_CY - WS_H/2
darr(ax, EA_CX, WS_BOT, 6.43, c="#444", ms=10)

# × alpha (zero-init)
ALP_CY, ALP_H = 6.10, 0.60
box(ax, EA_CX, ALP_CY, 3.0, ALP_H, fc=ALF_F, ec=ALF_E, lw=2.0)
txt(ax, EA_CX, ALP_CY, "× α   (zero-initialised,  trainable)", fs=9.5,
    bold=True, col=ALF_E)
ALP_BOT = ALP_CY - ALP_H/2   # 5.80

# ── Merge ⊕ ───────────────────────────────────────────────────────────────
MERGE_Y = 4.90
# FFN arm: line down from FFN_BOT to MERGE_Y, then right to ⊕
seg(ax, FFN_CX, FFN_BOT, FFN_CX, MERGE_Y, c="#888", lw=1.2)
txt(ax, FFN_CX - 0.42, (FFN_BOT + MERGE_Y)/2, "FFN(x)", fs=8.5,
    col=FFN_E, ha="right", va="center")
rarr(ax, FFN_CX, 5.0 - 0.28, MERGE_Y, c="#888", ms=10)

# EA arm: line down from ALP_BOT to MERGE_Y, then left to ⊕
seg(ax, EA_CX, ALP_BOT, EA_CX, MERGE_Y, c="#888", lw=1.2)
txt(ax, EA_CX + 0.42, (ALP_BOT + MERGE_Y)/2, "α·Δ", fs=8.5,
    col=ALF_E, ha="left", va="center")
rarr(ax, EA_CX, 5.0 + 0.28, MERGE_Y, c="#888", ms=10)  # left arrow

addsym(ax, 5.0, MERGE_Y)
darr(ax, 5.0, MERGE_Y - 0.28, 4.30, c="#444", ms=11)

# Output
box(ax, 5, 3.90, 7.0, 0.65, fc=IO_F, ec=IO_E)
txt(ax, 5, 3.90, "FFN(x)  +  α · Δ   ∈  ℝ^(B, N, 1024)", fs=10.5)

txt(ax, 5, 3.10, "bottleneck = 32  ·  FFT radius ρ=0.25  ·  α init=0 (grows during training)",
    fs=8.5, col="#666")
txt(ax, 5, 2.65, "N tokens: 4096 for Clay (64×64 grid)  /  1024 for DINOv2 (32×32 grid)",
    fs=8.5, col="#666")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — ADAC  (Atrous Depth-Wise Conv Adapter)
# ─────────────────────────────────────────────────────────────────────────────
ax = ax2

IN_F, IN_E   = "#FFF8F0", "#E65100"
DW_F, DW_E   = "#FFF3E0", "#E65100"
GN_F, GN_E   = "#FFFDE7", "#F57F17"
PW_F, PW_E   = "#EDE7F6", "#4527A0"
GM_F, GM_E   = "#FCE4EC", "#C62828"

txt(ax, 5, 14.75, "(b)  ADAC", fs=12, bold=True, col=DW_E)
txt(ax, 5, 14.25, "Atrous Depth-wise Conv Adapter  (post-hoc)", fs=8.5, col="#555")

# Input
box(ax, 5, 13.50, 9.0, 0.85, fc=IN_F, ec=IN_E)
txt(ax, 5, 13.68, "x  ∈  ℝ^(B, 1024, H, W)", fs=10.5)
txt(ax, 5, 13.30, "C = 1024  (ViT-L feature dim)   H×W: 32×32 (DINOv2) / 64×64 (Clay)",
    fs=8, col="#666")

# Fork → 3 branches
FORK_Y2 = 12.80
seg(ax, 5, 13.17, 5, FORK_Y2)
BR = [2.0, 5.0, 8.0]
seg(ax, BR[0], FORK_Y2, BR[-1], FORK_Y2)

DW_CY, DW_H, DW_W = 11.30, 1.90, 2.55
DW_TOP = DW_CY + DW_H/2   # 12.25
DW_BOT = DW_CY - DW_H/2   # 10.35
DW_NAMES = [
    "DW-Conv\n3×3, d = 1\n(field: 3×3)",
    "DW-Conv\n3×3, d = 2\n(field: 5×5)",
    "DW-Conv\n3×3, d = 3\n(field: 7×7)",
]
for bx, nm in zip(BR, DW_NAMES):
    darr(ax, bx, FORK_Y2, DW_TOP, c=DW_E, ms=9)
    box(ax, bx, DW_CY, DW_W, DW_H, fc=DW_F, ec=DW_E, lw=1.4)
    txt(ax, bx, DW_CY, nm, fs=9.5, col="#333", lsp=1.5)

# Join bar + sum label
JOIN_Y2 = DW_BOT - 0.30  # 10.05
for bx in BR:
    seg(ax, bx, DW_BOT, bx, JOIN_Y2)
join_bar(ax, BR, JOIN_Y2)

darr(ax, 5, JOIN_Y2, 9.68, ms=10)
addsym(ax, 5, 9.40)
txt(ax, 5.8, 9.40, "sum", fs=8.5, col="#555", ha="left")
darr(ax, 5, 9.12, 8.78, ms=10)

# GN + GELU
GN_CY, GN_H = 8.35, 0.65
box(ax, 5, GN_CY, 7.5, GN_H, fc=GN_F, ec=GN_E)
txt(ax, 5, GN_CY, "GroupNorm(1, 1024)  (GELU)", fs=10)
darr(ax, 5, GN_CY - GN_H/2, 7.63, ms=10)

# Pointwise Conv 1×1
PW_CY, PW_H = 7.20, 0.65
box(ax, 5, PW_CY, 7.5, PW_H, fc=PW_F, ec=PW_E)
txt(ax, 5, PW_CY, "Conv2d(C → C,  1×1)   →   y", fs=10, col=PW_E)
darr(ax, 5, PW_CY - PW_H/2, 6.48, ms=10)

# × γ
GM_CY, GM_H = 6.05, 0.65
box(ax, 5, GM_CY, 5.0, GM_H, fc=GM_F, ec=GM_E, lw=2.0)
txt(ax, 5, GM_CY, "× γ   (zero-initialised)", fs=10, bold=True, col=GM_E)
darr(ax, 5, GM_CY - GM_H/2, 5.28, ms=10)

# Residual add
addsym(ax, 5, 5.0)
darr(ax, 5, 4.72, 4.42, ms=10)

# Skip connection  (add_cx=5 matches addsym at (5, 5.0))
skip_lshape(ax, x_box_left=5 - 9.0/2, y_top=13.08,
            y_add=5.0, skip_x=0.35, add_cx=5.0, c="#AAA")
txt(ax, 0.22, 9.0, "x", fs=10, col="#AAA")

# Output
box(ax, 5, 4.05, 7.5, 0.65, fc=IN_F, ec=IN_E)
txt(ax, 5, 4.05, "x  +  γ · y   ∈  ℝ^(B, C, H, W)", fs=10.5)

txt(ax, 5, 3.20, "depth-wise: 1 filter per channel (groups=1024)", fs=8.5, col="#666")
txt(ax, 5, 2.78, "dilation 1→ 2→ 3:  receptive field 3×3 → 5×5 → 7×7  (no downsampling)",
    fs=8.5, col="#666")
txt(ax, 5, 2.36, "γ init = 0  →  adapter invisible at start, weight grows during training",
    fs=8.5, col="#666")

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — PPAdapter  (Pyramid Pooling Adapter)
# ─────────────────────────────────────────────────────────────────────────────
ax = ax3

PP_F, PP_E   = "#E8F5E9", "#2E7D32"
CA_F, CA_E   = "#EEF0FF", "#3333BB"
FS_F, FS_E   = "#E3F2FD", "#1565C0"
GM2_F, GM2_E = "#FCE4EC", "#C62828"
IN3_F, IN3_E = "#E8F5E9", "#2E7D32"

txt(ax, 5, 14.75, "(c)  PPAdapter", fs=12, bold=True, col=PP_E)
txt(ax, 5, 14.25, "Pyramid Pooling Adapter  (post-hoc)", fs=8.5, col="#555")

# Input
box(ax, 5, 13.50, 9.0, 0.85, fc=IN3_F, ec=IN3_E)
txt(ax, 5, 13.68, "x  ∈  ℝ^(B, 1024, H, W)", fs=10.5)
txt(ax, 5, 13.30, "C = 1024   H×W: 32×32 (DINOv2) / 64×64 (Clay)",
    fs=8, col="#666")

# Fork → 4 branches
FORK_Y3 = 12.80
seg(ax, 5, 13.07, 5, FORK_Y3)
P4 = [1.5, 3.7, 6.3, 8.5]   # branch centers
seg(ax, P4[0], FORK_Y3, P4[-1], FORK_Y3)

PP_CY, PP_H, PP_W = 10.85, 3.0, 1.85
PP_TOP = PP_CY + PP_H/2   # 12.35
PP_BOT = PP_CY - PP_H/2   # 9.35
SCALES = ["1×1", "2×2", "3×3", "6×6"]
for px, sc in zip(P4, SCALES):
    darr(ax, px, FORK_Y3, PP_TOP, c=PP_E, ms=9)
    box(ax, px, PP_CY, PP_W, PP_H, fc=PP_F, ec=PP_E, lw=1.3)
    txt(ax, px, PP_CY + 0.85, f"AvgPool", fs=9, bold=True, col=PP_E)
    txt(ax, px, PP_CY + 0.48, f"({sc})",  fs=10, bold=True, col=PP_E)
    txt(ax, px, PP_CY + 0.04, "Conv(1024→256)", fs=8.5, col="#444")
    txt(ax, px, PP_CY - 0.32, "GN + GELU",   fs=8.5, col="#444")
    txt(ax, px, PP_CY - 0.68, "↑ upsample",  fs=8.5, col=PP_E)
    txt(ax, px, PP_CY - 1.05, "to H×W",      fs=8.5, col=PP_E)
    txt(ax, px, PP_CY - 1.35, f"→ c{SCALES.index(sc)+1}", fs=9,
        bold=True, col=PP_E)

# Join bar
JOIN_Y3 = PP_BOT - 0.30   # 9.05
for px in P4:
    seg(ax, px, PP_BOT, px, JOIN_Y3)
join_bar(ax, P4, JOIN_Y3)
darr(ax, 5, JOIN_Y3, 8.68, ms=10)

# Cat box
CAT_CY, CAT_H = 8.28, 0.70
box(ax, 5, CAT_CY, 8.8, CAT_H, fc=CA_F, ec=CA_E)
txt(ax, 5, CAT_CY + 0.16, "Cat  [x, c₁, c₂, c₃, c₄]", fs=9.5,
    bold=True, col=CA_E)
txt(ax, 5, CAT_CY - 0.18, "(B, 1024 + 4×256, H, W)  =  (B, 2048, H, W)",
    fs=8.5, col="#444")
darr(ax, 5, CAT_CY - CAT_H/2, 7.43, ms=10)

# Fuse Conv
FS_CY, FS_H = 7.05, 0.65
box(ax, 5, FS_CY, 8.8, FS_H, fc=FS_F, ec=FS_E)
txt(ax, 5, FS_CY, "Conv2d(2048 → 1024,  1×1)   →   z", fs=10, col=FS_E)
darr(ax, 5, FS_CY - FS_H/2, 6.28, ms=10)

# × γ
GM2_CY, GM2_H = 5.85, 0.65
box(ax, 5, GM2_CY, 5.0, GM2_H, fc=GM2_F, ec=GM2_E, lw=2.0)
txt(ax, 5, GM2_CY, "× γ   (zero-initialised)", fs=10, bold=True, col=GM2_E)
darr(ax, 5, GM2_CY - GM2_H/2, 5.08, ms=10)

# Residual add
addsym(ax, 5, 4.80)
darr(ax, 5, 4.52, 4.22, ms=10)

# Skip from input  (add_cx=5 matches addsym at (5, 4.80))
skip_lshape(ax, x_box_left=5 - 9.0/2, y_top=13.07,
            y_add=4.80, skip_x=0.25, add_cx=5.0, c="#AAA")
txt(ax, 0.12, 9.0, "x", fs=10, col="#AAA")

# Output
box(ax, 5, 3.85, 8.5, 0.65, fc=IN3_F, ec=IN3_E)
txt(ax, 5, 3.85, "x  +  γ · z   ∈  ℝ^(B, 1024, H, W)", fs=10.5)

txt(ax, 5, 3.00, "pool scales (1,2,3,6) capture context from pixel to scene level",
    fs=8.5, col="#666")
txt(ax, 5, 2.58, "bottleneck 256 = C/4  saves compute;  γ init = 0  →  starts as identity",
    fs=8.5, col="#666")

# ── save ──────────────────────────────────────────────────────────────────────
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"adapter_blocks_v2.{fmt}",
                dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}/adapter_blocks_v2.{{pdf,png}}")
