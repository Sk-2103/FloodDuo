"""
Three-panel block diagram: EarthAdapter (EA) · ADAC · PPAdapter

Each panel shows the exact forward pass from the code in:
  src/models/earth_adapter.py  (EarthAdapter / FFNWithEA)
  src/models/adapters.py       (ADAC / PPAdapter)

Saved to:
  outputs/viz/adapter_blocks.pdf/.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

OUT = Path("outputs/viz")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})

# ── Global palette ─────────────────────────────────────────────────────────
C = dict(
    # EarthAdapter
    ffn_f  ="#F4F4F4", ffn_e  ="#888888",   # frozen FFN
    dft_f  ="#D8EEFA", dft_e  ="#1A6EA8",   # DFT split
    exp_f  ="#E0F5E8", exp_e  ="#2A7A40",   # experts
    rtr_f  ="#FEF0DC", rtr_e  ="#C06820",   # router
    ea_f   ="#EDE8FC", ea_e   ="#6E33A8",   # EA outer / alpha
    # ADAC
    dw_f   ="#FFF3E0", dw_e   ="#E65100",   # dw conv branches
    gn_f   ="#FFF8E1", gn_e   ="#F9A825",   # GN/GELU
    pw_f   ="#E8EAF6", pw_e   ="#3949AB",   # pointwise conv
    # PPAdapter
    pool_f ="#E8F5E9", pool_e ="#2E7D32",   # pool branches
    cat_f  ="#EEF0FF", cat_e  ="#3333BB",   # concat
    fuse_f ="#E3F2FD", fuse_e ="#1565C0",   # fuse conv
    # shared
    gate_f ="#FCE8EC", gate_e ="#B02040",   # gamma / alpha zero-init
    add_f  ="#FFFFFF", add_e  ="#444444",   # residual ⊕
    skip_c ="#AAAAAA",
    arr    ="#333333",
    title_c="#222222",
)

W, H = 22.0, 14.0
fig, ax = plt.subplots(figsize=(W, H), facecolor="white")
ax.set_xlim(0, W); ax.set_ylim(0, H)
ax.set_facecolor("white"); ax.axis("off")

# ── Primitive helpers ──────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.2, r=0.10, z=3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=z))

def arr(x1, y1, x2, y2, col=C["arr"], lw=1.3, ms=8, style="arc3,rad=0.0", z=6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),
        arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=col, connectionstyle=style, zorder=z))

def line(x1,y1,x2,y2, col=C["skip_c"], lw=1.1, ls="-", z=5):
    ax.plot([x1,x2],[y1,y2], color=col, lw=lw, ls=ls, zorder=z)

def lbl(x, y, s, fs=7.5, ha="center", va="center", col="#222",
        bold=False, rot=0, z=7):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=col,
            fontweight="bold" if bold else "normal", rotation=rot,
            zorder=z, fontfamily="DejaVu Sans", linespacing=1.3)

def add_sym(cx, cy, r=0.22, z=6):
    ax.add_patch(plt.Circle((cx,cy), r, fc=C["add_f"], ec=C["add_e"],
                             lw=1.3, zorder=z))
    lbl(cx, cy, "+", fs=9, bold=True, col=C["add_e"], z=z+1)

def tag_frozen(x, y):
    lbl(x, y, "❄ frozen", fs=6, col="#999", ha="left")

def tag_train(x, y):
    lbl(x, y, "✓ trainable", fs=6, col=C["ea_e"], ha="left")

def gamma_box(x, y, w=1.5, h=0.52, label="× γ  (zero-init)"):
    rbox(x, y, w, h, fc=C["gate_f"], ec=C["gate_e"], lw=1.3)
    lbl(x+w/2, y+h/2, label, fs=7.5, bold=True, col=C["gate_e"])

# ──────────────────────────────────────────────────────────────────────────────
# PANEL 1 — Earth-Adapter / FFNWithEA
# ──────────────────────────────────────────────────────────────────────────────
P1_X, P1_W = 0.25, 6.90
P1_M = P1_X + P1_W/2          # 3.70
rbox(P1_X, 0.30, P1_W, 13.40, fc="#FAFAFA", ec="#CCCCCC", lw=1.5, r=0.25, z=1)
lbl(P1_M, 13.92, "(a)  FFNWithEA  —  Earth-Adapter  (in-block, all 24 layers)",
    fs=9, bold=True, col=C["ea_e"])

# input
rbox(P1_M-1.6, 12.55, 3.2, 0.58, fc=C["ea_f"], ec=C["ea_e"], lw=1.3)
lbl(P1_M, 12.84, "x  ∈  ℝ^(B, N+prefix, D)", fs=7.5)
arr(P1_M, 12.55, P1_M, 12.0, col=C["arr"])

# split into two columns
FFN_M = P1_X + 1.55     # left  (frozen FFN)
EA_M  = P1_X + 5.10     # right (EA internals)

arr(P1_M, 12.0, FFN_M, 11.35, col=C["ffn_e"], lw=1.1, ms=7, style="arc3,rad=0.22")
arr(P1_M, 12.0, EA_M,  11.35, col=C["ea_e"],  lw=1.1, ms=7, style="arc3,rad=-0.22")

# ── left: frozen FFN ───────────────────────────────────────────────────────
rbox(P1_X+0.25, 9.00, 2.60, 2.25, fc=C["ffn_f"], ec=C["ffn_e"], lw=1.2, ls="--")
lbl(FFN_M, 10.76, "FFN  (frozen)", fs=8, bold=True, col=C["ffn_e"])
lbl(FFN_M, 10.36, "Linear(D, 4D)", fs=7, col="#666")
lbl(FFN_M, 10.02, "GELU",          fs=7, col="#666")
lbl(FFN_M, 9.68, "Linear(4D, D)", fs=7, col="#666")
lbl(FFN_M, 9.24, "→ FFN(x)",       fs=7.5, col=C["ffn_e"], bold=True)
tag_frozen(P1_X+0.28, 9.14)
arr(FFN_M, 11.35, FFN_M, 11.25, col=C["ffn_e"], ms=7)
arr(FFN_M, 9.00,  FFN_M, 7.64,  col=C["ffn_e"], lw=1.2, ms=7)

# ── right: EarthAdapter ─────────────────────────────────────────────────────
# prefix strip note
lbl(EA_M, 11.56, "patch tokens only", fs=6.5, col="#888")
lbl(EA_M, 11.28, "(strip cls / prefix)", fs=6.5, col="#888")

# DFT split box
rbox(EA_M-1.42, 9.70, 2.84, 1.38, fc=C["dft_f"], ec=C["dft_e"], lw=1.3)
lbl(EA_M, 11.16, "2-D FFT", fs=7.5, bold=True, col=C["dft_e"])
DFT_MID_Y = 10.39
lbl(EA_M, DFT_MID_Y+0.22, "FFT2 → fftshift → square mask  (radius ρ=0.25)",
    fs=6.8, col=C["dft_e"])
lbl(EA_M-0.55, DFT_MID_Y-0.18, "iFFT(F·mask)", fs=6.5, col="#555")
lbl(EA_M-0.55, DFT_MID_Y-0.44, "→  LF tokens", fs=6.5, col=C["dft_e"], bold=True)
lbl(EA_M+0.55, DFT_MID_Y-0.18, "iFFT(F·~mask)", fs=6.5, col="#555")
lbl(EA_M+0.55, DFT_MID_Y-0.44, "→  HF tokens", fs=6.5, col=C["dft_e"], bold=True)
arr(EA_M, 11.28, EA_M, 11.08, col=C["dft_e"], ms=7)

# 3 experts
EXP_W, EXP_H = 1.55, 1.25
EXP_Y = 7.75
EXP_XS = [P1_X+0.38, P1_X+2.05, P1_X+3.72]  # left edges
EXP_LABS = [
    ("Spatial Expert", "x  (raw)", "#E0F5E8"),
    ("LF Expert",      "LF tokens", "#D0EDDC"),
    ("HF Expert",      "HF tokens", "#C0E5D0"),
]
exp_xms = []
for (nm, inp, fc_exp), ex in zip(EXP_LABS, EXP_XS):
    exm = ex + EXP_W/2
    exp_xms.append(exm)
    rbox(ex, EXP_Y, EXP_W, EXP_H, fc=fc_exp, ec=C["exp_e"], lw=1.2)
    lbl(exm, EXP_Y+EXP_H-0.22, nm,  fs=7, bold=True, col=C["exp_e"])
    lbl(exm, EXP_Y+EXP_H-0.54, f"in: {inp}", fs=6.3, col="#555")
    lbl(exm, EXP_Y+EXP_H-0.80, "Lin(D,b)→ReLU", fs=6.3, col="#777")
    lbl(exm, EXP_Y+EXP_H-1.03, "→Lin(b,D)", fs=6.3, col="#777")

# arrows DFT → experts
arr(EA_M-0.60, 9.70, exp_xms[1], EXP_Y+EXP_H, col=C["dft_e"], ms=7, lw=1.0)
lbl(EA_M-0.30, 9.30, "LF", fs=6.5, col=C["dft_e"])
arr(EA_M+0.60, 9.70, exp_xms[2], EXP_Y+EXP_H, col=C["dft_e"], ms=7, lw=1.0)
lbl(EA_M+0.30, 9.30, "HF", fs=6.5, col=C["dft_e"])
# x → spatial expert (from input above DFT, pass-through)
arr(EA_M, 11.28, exp_xms[0], EXP_Y+EXP_H, col="#888",
    ms=7, lw=1.0, style="arc3,rad=0.30")
lbl(P1_X+1.10, 10.85, "x", fs=7, col="#888")

# Router
RTR_W, RTR_H = 2.80, 0.90
RTR_Y = 6.30
RTR_X = EA_M - RTR_W/2
rbox(RTR_X, RTR_Y, RTR_W, RTR_H, fc=C["rtr_f"], ec=C["rtr_e"], lw=1.3)
lbl(EA_M, RTR_Y+RTR_H/2+0.18, "Router", fs=8, bold=True, col=C["rtr_e"])
lbl(EA_M, RTR_Y+RTR_H/2-0.20, "Linear(D, 3)  on  mean(x, dim=N)  →  softmax",
    fs=6.8, col="#555")
# x → router
arr(EA_M, 11.28, RTR_X-0.06, RTR_Y+RTR_H/2, col=C["rtr_e"],
    ms=7, lw=1.0, style="arc3,rad=-0.35")
lbl(P1_X+3.45, 9.50, "x.mean(N)", fs=6.3, col=C["rtr_e"])
# experts → weighted-sum
for exm in exp_xms:
    arr(exm, EXP_Y, EA_M, RTR_Y+RTR_H+0.10, col=C["exp_e"],
        ms=6, lw=0.9, style="arc3,rad=0.0")

# weighted sum circle
WS_Y = 5.52
add_sym(EA_M, WS_Y, r=0.26)
lbl(EA_M+0.40, WS_Y+0.30, "Δ = Σ  gₖ · expertₖ", fs=7, col="#333", ha="left")
arr(RTR_X+RTR_W/2, RTR_Y, EA_M, WS_Y+0.26, col=C["rtr_e"], ms=7)
lbl(EA_M+0.35, (RTR_Y+WS_Y)/2, "g₁,g₂,g₃", fs=6.5, col=C["rtr_e"], ha="left")

# × α (zero-init)
ALP_W, ALP_H = 1.70, 0.52
ALP_X = EA_M - ALP_W/2
ALP_Y = 4.60
gamma_box(ALP_X, ALP_Y, w=ALP_W, h=ALP_H, label="× α  (zero-init)")
arr(EA_M, WS_Y-0.26, EA_M, ALP_Y+ALP_H, col=C["ea_e"], ms=7)
lbl(EA_M+0.50, (WS_Y+ALP_Y)/2, "Δ", fs=7.5, col=C["ea_e"], ha="left")

# ⊕ merge (FFN + αΔ)
MERGE_Y = 3.50
add_sym(P1_M, MERGE_Y)
lbl(P1_M-0.70, MERGE_Y+0.35, "FFN(x)", fs=7, col=C["ffn_e"])
lbl(P1_M+0.62, MERGE_Y+0.35, "α · Δ", fs=7, col=C["ea_e"])
arr(FFN_M, 8.98, P1_M-0.22, MERGE_Y, col=C["ffn_e"], lw=1.2, ms=7)
arr(EA_M,  ALP_Y, P1_M+0.22, MERGE_Y, col=C["ea_e"],  lw=1.2, ms=7)

# skip around entire block (original x → ⊕)
SKP1_X = P1_X + 0.12
line(SKP1_X, 12.55, SKP1_X, MERGE_Y, col=C["skip_c"], ls="--")
line(SKP1_X, 12.55, P1_M-0.02, 12.55, col=C["skip_c"], ls="--")
arr(SKP1_X, MERGE_Y, P1_M-0.22, MERGE_Y, col=C["skip_c"], ms=7, lw=1.1)
lbl(SKP1_X-0.18, 8.0, "x", fs=7, col=C["skip_c"], rot=90)

# output
arr(P1_M, MERGE_Y-0.22, P1_M, 2.80, col=C["arr"], lw=1.4, ms=9)
rbox(P1_M-1.60, 2.00, 3.20, 0.72, fc=C["ea_f"], ec=C["ea_e"], lw=1.3)
lbl(P1_M, 2.36, "FFN(x)  +  α · Δ   ∈  ℝ^(B, N+prefix, D)", fs=7.5)
lbl(P1_M, 0.65, "b = bottleneck = 32 · ρ = 0.25 · α init = 0",
    fs=6.5, col="#888")

# trainable note
tag_train(P1_X+3.80, 4.52)

# ──────────────────────────────────────────────────────────────────────────────
# PANEL 2 — ADAC
# ──────────────────────────────────────────────────────────────────────────────
P2_X, P2_W = 7.55, 6.60
P2_M = P2_X + P2_W/2      # 10.85
rbox(P2_X, 0.30, P2_W, 13.40, fc="#FAFAFA", ec="#CCCCCC", lw=1.5, r=0.25, z=1)
lbl(P2_M, 13.92, "(b)  ADAC  —  Atrous Depth-Wise Conv Adapter  (post-hoc)",
    fs=9, bold=True, col=C["dw_e"])

# input
rbox(P2_M-1.55, 12.55, 3.10, 0.58, fc=C["dw_f"], ec=C["dw_e"], lw=1.3)
lbl(P2_M, 12.84, "x  ∈  ℝ^(B, C, H, W)", fs=7.5)
arr(P2_M, 12.55, P2_M, 12.0, col=C["arr"])

# 3 parallel DW conv branches
DW_W, DW_H = 1.48, 2.40
DW_Y = 9.20
DW_GAP = 0.25
DW_TOTAL = 3*DW_W + 2*DW_GAP
DW_X0 = P2_M - DW_TOTAL/2
dw_info = [
    ("DW-Conv\n3×3, d=1", "receptive\nfield: 3×3"),
    ("DW-Conv\n3×3, d=2", "receptive\nfield: 5×5"),
    ("DW-Conv\n3×3, d=3", "receptive\nfield: 7×7"),
]
dw_xms = []
for i, (nm, rf) in enumerate(dw_info):
    dx = DW_X0 + i*(DW_W+DW_GAP)
    dxm = dx + DW_W/2
    dw_xms.append(dxm)
    rbox(dx, DW_Y, DW_W, DW_H, fc=C["dw_f"], ec=C["dw_e"], lw=1.3)
    lbl(dxm, DW_Y+DW_H-0.38, nm,  fs=7.5, bold=True, col=C["dw_e"])
    lbl(dxm, DW_Y+DW_H-1.00, "depth-wise", fs=6.5, col="#666")
    lbl(dxm, DW_Y+DW_H-1.30, "(groups=C)", fs=6.5, col="#666")
    lbl(dxm, DW_Y+DW_H-1.65, rf, fs=6.3, col=C["dw_e"])
    # arrow from input to branch
    arr(P2_M, 12.0, dxm, DW_Y+DW_H, col=C["dw_e"], ms=7, lw=1.0,
        style=f"arc3,rad={0.25*(i-1):.2f}")

# sum circle
SUM_Y = 8.50
add_sym(P2_M, SUM_Y)
lbl(P2_M+0.40, SUM_Y+0.30, "Σ  (element-wise sum)", fs=7, col="#333", ha="left")
for dxm in dw_xms:
    arr(dxm, DW_Y, P2_M, SUM_Y+0.23, col=C["dw_e"], ms=6, lw=0.9)

# GN → GELU
GN_W, GN_H = 3.20, 0.72
GN_Y = 7.30
rbox(P2_M-GN_W/2, GN_Y, GN_W, GN_H, fc=C["gn_f"], ec=C["gn_e"], lw=1.2)
lbl(P2_M, GN_Y+GN_H/2, "GroupNorm(1, C)   →   GELU", fs=7.5)
arr(P2_M, SUM_Y-0.23, P2_M, GN_Y+GN_H, col=C["arr"], ms=7)

# pointwise conv 1×1
PW_W, PW_H = 3.20, 0.72
PW_Y = 5.90
rbox(P2_M-PW_W/2, PW_Y, PW_W, PW_H, fc=C["pw_f"], ec=C["pw_e"], lw=1.2)
lbl(P2_M, PW_Y+PW_H/2, "Conv2d(C, C, 1×1)   →   y", fs=7.5, col=C["pw_e"])
arr(P2_M, GN_Y, P2_M, PW_Y+PW_H, col=C["arr"], ms=7)

# × γ
GM_W, GM_H = 1.80, 0.52
GM_Y = 4.80
gamma_box(P2_M-GM_W/2, GM_Y, w=GM_W, h=GM_H)
arr(P2_M, PW_Y, P2_M, GM_Y+GM_H, col=C["arr"], ms=7)

# skip + residual add
SKP2_X = P2_X + 0.18
ADD2_Y  = 3.70
add_sym(P2_M, ADD2_Y)
line(SKP2_X, 12.55, SKP2_X, ADD2_Y, col=C["skip_c"], ls="--")
line(SKP2_X, 12.55, P2_M, 12.55, col=C["skip_c"], ls="--")
arr(SKP2_X, ADD2_Y, P2_M-0.22, ADD2_Y, col=C["skip_c"], ms=7, lw=1.1)
lbl(SKP2_X-0.18, 8.0, "x", fs=7, col=C["skip_c"], rot=90)
arr(P2_M, GM_Y, P2_M, ADD2_Y+0.22, col=C["arr"], ms=7)
lbl(P2_M+0.42, (GM_Y+ADD2_Y)/2, "γ·y", fs=7.5, col=C["gate_e"], ha="left")

# output
arr(P2_M, ADD2_Y-0.22, P2_M, 2.80, col=C["arr"], lw=1.4, ms=9)
rbox(P2_M-1.55, 2.00, 3.10, 0.72, fc=C["dw_f"], ec=C["dw_e"], lw=1.3)
lbl(P2_M, 2.36, "x  +  γ·y   ∈  ℝ^(B, C, H, W)", fs=7.5)
lbl(P2_M, 0.65, "dilations: (1, 2, 3)  ·  γ init = 0",
    fs=6.5, col="#888")
tag_train(P2_M+0.50, GM_Y-0.06)

# ──────────────────────────────────────────────────────────────────────────────
# PANEL 3 — PPAdapter
# ──────────────────────────────────────────────────────────────────────────────
P3_X, P3_W = 14.65, 7.10
P3_M = P3_X + P3_W/2       # 18.20
rbox(P3_X, 0.30, P3_W, 13.40, fc="#FAFAFA", ec="#CCCCCC", lw=1.5, r=0.25, z=1)
lbl(P3_M, 13.92, "(c)  PPAdapter  —  Pyramid Pooling Adapter  (post-hoc)",
    fs=9, bold=True, col=C["pool_e"])

# input
rbox(P3_M-1.55, 12.55, 3.10, 0.58, fc=C["pool_f"], ec=C["pool_e"], lw=1.3)
lbl(P3_M, 12.84, "x  ∈  ℝ^(B, C, H, W)", fs=7.5)
arr(P3_M, 12.55, P3_M, 12.0, col=C["arr"])

# 4 pool branches
POOL_W, POOL_H = 1.38, 2.60
POOL_Y = 8.95
POOL_GAP = 0.18
POOL_TOTAL = 4*POOL_W + 3*POOL_GAP
POOL_X0 = P3_M - POOL_TOTAL/2
pool_scales = [("1×1","1×1"), ("2×2","2×2"), ("3×3","3×3"), ("6×6","6×6")]
pool_xms = []
for i, (ps, label) in enumerate(pool_scales):
    px = POOL_X0 + i*(POOL_W+POOL_GAP)
    pxm = px + POOL_W/2
    pool_xms.append(pxm)
    rbox(px, POOL_Y, POOL_W, POOL_H, fc=C["pool_f"], ec=C["pool_e"], lw=1.2)
    lbl(pxm, POOL_Y+POOL_H-0.26, f"AvgPool",    fs=6.8, bold=True, col=C["pool_e"])
    lbl(pxm, POOL_Y+POOL_H-0.52, f"({ps})",     fs=7.5, bold=True, col=C["pool_e"])
    lbl(pxm, POOL_Y+POOL_H-0.84, "Conv(C→C/4)", fs=6.3, col="#555")
    lbl(pxm, POOL_Y+POOL_H-1.08, "GN  +  GELU", fs=6.3, col="#555")
    lbl(pxm, POOL_Y+POOL_H-1.40, "bilinear",    fs=6.3, col="#777")
    lbl(pxm, POOL_Y+POOL_H-1.65, "↑ H×W",       fs=6.3, col=C["pool_e"])
    lbl(pxm, POOL_Y+POOL_H-1.95, "→ cᵢ",        fs=6.8, bold=True, col=C["pool_e"])
    rad = 0.30*(i - 1.5)
    arr(P3_M, 12.0, pxm, POOL_Y+POOL_H, col=C["pool_e"],
        ms=7, lw=1.0, style=f"arc3,rad={rad:.2f}")

# identity pass-through of x (for concat)
XPASS_X = P3_X + 0.35
line(XPASS_X, 12.55, XPASS_X, 7.80, col=C["skip_c"], lw=1.1, ls="--")
lbl(XPASS_X-0.18, 10.0, "x", fs=7, col=C["skip_c"], rot=90)

# concat box
CAT_W, CAT_H = 4.20, 0.78
CAT_Y = 7.80
CAT_X = P3_M - CAT_W/2
rbox(CAT_X, CAT_Y, CAT_W, CAT_H, fc=C["cat_f"], ec=C["cat_e"], lw=1.3)
lbl(P3_M, CAT_Y+CAT_H/2+0.12, "Concatenate  [x, c₁, c₂, c₃, c₄]", fs=7.5,
    bold=True, col=C["cat_e"])
lbl(P3_M, CAT_Y+CAT_H/2-0.20, "→  (B,  C + 4·(C/4),  H, W)  =  (B, 2C, H, W)",
    fs=6.8, col="#555")
for pxm in pool_xms:
    arr(pxm, POOL_Y, P3_M, CAT_Y+CAT_H, col=C["pool_e"], ms=6, lw=0.9)
arr(XPASS_X, 7.80, CAT_X-0.04, CAT_Y+CAT_H/2, col=C["skip_c"], ms=7, lw=1.1)

# fuse conv 1×1
FC_W, FC_H = 3.60, 0.72
FC_Y = 6.50
rbox(P3_M-FC_W/2, FC_Y, FC_W, FC_H, fc=C["fuse_f"], ec=C["fuse_e"], lw=1.3)
lbl(P3_M, FC_Y+FC_H/2, "Conv2d(2C, C, 1×1)   →   z", fs=7.5, col=C["fuse_e"])
arr(P3_M, CAT_Y, P3_M, FC_Y+FC_H, col=C["arr"], ms=7)

# × γ
PGM_W, PGM_H = 1.80, 0.52
PGM_Y = 5.45
gamma_box(P3_M-PGM_W/2, PGM_Y, w=PGM_W, h=PGM_H)
arr(P3_M, FC_Y, P3_M, PGM_Y+PGM_H, col=C["arr"], ms=7)

# residual add
SKP3_X = P3_X + 0.18
ADD3_Y  = 4.30
add_sym(P3_M, ADD3_Y)
line(SKP3_X, 12.55, SKP3_X, ADD3_Y, col=C["skip_c"], ls="--")
arr(SKP3_X, ADD3_Y, P3_M-0.22, ADD3_Y, col=C["skip_c"], ms=7, lw=1.1)
lbl(SKP3_X-0.18, 8.0, "x", fs=7, col=C["skip_c"], rot=90)
arr(P3_M, PGM_Y, P3_M, ADD3_Y+0.22, col=C["arr"], ms=7)
lbl(P3_M+0.42, (PGM_Y+ADD3_Y)/2, "γ·z", fs=7.5, col=C["gate_e"], ha="left")

# output
arr(P3_M, ADD3_Y-0.22, P3_M, 2.80, col=C["arr"], lw=1.4, ms=9)
rbox(P3_M-1.55, 2.00, 3.10, 0.72, fc=C["pool_f"], ec=C["pool_e"], lw=1.3)
lbl(P3_M, 2.36, "x  +  γ·z   ∈  ℝ^(B, C, H, W)", fs=7.5)
lbl(P3_M, 0.65, "pool scales: (1, 2, 3, 6)  ·  C/4 per branch  ·  γ init = 0",
    fs=6.5, col="#888")
tag_train(P3_M+0.50, PGM_Y-0.06)

# ── shared note about zero-init γ/α ────────────────────────────────────────
lbl(W/2, 0.20,
    "Zero-init  γ / α:  adapter output = 0 at initialisation  →  "
    "training starts from frozen backbone performance",
    fs=7, col="#666")

# ── save ───────────────────────────────────────────────────────────────────
for fmt in ["pdf", "png"]:
    fig.savefig(OUT / f"adapter_blocks.{fmt}",
                dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT}/adapter_blocks.{{pdf,png}}")
