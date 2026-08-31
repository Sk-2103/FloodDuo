#!/usr/bin/env python3
"""FloodDuo (arch6) full architecture schematic — publication standard.
Main pipeline + detail panels: ViT block, FFN+Earth-Adapter (EA), ADAC, PPA,
and the NEW disagreement-gated fusion. Vector PDF + high-dpi PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_DINO="#cfe0f3"; C_CLAY="#d9ead3"; C_ADAP="#fde9c8"; C_FUSE="#f6d7c4"
C_DEC="#e6d5f0"; C_AUX="#fff2b2"; C_D="#f7c5c5"; C_EA="#fde9c8"; EDGE="#3a3a3a"
plt.rcParams.update({"font.size":9,"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(15.5,16)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")

def box(x,y,w,h,fc,t="",fs=9,bold=False,ec=EDGE,lw=1.1,r=2.0,fc2=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.02,rounding_size={r}",
                 fc=fc,ec=ec,lw=lw,mutation_aspect=1))
    if t: ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,
                  fontweight=("bold" if bold else "normal"),zorder=5)
def ar(x1,y1,x2,y2,lw=1.5,color=EDGE,style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=13,
                 lw=lw,color=color,shrinkA=1,shrinkB=1,zorder=4))
def lbl(x,y,t,fs=8,c="#222",it=False,bold=False):
    ax.text(x,y,t,ha="center",va="center",fontsize=fs,color=c,
            style=("italic" if it else "normal"),fontweight=("bold" if bold else "normal"),zorder=6)
def panel(x,y,w,h,title):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3",fc="#fcfcfc",ec="#9a9a9a",lw=1.0,ls="--"))
    ax.text(x+1.5,y+h-1.6,title,fontsize=10.5,fontweight="bold")

# ================= LEGEND =================
leg=[("DINOv3 branch",C_DINO),("Clay branch",C_CLAY),("Adapters EA/ADAC/PPA",C_ADAP),
     ("Aux head (new)",C_AUX),("Disagreement D (new)",C_D),("Gated fusion (new)",C_FUSE),("LinkNet decoder",C_DEC)]
xx=6
for t,c in leg:
    ax.add_patch(FancyBboxPatch((xx,96.3),1.4,1.1,boxstyle="round,pad=0.02",fc=c,ec=EDGE,lw=0.8))
    ax.text(xx+1.8,96.85,t,ha="left",va="center",fontsize=7.3); xx+=1.8+len(t)*0.62+1.4
ax.add_patch(plt.Rectangle((6,94.4),1.3,0.9,fc="#dbe9ff",ec="#2456a6",lw=1.4)); ax.text(7.6,94.85,"Frozen encoders",fontsize=7.3,va="center")
ax.add_patch(plt.Rectangle((26,94.4),1.3,0.9,fc="#ffe2c2",ec="#d2691e",lw=1.4,hatch="///")); ax.text(27.6,94.85,"Learnable (adapters · aux heads · fusion · decoder)",fontsize=7.3,va="center")

# ================= MAIN PIPELINE (y 80-92) =================
TOP,BOT=86,80
box(2,82.5,9,7,"#ededed"); lbl(6.5,88,"Input",10); lbl(6.5,86.4,"PlanetScope",8); lbl(6.5,84.9,"512×512×4",8); lbl(6.5,83.4,"B,G,R,NIR",7,c="#555")
ar(11,86,15,TOP+4); lbl(13,90.3,"RGB",8,c="#2456a6"); ar(11,85,15,BOT+1.4); lbl(13,78.7,"B,G,R,NIR",8,c="#3a7d34")
box(15,TOP,30,6.5,C_DINO); lbl(30,TOP+4.7,"DINOv3-L  (frozen, ViT-L/16)",9.5); lbl(30,TOP+2.9,"RGB · patch 16 · blocks 6/12/18/24",8); lbl(30,TOP+1.2,"in-block EA  ·  post-hoc ADAC + PPA",7.4,c="#a85b00")
lbl(46.6,TOP+5,"4×(1024,\n32×32)",6.8,c="#2456a6")
box(15,BOT,30,6.5,C_CLAY); lbl(30,BOT+4.7,"Clay v1.5-L  (frozen, ViT-L/8)",9.5); lbl(30,BOT+2.9,"B,G,R,NIR · patch 8 · GSD-aware · 6/12/18/24",7.4); lbl(30,BOT+1.2,"in-block EA  ·  post-hoc ADAC + PPA",7.4,c="#a85b00")
lbl(46.6,BOT+1.3,"4×(1024,\n64×64)",6.8,c="#3a7d34")
box(49,TOP+1.2,7,3.6,C_AUX,"Aux head",7.8); box(49,BOT+1.2,7,3.6,C_AUX,"Aux head",7.8)
ar(45,TOP+3,49,TOP+3); ar(45,BOT+3,49,BOT+3); lbl(52.5,TOP+0.5,"$p_{DINO}$",7.5); lbl(52.5,BOT+0.6,"$p_{Clay}$",7.5)
box(58.5,85.2,8,4.2,C_D,"Disagree.\nmap D",8,bold=True); ar(56,TOP+2,58.5,88); ar(56,BOT+2.5,58.5,86)
lbl(62.5,90.0,"D = JSD($p_{DINO},p_{Clay}$)",7,it=True)
box(58.5,79,10,5,C_FUSE); lbl(63.5,82.5,"Disagreement-",8.2); lbl(63.5,81.2,"Gated Fusion",8.2); lbl(63.5,79.8,"(×4 depths)",7,c="#555")
ar(45,BOT+1.5,58.5,81.3); ar(45,TOP+0.8,49.5,78.4); ar(49.5,78.4,58.5,80.8); ar(62.5,85.2,63,84)
lbl(70.5,84.0,"4×(256,\n64×64)",6.8,c="#7a3aa6")
dx=72
for i,(nm,sh) in enumerate([("Dec3","16²"),("Dec2","32²"),("Dec1","64²"),("Final","512²")]):
    box(dx+i*5,80,4.4,6,C_DEC,f"{nm}\n{sh}",7.8)
    if i: ar(dx+i*5-0.6,83,dx+i*5,83)
ar(68.5,82.8,dx,83); lbl(dx+1.5*5+2,87.2,"LinkNet Decoder",9)
box(95,80,4.5,6,"#1c1c1c"); ar(dx+3*5+4.4,83,95,83); lbl(97.2,87.1,"Flood\nmask",8.3)

# ================= ViT BLOCK detail (y 60-75 left) =================
panel(2,60,30,15,"ViT Block (×24, frozen)")
seq=[("LayerNorm","#eef2f7"),("Multi-Head\nAttention","#dfe9f6"),("LayerNorm","#eef2f7"),("FFN + EA","#fde9c8")]
yb=70.5
for i,(t,c) in enumerate(seq):
    box(5,yb-i*2.6,12,2.1,c,t,7.6)
    if i: ar(11,yb-(i-1)*2.6,11,yb-i*2.6+2.1)
ax.text(20,72.5,"residual ⊕ around\nattention and FFN",fontsize=7,va="center",style="italic",color="#555")
ax.text(20,64.0,"At tap depths the (C,H,W)\nfeature map also passes\nthrough ADAC + PPA →",fontsize=7,va="center",color="#a85b00")
ar(17,61.2,30.5,61.2,color="#a85b00")

# ================= FFN + EARTH-ADAPTER detail (y 60-75 right) =================
panel(34,60,64,15,"FFN with Earth-Adapter (EA)   —   x ∈ ℝ^(B,N,1024),  N = 1024 (DINOv3 32²) / 4096 (Clay 64²)")
box(37,70.5,11,3,"#e9eef5","Linear\n1024→4096",7.4); box(50,70.5,7,3,"#eef7ea","GELU",7.6); box(59,70.5,12,3,"#e9eef5","Linear\n4096→1024",7.4)
ar(48,72,50,72); ar(57,72,59,72); lbl(54,74.2,"FFN(x)",7.5,bold=True)
# EA path
box(37,64.5,13,3.4,"#f3e6d2","2D FFT split\n(ρ=0.25) LF/HF",7.2)
for j,(t) in enumerate(["Spatial\nexpert","LF\nexpert","HF\nexpert"]):
    box(53+j*8,64.5,7,3.4,C_EA,t,7.2)
ar(50,66.2,53,66.2)
box(78,64.5,11,3.4,"#f0e6f7","Router: mean→\nLinear 1024→3\n→ softmax g",6.8)
ar(75,66.2,78,66.2)
box(90,64.5,6.5,3.4,C_FUSE,"Δ = Σ gₖ·eₖ",6.8)
ar(89,66.2,90,66.2)
box(78,61,18.5,2.4,"#f6d7c4","Output = FFN(x) + α·Δ   (α zero-init)",7.4,bold=True)
ar(87,64.5,87,63.4); ar(65,70.5,65,63.4); lbl(67.5,63.9,"+",9)

# ================= ADAC detail (y 40-56 left) =================
panel(2,40,46,16,"ADAC — Atrous Depth-wise Convolution Adapter   x ∈ ℝ^(B,1024,H,W)")
for j,d in enumerate([1,2,3]):
    box(5+j*12,49,10.5,3.4,"#dfe9f6",f"DW-Conv 3×3\ndilation {d}",7.2)
ar(15.5,50.7,16,50.7); ar(27.5,50.7,28,50.7)
box(16,44,12,3.2,"#eef2f7","Sum → GN\n+ GELU",7.4); ar(10,49,16,46.6); ar(22,49,22,47.2); ar(34,49,28,46.6)
box(30,44,8,3.2,"#e9eef5","1×1\nconv",7.4); ar(28,45.6,30,45.6)
box(40,44,6.5,3.2,C_FUSE,"×γ\n(0-init)",7.2); ar(38,45.6,40,45.6)
lbl(24,41.5,"Output:  x + γ·y   (residual)",7.6,bold=True)

# ================= PPA detail (y 40-56 right) =================
panel(50,40,48,16,"PPA — Pyramid Pooling Adapter   x ∈ ℝ^(B,1024,H,W)")
for j,s in enumerate(["1×1\n(global)","2×2","3×3","6×6"]):
    box(53+j*7,49,6,3.4,"#dfe9f6",s,7.2)
box(82,49,7,3.4,"#eef2f7","Concat",7.4); ar(81,50.7,82,50.7)
box(90,49,6.5,3.4,"#e9eef5","Fuse 1×1\nGN+GELU",6.8); ar(89,50.7,90,50.7)
box(73,44,9,3.2,C_FUSE,"×γ (0-init)",7.4); ar(93,49,93,47.2); ar(93,47.2,82,45.6); ar(82,45.6,82,45.6)
lbl(66,41.5,"Output:  x + γ·y   (residual)",7.6,bold=True)
ar(82,45.6,82,45.6)

# ================= DISAGREEMENT-GATED FUSION detail (y 3-34) =================
panel(2,3,96,31,"Disagreement-Gated Fusion   —   per tapped depth  k ∈ {1,2,3,4}")
box(4,24,14,4,C_DINO,"DINOv3 tap\n(1024, 32×32)",7.8); box(4,17,14,4,C_CLAY,"Clay tap\n(1024, 64×64)",7.8)
box(22,24,10,4,"#eef3fb","1×1 proj→256",7.6); ar(18,26,22,26)
box(22,17,10,4,"#eef7ea","1×1 proj→256",7.6); ar(18,19,22,19); lbl(27,22.4,"↑ upsample 32²→64²",6.6,c="#555")
box(22,8,11,4.2,C_AUX,"Aux heads\n(deep-sup.)",7.6); ar(11,17,11,12.3); ar(11,12.3,22,10.1)
box(37,8,15,4.2,C_D,"D = JSD(p$_D$,p$_C$)\nfp32 · detached",7.2,bold=True); ar(33,10.1,37,10.1)
box(37,18,9,6,"#f0e6f7","concat\n[a,b,D]",7.8); ar(32,26,37,22); ar(32,19,37,20); ar(44,12.2,44.5,18)
box(50,18,12,6,C_FUSE,"gate conv\n→ softmax\n→ (w$_D$,w$_C$)",7.6); ar(46,21,50,21)
box(66,18,13,6,"#f6d7c4","fused =\nw$_D$·a + w$_C$·b",8,bold=True); ar(62,21,66,21)
box(83,18,9,6,"#f6d7c4","3×3 conv\nGN+GELU",7.6); ar(79,21,83,21)
box(94,18.5,4,5,"#efe2f6","f$_k$",8); ar(92,21,94,21)
ax.text(4,5.2,"D conditions the gate on WHERE the two encoders disagree (label-free OOD signal); the routing policy is learned. "
        "Aux heads are trained by independent deep supervision (BCE+Dice) — no term couples them, so D stays emergent.",
        fontsize=7.5,style="italic",color="#444")

fig.suptitle("FloodDuo (arch6): dual frozen foundation encoders + lightweight adapters → disagreement-gated fusion → LinkNet decoder",
             fontsize=12.5,y=0.995)
fig.tight_layout(rect=[0,0,1,0.985])
for ext in ("png","pdf"):
    p=f"./FloodDuo_arch6_schematic.{ext}"
    fig.savefig(p,dpi=200,bbox_inches="tight"); print("->",p)
