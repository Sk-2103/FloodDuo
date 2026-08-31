#!/usr/bin/env python3
"""F4: Disagreement map D as a label-free OOD / reliability signal.
(a) pixel error-rate vs D (reliability + ECE), (b) per-region mean-D vs IoU,
(c-d) example D heatmaps over tiles. Reads diag_stats.json + diag_examples_*.npz."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

VIZ = "outputs/viz/manuscript_figs"
s = json.load(open(f"{VIZ}/diag_stats.json"))
dbins = np.array(s["dbins"]); ctr = 0.5*(dbins[:-1]+dbins[1:]); ln2 = np.log(2)
plt.rcParams.update({"font.size": 13})


def cir(img):
    x = img[[3,2,1]].astype(float); o = np.empty_like(x)
    for k in range(3):
        lo,hi = np.percentile(x[k],2), np.percentile(x[k],98); o[k]=np.clip((x[k]-lo)/(hi-lo+1e-6),0,1)
    return np.transpose(o,(1,2,0))

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1])

# (a) reliability: error vs D, pooled both datasets
axa = fig.add_subplot(gs[0, 0:2])
tot = np.array(s["dcal"]["floodplanet"]) + np.array(s["dcal"]["ufo"])
rate = tot[:,0]/np.clip(tot[:,1],1,None); frac = tot[:,1]/tot[:,1].sum()
ece = float(np.sum(frac*np.abs(rate - ctr/ln2)))
axa.plot(ctr, rate, "-o", color="#b2182b", lw=2)
axa.plot([0,ln2],[0,1],"--",color="0.6")
axa.set_xlabel("Disagreement  D  (JSD)", fontsize=14); axa.set_ylabel("Pixel error rate", fontsize=14)
axa.tick_params(labelsize=12)
axa.text(0.5,0.04,f"(a) Reliability (ECE={ece:.3f})",transform=axa.transAxes,ha="center",
         fontsize=14,fontweight="bold",bbox=dict(fc="white",ec="0.6",alpha=.85,pad=3)); axa.grid(ls=":",alpha=.4)

# (b) region mean-D vs IoU
axb = fig.add_subplot(gs[0, 2:4])
xs=[]; ys=[]
for ds,mk,c in [("floodplanet","o","#1f78b4"),("ufo","^","#33a02c")]:
    rx=[s["regD"][ds][r] for r in s["regD"][ds] if r in s["regIoU"][ds]]
    ry=[s["regIoU"][ds][r] for r in s["regD"][ds] if r in s["regIoU"][ds]]
    axb.scatter(rx,ry,marker=mk,c=c,s=45,edgecolor="k",lw=.5,label=("FloodPlanet" if ds=="floodplanet" else "UFO"))
    xs+=rx; ys+=ry
if len(xs)>2:
    r=np.corrcoef(xs,ys)[0,1]
    a,b=np.polyfit(xs,ys,1); xl=np.array([min(xs),max(xs)]); axb.plot(xl,a*xl+b,color="0.4",lw=1.4)
    axb.text(0.5,0.04,f"(b) Region mean-D vs IoU (r={r:.2f})",transform=axb.transAxes,ha="center",
             fontsize=14,fontweight="bold",bbox=dict(fc="white",ec="0.6",alpha=.85,pad=3))
axb.set_xlabel("Region mean disagreement  D", fontsize=14); axb.set_ylabel("Region IoU", fontsize=14)
axb.tick_params(labelsize=12); axb.grid(ls=":",alpha=.4); axb.legend(fontsize=12)

# (c,d) two example D heatmaps (one per dataset), each: RGB | D (+error contour)
import os
exs=[]
for ds in ["floodplanet","ufo"]:
    f=f"{VIZ}/diag_examples_{ds}.npz"
    if not os.path.exists(f): continue
    e=np.load(f, allow_pickle=True); meta=json.loads(str(e["meta"]))
    for i in range(4):
        if f"{i}_img" in e.files:
            exs.append((meta[i]["region"], e[f"{i}_img"], e[f"{i}_gt"], e[f"{i}_pred"], e[f"{i}_D"])); break
for j,(reg,img,gt,pred,D) in enumerate(exs[:2]):
    a1=fig.add_subplot(gs[1, j*2]); a1.imshow(cir(img))
    a1.set_title(f"({chr(99+j)}) {reg}: RGB (4-3-2)",fontsize=13); a1.axis("off")
    a2=fig.add_subplot(gs[1, j*2+1])
    Dup=np.kron(D,np.ones((gt.shape[0]//D.shape[0],gt.shape[1]//D.shape[1])))[:gt.shape[0],:gt.shape[1]]
    im=a2.imshow(Dup, cmap="magma", vmin=0, vmax=ln2)
    a2.contour((pred!=(gt>=0.5)).astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
    a2.set_title("Disagreement D  (cyan = error)",fontsize=13); a2.axis("off")
    cb2=fig.colorbar(im, ax=a2, fraction=0.046, pad=0.02); cb2.ax.tick_params(labelsize=11)
fig.tight_layout()
for ext in ("png","pdf"):
    p=f"{VIZ}/Fig_disagreement.{ext}"; fig.savefig(p,dpi=200,bbox_inches="tight"); print("->",p)
