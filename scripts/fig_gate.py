#!/usr/bin/env python3
"""F5: Disagreement-gated fusion routing — per-pixel gate weight w_spatial
(DINOv3) vs w_spectral (Clay). Shows the model leans on the spatial branch at
boundaries/urban edges and the spectral branch over homogeneous water."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIZ = "outputs/viz/manuscript_figs"

def cir(img):
    x = img[[3,2,1]].astype(float); o = np.empty_like(x)
    for k in range(3):
        lo,hi=np.percentile(x[k],2),np.percentile(x[k],98); o[k]=np.clip((x[k]-lo)/(hi-lo+1e-6),0,1)
    return np.transpose(o,(1,2,0))

# gather example tiles across datasets (up to 3)
tiles=[]
for ds in ["ufo","floodplanet"]:
    f=f"{VIZ}/diag_examples_{ds}.npz"
    if not os.path.exists(f): continue
    e=np.load(f,allow_pickle=True); meta=json.loads(str(e["meta"]))
    for i in range(4):
        if f"{i}_img" in e.files and e[f"{i}_wsp"].ndim==2:
            tiles.append((meta[i]["region"], e[f"{i}_img"], e[f"{i}_gt"], e[f"{i}_wsp"]))
        if len(tiles)>=3: break
    if len(tiles)>=3: break

plt.rcParams.update({"font.size": 13})
n=len(tiles)
fig,ax=plt.subplots(n,3,figsize=(9.5,3.2*n))
if n==1: ax=ax[None,:]
for r,(reg,img,gt,wsp) in enumerate(tiles):
    H=gt.shape[0]
    wup=np.kron(wsp,np.ones((H//wsp.shape[0],H//wsp.shape[1])))[:H,:H]
    ax[r,0].imshow(cir(img)); ax[r,0].set_ylabel(reg,fontsize=10)
    ax[r,1].imshow(cir(img)); ax[r,1].imshow(np.ma.masked_where(gt<0.5,gt),
                   cmap=plt.cm.colors.ListedColormap([(0.1,0.45,1.0)]),alpha=.5)
    im=ax[r,2].imshow(wup,cmap="RdBu_r",vmin=0.2,vmax=0.8)
    ax[r,2].contour((gt>=0.5).astype(float),levels=[0.5],colors="k",linewidths=0.4)
    ax[r,0].set_ylabel(reg,fontsize=13)
    if r==0:
        ax[r,0].set_title("RGB (4-3-2)",fontsize=14); ax[r,1].set_title("Ground truth",fontsize=14)
        ax[r,2].set_title("Gate weight $w_{spatial}$",fontsize=14)
    for c in range(3): ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
fig.subplots_adjust(wspace=0.03, hspace=0.06)
cb=fig.colorbar(im,ax=ax[:, 2],fraction=0.04,pad=0.02)
cb.set_label("← Clay (spectral)     DINOv3 (spatial) →",fontsize=12); cb.ax.tick_params(labelsize=11)
for ext in ("png","pdf"):
    p=f"{VIZ}/Fig_gate.{ext}"; fig.savefig(p,dpi=200,bbox_inches="tight"); print("->",p)
