
import sys, os
# repo root, for `import src` -- works whether this cell runs inside Jupyter
# (cwd = notebook/) or as a plain script via `python notebook/*.py` (any cwd)
_repo_root = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
              if "__file__" in globals() else os.path.abspath(".."))
sys.path.insert(0, _repo_root)
_nb_dir = (os.path.dirname(os.path.abspath(__file__))
           if "__file__" in globals() else os.getcwd())  # write outputs next to this file/notebook

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB) -- using cuda")
else:
    print("No GPU detected -- running on CPU, this will be slower")

from huggingface_hub import hf_hub_download

REPO = "Sk-21/FloodDuo"
ckpt_path = hf_hub_download(REPO, "floodduo_full.pt")
sample_image_path = hf_hub_download(REPO, "examples/sample_tile.tif")
sample_mask_path = hf_hub_download(REPO, "examples/sample_mask.tif")
print("downloaded:", ckpt_path)

image_path = sample_image_path   # <- point this at your own .tif to use your own imagery
mask_path = sample_mask_path     # <- set to None if you don't have a ground-truth mask

from src.inference import read_image, to_reflectance

raw_image, profile = read_image(image_path)
image = to_reflectance(raw_image)

from src.inference import load_model, predict_array, save_mask

model = load_model(ckpt_path, device=device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"loaded on {device}, {n_params:.0f}M parameters")

prob, mask = predict_array(model, image, device=device, auto_reflectance=False)  # already converted above
print(f"predicted water fraction: {mask.mean():.1%}")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of displaying
import rasterio
import matplotlib.pyplot as plt


def false_color(img):
    x = img[[3, 2, 1]].astype(float)  # NIR, R, G
    out = np.empty_like(x)
    for k in range(3):
        lo, hi = np.percentile(x[k], 2), np.percentile(x[k], 98)
        out[k] = np.clip((x[k] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.transpose(out, (1, 2, 0))


gt = None
if mask_path is not None:
    with rasterio.open(mask_path) as src:
        gt = src.read(1) > 0.5

panels = [("False color (NIR-R-G)", false_color(raw_image), None)]
if gt is not None:
    panels.append(("Ground truth", gt, "water"))
panels.append(("FloodDuo prediction", mask, "water"))

fig, ax = plt.subplots(1, len(panels) + 1, figsize=(4.5 * (len(panels) + 1), 5))
for a, (title, data, kind) in zip(ax, panels):
    a.imshow(data, cmap=("Blues" if kind == "water" else None), vmin=0, vmax=1)
    a.set_title(title)
ax[-1].imshow(false_color(raw_image)); ax[-1].imshow(mask, cmap="Blues", alpha=0.45)
ax[-1].set_title("Prediction overlay")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
_fig_path = os.path.join(_nb_dir, "floodduo_inference_example.png")
fig.savefig(_fig_path, dpi=150)
print("saved figure:", _fig_path)

out_path = os.path.join(_nb_dir, "floodduo_prediction.tif")
save_mask(mask, profile, out_path)
print("saved:", out_path)
