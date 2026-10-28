

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

from huggingface_hub import hf_hub_download, snapshot_download

REPO = "Sk-21/FloodDuo"
ckpt_path = hf_hub_download(REPO, "floodduo_full.pt")

example_root = snapshot_download(REPO, allow_patterns=["examples/finetune/*/*/*.tif"])
example_root = os.path.join(example_root, "examples", "finetune")
print("downloaded example set to:", example_root)

from src.finetune import list_tile_folder

finetune_dir = os.path.join(example_root, "train")   # <- your fine-tuning folder
test_dir = os.path.join(example_root, "test")         # <- your separate same-region test folder

train_images, train_masks = list_tile_folder(finetune_dir)
test_images, test_masks = list_tile_folder(test_dir)
print(f"fine-tuning on {len(train_images)} tiles, evaluating on {len(test_images)} "
      f"separate tiles from the same region")

import numpy as np
from src.inference import load_model, read_image, to_reflectance, predict_array

model = load_model(ckpt_path, device=device)


def evaluate(model, image_paths, mask_paths):
    ious = []
    for ip, mp in zip(image_paths, mask_paths):
        raw, _ = read_image(ip)
        image = to_reflectance(raw, verbose=False)
        _, pred = predict_array(model, image, device=device, auto_reflectance=False)
        gt_raw, _ = read_image(mp)
        gt = gt_raw[0] > 0.5
        inter = (pred & gt).sum(); union = (pred | gt).sum()
        ious.append(float(inter / union) if union > 0 else 1.0)
    return np.array(ious)


before = evaluate(model, test_images, test_masks)
print(f"BEFORE fine-tuning: mean IoU {before.mean():.3f}  (n={len(before)} test tiles)")

from src.finetune import load_tiles, finetune

images, masks = load_tiles(train_images, train_masks, verbose=False)
history = finetune(model, images, masks, epochs=40, lr=1e-4, device=device, log_every=10)

after = evaluate(model, test_images, test_masks)
print(f"AFTER fine-tuning:  mean IoU {after.mean():.3f}  (n={len(after)} test tiles)")
print(f"delta: {after.mean() - before.mean():+.3f}  "
      f"({int((after > before).sum())} improved, {int((after < before).sum())} worsened, "
      f"{int((after == before).sum())} unchanged, out of {len(before)})")

import matplotlib
matplotlib.use("Agg")  # headless: save figures instead of displaying
import rasterio
import matplotlib.pyplot as plt


def false_color(img):
    x = img[[3, 2, 1]].astype(float)
    out = np.empty_like(x)
    for k in range(3):
        lo, hi = np.percentile(x[k], 2), np.percentile(x[k], 98)
        out[k] = np.clip((x[k] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.transpose(out, (1, 2, 0))


i = int(np.argmax(after - before))  # the test tile fine-tuning helped most, for illustration
raw, _ = read_image(test_images[i])
image = to_reflectance(raw, verbose=False)
with rasterio.open(test_masks[i]) as src:
    gt = src.read(1) > 0.5
_, pred = predict_array(model, image, device=device, auto_reflectance=False)

fig, ax = plt.subplots(1, 3, figsize=(13, 5))
ax[0].imshow(false_color(raw)); ax[0].set_title("False color")
ax[1].imshow(gt, cmap="Blues", vmin=0, vmax=1); ax[1].set_title("Ground truth")
ax[2].imshow(pred, cmap="Blues", vmin=0, vmax=1)
ax[2].set_title(f"Fine-tuned prediction (IoU {after[i]:.2f}, was {before[i]:.2f})")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
_fig_path = os.path.join(_nb_dir, "floodduo_finetune_example.png")
fig.savefig(_fig_path, dpi=150)
print("saved figure:", _fig_path)

out_path = os.path.join(_nb_dir, "floodduo_finetuned.pt")
torch.save(model.state_dict(), out_path)
print("saved:", out_path)
# Reload later with: model.load_state_dict(torch.load(out_path))
