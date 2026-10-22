"""Fine-tune a released FloodDuo checkpoint on a handful of labeled tiles from
a new region, then run inference the same way as src/inference.py.

    from src.finetune import list_tile_folder, load_tiles, finetune
    from src.inference import load_model, predict_array

    model = load_model("floodduo_full.pt", device="cuda")
    image_paths, mask_paths = list_tile_folder("my_region/finetune")
    images, masks = load_tiles(image_paths, mask_paths)
    history = finetune(model, images, masks, epochs=30, lr=1e-4, device="cuda")
    prob, mask = predict_array(model, held_out_image, device="cuda")

Only FloodDuo's adapters/fusion/decoder update (~65M params); both backbones
stay frozen, exactly as in the original training -- `model.trainable_parameters()`
already reflects that, nothing extra to freeze here. With only a handful of
tiles, this trains on random 512px crops (flips/90-degree rotations) rather
than full 1024px tiles, both for VRAM and so each epoch sees different
windows of the same images.
"""
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .inference import read_image, to_reflectance
from .losses import FloodLoss, aux_seg_loss


def list_tile_folder(folder: str):
    """Pair up <folder>/image/*.tif with <folder>/mask/*.tif by filename.

    This is the layout used throughout this repo (FloodPlanet/UFO, and the
    bundled fine-tuning examples): a folder per site/region/split, each with
    `image/` and `mask/` subfolders of identically-named GeoTIFFs. Point this
    at two such folders -- one to fine-tune on, a separate one (same region,
    different tiles) to check the adaptation generalizes across the region
    rather than just memorizing the fine-tuning tiles."""
    images = sorted(glob.glob(os.path.join(folder, "image", "*.tif")))
    image_paths, mask_paths = [], []
    for ip in images:
        mp = os.path.join(folder, "mask", os.path.basename(ip))
        if os.path.exists(mp):
            image_paths.append(ip)
            mask_paths.append(mp)
    return image_paths, mask_paths


def load_tiles(image_paths, mask_paths, verbose: bool = True):
    """Read + convert a list of (image, mask) GeoTIFF pairs to arrays ready
    for fine-tuning. Masks are read as-is (0/1); images go through the same
    to_reflectance auto-detection as inference."""
    images, masks = [], []
    for ip, mp in zip(image_paths, mask_paths):
        raw, _ = read_image(ip)
        images.append(to_reflectance(raw, verbose=verbose))
        mask_raw, _ = read_image(mp)
        masks.append((mask_raw[0] > 0).astype(np.float32))
    return images, masks


def _random_crop_augment(image, mask, crop: int):
    _, h, w = image.shape
    if h > crop:
        y0 = random.randint(0, h - crop)
        x0 = random.randint(0, w - crop)
        image = image[:, y0:y0 + crop, x0:x0 + crop]
        mask = mask[y0:y0 + crop, x0:x0 + crop]
    k = random.randint(0, 3)
    image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)
    if random.random() < 0.5:
        image, mask = image[:, :, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        image, mask = image[:, ::-1, :], mask[::-1, :]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def finetune(model, images, masks, epochs: int = 30, lr: float = 1e-4,
             crop: int = 512, batch_size: int = 2, weight_decay: float = 0.01,
             aux_loss_weight: float = 0.3, device: str = "cuda", seed: int = 42,
             log_every: int = 5):
    """Fine-tune in place (model is modified and also returned). Returns the
    per-epoch mean loss."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    loss_fn = FloodLoss()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    n = len(images)
    use_amp = device.startswith("cuda")
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        order = list(range(n)); random.shuffle(order)
        ep_loss, n_steps = 0.0, 0
        for b0 in range(0, n, batch_size):
            idx = order[b0:b0 + batch_size]
            xb, yb = [], []
            for i in idx:
                img_c, msk_c = _random_crop_augment(images[i], masks[i], crop)
                xb.append(img_c); yb.append(msk_c)
            x = torch.from_numpy(np.stack(xb)).float().to(device)
            y = torch.from_numpy(np.stack(yb)).float().unsqueeze(1).to(device)

            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.enable_grad()
            with ctx:
                logits, extras = model(x, return_aux=True)
                loss, _ = loss_fn(logits.float(), y)
                if extras.get("aux_dino") is not None:
                    loss = loss + aux_loss_weight * (
                        aux_seg_loss(extras["aux_dino"].float(), y)
                        + aux_seg_loss(extras["aux_clay"].float(), y))
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_steps += 1

        mean_loss = ep_loss / max(n_steps, 1)
        history.append(mean_loss)
        if epoch % log_every == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  loss {mean_loss:.4f}")

    model.eval()
    return history
