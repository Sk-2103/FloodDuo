"""External benchmark datasets: Sen1Floods11 (S2 13-band) and KuroSiwo (SAR 2-band).

Layout assumed on disk (FloodBench benchmark root):
  sen1floods11/
    source/data/{ID}_{modality}.tif   (modalities: S2Hand, S1Hand, LabelHand, ...)
    splits/{train,val,test}.txt        (one tile ID per line)
  kurosiwo/
    {train,val,test}/image/{sample_id}/*.tif   (MS1_IVV_*.tif, MS1_IVH_*.tif, ...)
    {train,val,test}/mask/{sample_id}.tif
    {train,val,test}/valid_mask/{sample_id}.tif

Sen1Floods11 label:  -1 = nodata/ignore, 0 = non-water, 1 = water
KuroSiwo label:      0 = no water, 1 = permanent water, 2 = flood, 3 = invalid
  -> binary target: 1 if class in {1, 2}, 0 if class 0; ignore_index=255 for class 3
     and for pixels where valid_mask == 0.
"""

import os
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(os.environ.get("FLOODBENCH_ROOT", "data/floodbench"))
S11_ROOT  = DATA_ROOT / "sen1floods11"
S11_SRC   = S11_ROOT / "source" / "data"   # subdirs: S2L1CHand/, LabelHand/, ...
S11_IMG_DIR = S11_SRC / "S2L1CHand"
S11_MSK_DIR = S11_SRC / "LabelHand"
KURO_ROOT = DATA_ROOT / "kurosiwo"

S11_IGNORE  = -1    # nodata value in LabelHand masks
KURO_IGNORE = 255   # value we remap invalid pixels to


# ---------------------------------------------------------------------------
# Sen1Floods11
# ---------------------------------------------------------------------------

class Sen1Floods11Dataset(Dataset):
    """13-band Sentinel-2 (S2Hand) + hand-labeled binary water mask.

    Image: float32 (13, 512, 512), raw DN (0-~10000) — use per_image norm.
    Mask:  float32 (512, 512), values {-1, 0, 1}; train loss/metrics ignore -1.
    Train: random flip + rot90 (tiles already 512x512, crop=512 is whole tile).
    """

    IGNORE_INDEX = S11_IGNORE

    def __init__(self, split="train", crop=512, train=None, spectral_aug=None):
        self.train = (split == "train") if train is None else train
        self.crop = crop
        ids = (S11_ROOT / "splits" / f"{split}.txt").read_text().strip().splitlines()
        self.items = []
        for sid in ids:
            img_p = S11_IMG_DIR / f"{sid}_S2Hand.tif"
            msk_p = S11_MSK_DIR / f"{sid}_LabelHand.tif"
            if img_p.exists() and msk_p.exists():
                self.items.append((img_p, msk_p))
        assert self.items, f"no S11 tiles found for split={split}"

        # optional spectral augmentation (same interface as FloodDataset)
        if spectral_aug is True:
            spectral_aug = {}
        self.spectral_aug = (
            dict(p=0.8, gain=0.2, offset=0.02, gamma=0.25) | spectral_aug
        ) if isinstance(spectral_aug, dict) else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_p, msk_p = self.items[idx]
        with rasterio.open(img_p) as src:
            img = src.read().astype(np.float32)   # (13, H, W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        with rasterio.open(msk_p) as src:
            mask = src.read(1).astype(np.float32) # (H, W)

        if self.train:
            c = self.crop
            _, H, W = img.shape
            y0 = random.randint(0, H - c)
            x0 = random.randint(0, W - c)
            img  = img[:, y0:y0+c, x0:x0+c]
            mask = mask[y0:y0+c, x0:x0+c]
            if random.random() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[::-1]
            k = random.randint(0, 3)
            if k:
                img  = np.rot90(img,  k, axes=(1, 2))
                mask = np.rot90(mask, k)
            img, mask = img.copy(), mask.copy()
            if self.spectral_aug and random.random() < self.spectral_aug["p"]:
                a = self.spectral_aug
                # apply only to valid (non-ignore) pixels
                valid_px = mask != S11_IGNORE
                if valid_px.any():
                    from .flood_dataset import spectral_augment
                    # augment all 13 bands using same gain/offset/gamma logic
                    n_bands = img.shape[0]
                    g = 1.0 + np.random.uniform(-a["gain"], a["gain"], (n_bands, 1, 1)).astype(np.float32)
                    o = np.random.uniform(-a["offset"], a["offset"], (n_bands, 1, 1)).astype(np.float32)
                    img = img * g + o
                    if a["gamma"] > 0:
                        gam = float(np.exp(np.random.uniform(-a["gamma"], a["gamma"])))
                        img = np.clip(img, 0, None) ** gam
                    img = np.clip(img, 0, None)

        return {
            "image": torch.from_numpy(img),
            "mask":  torch.from_numpy(mask),
            "dataset": "sen1floods11",
            "name": img_p.stem.replace("_S2Hand", ""),
        }


# ---------------------------------------------------------------------------
# KuroSiwo
# ---------------------------------------------------------------------------

class KuroSiwoDataset(Dataset):
    """2-band SAR (MS1 VV + MS1 VH) + binary flood mask.

    Image: float32 (2, 224, 224), linear SAR backscatter — use per_image norm.
    Mask:  float32 (224, 224), {0=non-water, 1=water, 255=invalid/ignore}.
           Original classes: 0=no water, 1=permanent water, 2=flood, 3=invalid.
           We collapse: water = {1,2}, non-water = {0}, ignore = {3} ∪ {valid_mask=0}.
    max_tiles: cap training tiles (None = all; suggest 5000 for manageable runs).
    """

    IGNORE_INDEX = KURO_IGNORE

    def __init__(self, split="train", train=None, spectral_aug=None,
                 max_tiles=None):
        self.train = (split == "train") if train is None else train
        self.split = split
        split_dir = KURO_ROOT / split

        img_dir   = split_dir / "image"
        mask_dir  = split_dir / "mask"
        valid_dir = split_dir / "valid_mask"

        # each item in image/ is either a directory of per-file tifs or
        # a single tif; mask/ and valid_mask/ are flat tifs named {sample_id}.tif
        self.items = []
        for sample_path in sorted(img_dir.iterdir()):
            sample_id = sample_path.name if sample_path.is_dir() else sample_path.stem
            msk_p   = mask_dir  / f"{sample_id}.tif"
            valid_p = valid_dir / f"{sample_id}.tif"
            if not msk_p.exists():
                continue
            if sample_path.is_dir():
                # find MS1 (flood-time) VV and VH files
                vv_files = sorted(sample_path.glob("MS1_IVV_*.tif"))
                vh_files = sorted(sample_path.glob("MS1_IVH_*.tif"))
                if not (vv_files and vh_files):
                    continue
                vv_p, vh_p = vv_files[0], vh_files[0]
            else:
                continue  # unexpected layout
            self.items.append((vv_p, vh_p, msk_p,
                               valid_p if valid_p.exists() else None))

        if max_tiles and len(self.items) > max_tiles:
            rng = random.Random(42)
            self.items = rng.sample(self.items, max_tiles)

        assert self.items, f"no KuroSiwo tiles for split={split}"

        if spectral_aug is True:
            spectral_aug = {}
        self.spectral_aug = (
            dict(p=0.8, gain=0.2, offset=0.02, gamma=0.25) | spectral_aug
        ) if isinstance(spectral_aug, dict) else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vv_p, vh_p, msk_p, valid_p = self.items[idx]

        with rasterio.open(vv_p) as s: vv = s.read(1).astype(np.float32)
        with rasterio.open(vh_p) as s: vh = s.read(1).astype(np.float32)
        img = np.stack([vv, vh], axis=0)   # (2, H, W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)  # nodata fills
        img = np.log1p(np.clip(img, 0, None))  # log1p tames extreme backscatter outliers

        with rasterio.open(msk_p) as s:
            raw_mask = s.read(1).astype(np.int16)

        # remap: {0->0, 1->1, 2->1, 3->255}
        mask = np.zeros_like(raw_mask, dtype=np.float32)
        mask[(raw_mask == 1) | (raw_mask == 2)] = 1.0
        mask[raw_mask == 3] = KURO_IGNORE

        # apply valid_mask (0 = invalid region -> also ignore)
        if valid_p is not None:
            with rasterio.open(valid_p) as s:
                vm = s.read(1)
            mask[vm == 0] = KURO_IGNORE

        if self.train:
            if random.random() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[::-1]
            k = random.randint(0, 3)
            if k:
                img  = np.rot90(img,  k, axes=(1, 2))
                mask = np.rot90(mask, k)
            img, mask = img.copy(), mask.copy()
            if self.spectral_aug and random.random() < self.spectral_aug["p"]:
                a = self.spectral_aug
                g = 1.0 + np.random.uniform(-a["gain"], a["gain"], (2, 1, 1)).astype(np.float32)
                o = np.random.uniform(-a["offset"], a["offset"], (2, 1, 1)).astype(np.float32)
                img = img * g + o
                if a["gamma"] > 0:
                    gam = float(np.exp(np.random.uniform(-a["gamma"], a["gamma"])))
                    img = np.clip(img, 0, None) ** gam
                img = np.clip(img, 0, None)

        return {
            "image": torch.from_numpy(img),
            "mask":  torch.from_numpy(mask),
            "dataset": "kurosiwo",
            "name": vv_p.stem,
        }
