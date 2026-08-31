"""FloodPlanet + UFO dataset. Returns raw reflectance (4, H, W) float32 and
binary mask; train: random crop + flips/rot90, val/test: full 1024 tiles.
"""

import os
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(os.environ.get("FLOODBENCH_ROOT", "data/floodbench"))
DATASETS = {"floodplanet": "floodplanet_planetScope", "ufo": "UFO"}
# paired Sentinel-2 (10 bands [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12], 320x320,
# same footprint as the 1024x1024 PS tiles; names = PS name without FP_<Country>_)
S2_ROOT = DATA_ROOT / "floodPlanet_Sen2" / "data"


def s2_path_for(ps_path: Path, split: str):
    short = "_".join(ps_path.stem.split("_")[2:])  # FP_Bangladesh_BGD_40_119 -> BGD_40_119
    for sp in (split, "train", "val", "test"):  # name-matched, split may differ
        p = S2_ROOT / sp / "image" / f"{short}.tif"
        if p.exists():
            return p
    return None


def tile_region(path) -> str:
    """Region from filename: FP_<Region>_... / UFO_<SITE>_... -> 2nd token."""
    return Path(path).stem.split("_")[1]


def spectral_augment(img: np.ndarray, gain=0.2, offset=0.02, gamma=0.25):
    """Radiometric jitter on raw reflectance (4,H,W), train-time only.

    Simulates cross-region appearance shifts: per-band gain (turbidity /
    sediment changes band ratios), per-band offset (atmospheric residual),
    global gamma (illumination). Magnitudes chosen to span the regional
    radiometry spread observed in FloodPlanet/UFO.
    """
    g = 1.0 + np.random.uniform(-gain, gain, (4, 1, 1)).astype(np.float32)
    o = np.random.uniform(-offset, offset, (4, 1, 1)).astype(np.float32)
    img = img * g + o
    if gamma > 0:
        gam = float(np.exp(np.random.uniform(-gamma, gamma)))
        img = np.clip(img, 0, None) ** gam
    return np.clip(img, 0, None)


class FloodDataset(Dataset):
    def __init__(self, datasets=("floodplanet", "ufo"), split="train",
                 crop=512, train=None, include_regions=None,
                 exclude_regions=None, spectral_aug=None, swir_aux=False,
                 patchify=False, tile=1024, include_names=None, exclude_names=None):
        """split: a split name, or list of splits (e.g. all three for LORO
        region-based re-splitting via include/exclude_regions).
        spectral_aug: None/False, True (defaults), or dict with keys
        p, gain, offset, gamma. Train-time only.
        swir_aux: also return MNDWI target from paired S2 (train-time
        SWIR distillation); tiles without a pair get aux_valid=0."""
        self.train = (split == "train") if train is None else train
        self.crop = crop
        self.swir_aux = swir_aux
        # patchify: deterministic non-overlapping grid crops (train only), so every
        # pixel of every tile is seen each epoch (tile/crop patches per tile) instead
        # of one random window. Flips/rot90 still applied on top.
        self.patchify = bool(patchify) and self.train
        self.grid = (tile // crop) if self.patchify else 1
        self.ppt = self.grid * self.grid  # patches per tile
        if spectral_aug is True:
            spectral_aug = {}
        self.spectral_aug = (dict(p=0.8, gain=0.2, offset=0.02, gamma=0.25)
                             | spectral_aug) if isinstance(spectral_aug, dict) \
            else None
        splits = [split] if isinstance(split, str) else list(split)
        inc_n = set(include_names) if include_names is not None else None
        exc_n = set(exclude_names) if exclude_names is not None else None
        self.items = []
        for ds in datasets:
            for sp in splits:
                img_dir = DATA_ROOT / DATASETS[ds] / sp / "image"
                for f in sorted(img_dir.glob("*.tif")):
                    r = tile_region(f)
                    if include_regions is not None and r not in include_regions:
                        continue
                    if exclude_regions is not None and r in exclude_regions:
                        continue
                    if inc_n is not None and f.stem not in inc_n:
                        continue
                    if exc_n is not None and f.stem in exc_n:
                        continue
                    self.items.append(
                        (f, Path(str(f).replace("/image/", "/mask/")), ds))
        assert self.items, f"no tiles found for {datasets} {splits}"

    def __len__(self):
        return len(self.items) * self.ppt

    def _load_mndwi(self, img_path, ds, shape):
        """MNDWI (G-SWIR1)/(G+SWIR1) from paired S2, upsampled to PS grid."""
        if ds == "floodplanet":
            s2p = s2_path_for(img_path, "train")
            if s2p is not None:
                with rasterio.open(s2p) as src:
                    g = src.read(2).astype(np.float32)    # B3 green
                    s1 = src.read(9).astype(np.float32)   # B11 SWIR1
                mndwi = (g - s1) / (g + s1 + 1e-6)
                t = torch.from_numpy(mndwi)[None, None]
                mndwi = torch.nn.functional.interpolate(
                    t, size=shape, mode="bilinear",
                    align_corners=False)[0, 0].numpy()
                return mndwi, 1.0
        return np.zeros(shape, np.float32), 0.0

    def __getitem__(self, idx):
        pidx = idx % self.ppt          # which grid patch (0 when not patchify)
        img_path, mask_path, ds = self.items[idx // self.ppt]
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        aux_valid = 0.0
        if self.swir_aux:
            mndwi, aux_valid = self._load_mndwi(img_path, ds, mask.shape)
        else:
            mndwi = None

        if self.train:
            c = self.crop
            _, H, W = img.shape
            if self.patchify:
                y = (pidx // self.grid) * c    # deterministic non-overlapping quadrant
                x = (pidx % self.grid) * c
            else:
                y = random.randint(0, H - c)
                x = random.randint(0, W - c)
            img = img[:, y:y + c, x:x + c]
            mask = mask[y:y + c, x:x + c]
            if mndwi is not None:
                mndwi = mndwi[y:y + c, x:x + c]
            if random.random() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
                if mndwi is not None:
                    mndwi = mndwi[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[::-1]
                if mndwi is not None:
                    mndwi = mndwi[::-1]
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                mask = np.rot90(mask, k)
                if mndwi is not None:
                    mndwi = np.rot90(mndwi, k)
            img, mask = img.copy(), mask.copy()
            if mndwi is not None:
                mndwi = mndwi.copy()
            if self.spectral_aug and random.random() < self.spectral_aug["p"]:
                a = self.spectral_aug
                img = spectral_augment(img, a["gain"], a["offset"], a["gamma"])

        out = {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "dataset": ds,
            "name": img_path.stem,
        }
        if self.swir_aux:
            out["mndwi"] = torch.from_numpy(
                mndwi if mndwi is not None else
                np.zeros(mask.shape, np.float32))
            out["aux_valid"] = torch.tensor(aux_valid)
        return out
