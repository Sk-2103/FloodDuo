"""Load the released FloodDuo checkpoint and run inference on a GeoTIFF tile.

    from src.inference import load_model, predict_file
    model = load_model("floodduo_full.pt")
    prob, mask, profile = predict_file(model, "my_tile.tif")

Expects a 4-band (B, G, R, NIR) PlanetScope-style surface-reflectance GeoTIFF,
float32, values roughly in [0, 0.5+] -- the same format as the FloodPlanet/UFO
benchmarks. No manual normalization needed: both encoders normalize internally.
"""
import torch
import numpy as np
import rasterio

from .models.flood_model import FloodDualSeg


def load_model(checkpoint_path: str, device: str = "cpu") -> FloodDualSeg:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    cfg.pop("norm_mode", None)  # passed per-branch inside the constructor, not a kwarg
    model = FloodDualSeg(**cfg, pretrained_backbones=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval()


def predict_array(model: FloodDualSeg, image: np.ndarray, device: str = "cpu",
                   threshold: float = 0.5):
    """image: (4, H, W) float32 surface reflectance, bands B,G,R,NIR."""
    assert image.ndim == 3 and image.shape[0] == 4, "expected (4, H, W) B/G/R/NIR"
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob, prob > threshold


def predict_file(model: FloodDualSeg, image_path: str, device: str = "cpu",
                  threshold: float = 0.5):
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile
    prob, mask = predict_array(model, image, device=device, threshold=threshold)
    return prob, mask, profile


def save_mask(mask: np.ndarray, profile: dict, out_path: str):
    """Write a predicted binary mask as a single-band GeoTIFF, georeferenced
    like the input tile it came from."""
    out_profile = dict(profile)
    out_profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(mask.astype("uint8"), 1)
