"""Load the released FloodDuo checkpoint and run inference on a GeoTIFF tile.

    from src.inference import load_model, predict_file
    model = load_model("floodduo_full.pt")
    prob, mask, profile = predict_file(model, "my_tile.tif")  # auto-detects DN vs. reflectance

The model expects raw 4-band (B, G, R, NIR) surface reflectance, float, values
roughly in [0, 0.6]; both encoders standardize that internally, so nothing
further is needed once the input is in reflectance units. Many real-world
downloads (e.g. Planet's Surface Reflectance product, Sentinel-2 L2A) are
NOT already in that form -- they're uint16 GeoTIFFs scaled by 10000. See
`to_reflectance` below, which both branches of this module call automatically
unless you've already converted the array yourself.
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


def read_image(image_path: str):
    """Read a GeoTIFF as-is (raw values, original dtype) plus its rasterio profile."""
    with rasterio.open(image_path) as src:
        return src.read(), src.profile


def to_reflectance(image: np.ndarray, scale: float = None, verbose: bool = True) -> np.ndarray:
    """Convert a raw 4-band array to surface reflectance in [0, ~1].

    If `scale` is given, divides by it directly. Otherwise auto-detects:
    integer dtype, or float values well above 1 (reflectance realistically
    tops out under ~1.0), is treated as DN scaled by 10000 -- the convention
    used by Planet's Surface Reflectance product and Sentinel-2 L2A. Values
    already in a plausible reflectance range are left untouched.

    This is a heuristic for the common case, not a universal radiometric
    calibration -- if your data uses a different scale factor or is raw
    at-sensor radiance/DN needing per-band coefficients, pass `scale`
    explicitly (or convert before calling this).
    """
    raw_dtype, raw_max = image.dtype, float(image.max())
    image = image.astype(np.float32)
    if scale is not None:
        image = image / scale
        msg = f"divided by scale={scale} (explicit)"
    elif np.issubdtype(raw_dtype, np.integer) or raw_max > 1.5:
        image = image / 10000.0
        msg = f"detected scaled DN (dtype={raw_dtype}, max={raw_max:.0f}) -> divided by 10000"
    else:
        msg = f"already looks like reflectance (dtype={raw_dtype}, max={raw_max:.3f}) -> unchanged"
    if verbose:
        print(f"to_reflectance: {msg}; result range [{image.min():.4f}, {image.max():.4f}]")
    return image


def predict_array(model: FloodDualSeg, image: np.ndarray, device: str = "cpu",
                   threshold: float = 0.5, auto_reflectance: bool = True):
    """image: (4, H, W), bands B,G,R,NIR. Raw DN or reflectance -- see to_reflectance."""
    assert image.ndim == 3 and image.shape[0] == 4, "expected (4, H, W) B/G/R/NIR"
    if auto_reflectance:
        image = to_reflectance(image, verbose=False)
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob, prob > threshold


def predict_file(model: FloodDualSeg, image_path: str, device: str = "cpu",
                  threshold: float = 0.5, auto_reflectance: bool = True):
    image, profile = read_image(image_path)
    prob, mask = predict_array(model, image, device=device, threshold=threshold,
                                auto_reflectance=auto_reflectance)
    return prob, mask, profile


def save_mask(mask: np.ndarray, profile: dict, out_path: str):
    """Write a predicted binary mask as a single-band GeoTIFF, georeferenced
    like the input tile it came from."""
    out_profile = dict(profile)
    out_profile.update(count=1, dtype="uint8", nodata=None)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(mask.astype("uint8"), 1)
