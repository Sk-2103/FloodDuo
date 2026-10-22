"""Dataset/runs/checkpoint paths, each overridable by its env var (see README)."""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("FLOODDUO_DATA", "data"))
RUNS_ROOT = Path(os.environ.get("FLOODDUO_RUNS", "runs"))

CLAY_CKPT = os.environ.get("FLOODDUO_CLAY_CKPT", "pretrained/clay-v1.5.ckpt")
DOFA_CKPT_BASE = os.environ.get("FLOODDUO_DOFA_BASE_CKPT", "pretrained/DOFA_ViT_base_e100.pth")
DOFA_CKPT_LARGE = os.environ.get("FLOODDUO_DOFA_LARGE_CKPT", "pretrained/DOFA_ViT_large_e100.pth")
# Prithvi/TerraMind/CROMA/SSL4EO-DINO: vendored pangaea-bench checkout root
# (sys.path + its pretrained_models/), see README.
PANGAEA_ROOT = os.environ.get("FLOODDUO_PANGAEA_ROOT", "third_party/pangaea-bench")
