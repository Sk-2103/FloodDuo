"""Shared paths for scripts/, relative to the repo root unless overridden by env var."""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.environ.get("FLOODDUO_DATA", os.path.join(_REPO_ROOT, "data"))
RUNS_ROOT = os.environ.get("FLOODDUO_RUNS", os.path.join(_REPO_ROOT, "runs"))
VIZ_ROOT = os.environ.get("FLOODDUO_VIZ", os.path.join(_REPO_ROOT, "outputs"))
FIGS_ROOT = os.path.join(VIZ_ROOT, "manuscript_figs")
ASSETS_ROOT = os.path.join(VIZ_ROOT, "assets")
