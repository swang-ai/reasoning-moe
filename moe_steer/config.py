import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERTS_DIR = Path(os.environ.get("MOE_STEER_EXPERTS", REPO_ROOT / "data" / "experts"))

TRACES_DIR = Path(os.environ.get("MOE_STEER_TRACES", REPO_ROOT / "data" / "traces"))

TMP_DIR = Path(os.environ.get("MOE_STEER_TMP", REPO_ROOT / ".moe_steer_tmp"))

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))

for _d in (EXPERTS_DIR, TRACES_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)
