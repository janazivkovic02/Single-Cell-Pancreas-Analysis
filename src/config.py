# src/config.py

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    out_figures: Path = PROJECT_ROOT / "outputs" / "figures"
    out_adata: Path = PROJECT_ROOT / "outputs" / "adata"

PATHS = Paths()


RANDOM_STATE = 42
