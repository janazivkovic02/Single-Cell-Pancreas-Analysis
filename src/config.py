from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    out_figures: Path = PROJECT_ROOT / "outputs" / "figures"
    out_tables: Path = PROJECT_ROOT / "outputs" / "tables"
    out_models: Path = PROJECT_ROOT / "outputs" / "models"

PATHS = Paths()

RANDOM_STATE = 42

# ========= Preprocess params =========
TARGET_SUM = 1e4
N_HVG_MAIN = 4000
N_PCS = 50
N_NEIGHBORS = 15
N_PCS_NEIGHBORS = 40

# ========= Rules params =========
N_HVG_RULES = 100
MIN_SUPPORT = 0.01
MAX_LEN = 2
ECLAT_MIN_SUPPORT = 0.02
CORR_THRESHOLD = 0.2
CHI2_P_THRESHOLD = 0.01
CHI2_MAX_GENES = 200

# ========= Clustering params =========
K_CLUSTERS = 7
LEIDEN_RESOLUTIONS = [0.25, 0.5, 1.0]

# ========= Classification params =========
TEST_SIZE = 0.3