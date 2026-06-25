from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
 
DATA_RAW = PROJECT_ROOT / "data" / "raw"
OUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUT_ADATA = PROJECT_ROOT / "outputs" / "adata"
 
RANDOM_STATE = 42
 
# AnnData keys (ČEMU SLUŽI OVO) 
CLUSTER_KEY = "assigned_cluster"
BATCH_KEY = "batch"
BARCODE_KEY = "barcode"
COUNTS_LAYER = "counts"
 