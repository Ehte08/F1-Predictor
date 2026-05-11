from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "f1_datasets"
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "cache_folder"

MODEL_PATH = MODEL_DIR / "lgbm_ranker.pkl"
SNAPSHOT_PATH = MODEL_DIR / "feature_snapshot.pkl"

MIN_YEAR = 2018
TEST_FRAC = 0.20

ACTIVE_DRIVERS = [
    "Norris", "Piastri", "Leclerc", "Hamilton", "Russell", "Antonelli",
    "Verstappen", "Hadjar", "Alonso", "Stroll", "Bearman", "Ocon",
    "Gasly", "Colapinto", "Hulkenberg", "Bortoleto", "Sainz", "Albon",
    "Lawson", "Lindblad", "Perez", "Bottas",
]

ACTIVE_CONSTRUCTORS = [
    "McLaren", "Ferrari", "Mercedes", "Red Bull", "Aston Martin",
    "Audi", "Haas F1 Team", "Alpine", "Racing Bulls", "Williams", "Cadillac",
]

CAT_COLS = ["GP name", "team", "driver", "circuit name"]
DROP_COLS = ["finish", "date", "dob"]
