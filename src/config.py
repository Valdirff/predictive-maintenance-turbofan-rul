"""Central configuration for the turbofan RUL prognostics project."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "CMAPSSData"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
SCALERS_DIR = ARTIFACTS_DIR / "scalers"
MODELS_DIR = ARTIFACTS_DIR / "trained_models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ── Dataset ───────────────────────────────────────────────────────────────────
# Available subsets: "FD001", "FD002", "FD003", "FD004"
SUBSET = "FD001"

# Column names (26 columns total in C-MAPSS)
COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors to DROP — uninformative in FD001 as per 'artigo_XGBoost_1'
SENSORS_TO_DROP = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",   # constant
    "sensor_5",   # constant
    "sensor_6",   # constant
    "sensor_10",  # constant
    "sensor_16",  # constant
    "sensor_18",  # constant
    "sensor_19",  # constant
]

# Operational settings columns
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# ── Target ────────────────────────────────────────────────────────────────────
RUL_CAP: int | None = 130        # set None to disable early-life capping
VALIDATION_ENGINE_RATIO = 0.2    # fraction of engines used for validation

# ── Health Indicator ──────────────────────────────────────────────────────────
HI_SMOOTHING_WINDOW = 5          # rolling mean window for smoothing sensors
HI_METHOD = "logistic"           # "logistic" | "pca" | "weighted"
HI_FAILURE_THRESHOLD = 0.1       # HI value fallback (used by exponential model only)

# ── Stochastic Degradation Model ──────────────────────────────────────────────
HI_LOGISTIC_N_SAMPLES = 5        # healthy/failure samples per engine for logistic HI
STOCHASTIC_WEIGHT_Q = 1.2        # geometric series ratio for WLS weights (q > 1)
STOCHASTIC_MAX_EXTRAPOLATION = 200   # max cycles to extrapolate forward (conservative)
STOCHASTIC_N_BOOTSTRAP = 200     # Monte Carlo samples for uncertainty (CI)

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGBOOST_WINDOW = 30              # rolling window size for feature engineering
XGBOOST_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}
XGBOOST_CV_FOLDS = 3

# ── LSTM ──────────────────────────────────────────────────────────────────────
LSTM_WINDOW = 30                 # sequence length
LSTM_HIDDEN_SIZE = 64            # Increase for better capacity (XGBoost matching)
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3               # Increased for better regularization (prevent overfitting)
LSTM_BATCH_SIZE = 64             # slightly larger for smoother gradients
LSTM_EPOCHS = 100
LSTM_LR = 0.0005                 # Slower learning for better precision
LSTM_PATIENCE = 20               # be patient to find better minima

# ── Visualization ─────────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
COLOR_PALETTE = {
    "exponential": "#E07B39",   # amber
    "stochastic": "#8B5CF6",    # violet
    "xgboost": "#3B82F6",       # blue
    "lstm": "#10B981",          # emerald
    "actual": "#6B7280",        # gray
}
