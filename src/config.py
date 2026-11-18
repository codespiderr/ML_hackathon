import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Student folder" / "Student folder"
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"

MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MAPS_DIR = BASE_DIR / "maps"
REPORT_DIR = BASE_DIR / "report"

# Ensure directories exist
for p in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR, MAPS_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_SEED = 42

# Spatial & Candidate Generation Parameters
SEARCH_RADIUS_METERS = 75.0  # Search radius around GPS observation for candidate arcs
TOP_K_CANDIDATES = 10        # Max candidate road arcs per GPS observation
METERS_PER_DEGREE_LAT = 111320.0  # Approx meters per degree latitude at ~49-50 deg N

# HMM Parameters
SIGMA_Z = 10.0   # GPS measurement error standard deviation in meters
BETA = 5.0       # Route distance error scale parameter for transition probabilities
MAX_NETWORK_HOP_SEARCH = 15  # Graph search depth for transition shortest paths

# API & Server Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
UI_PORT = 8501
