from pathlib import Path

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
RANDOM_STATE = 42
CV_SPLITS = 5

# Models included in the ensemble pipeline
ENSEMBLE_MODELS = ["random_forest", "extra_trees", "adaboost", "gradient_boosting", "svc"]
