from pathlib import Path
import pandas as pd

from ..config import DATA_DIR, TRAIN_FILENAME, TEST_FILENAME


def find_csv_file(data_dir: Path, filename: str) -> Path:
    candidate = data_dir / filename
    if candidate.exists():
        return candidate

    kaggle_path = Path("../input") / filename
    if kaggle_path.exists():
        return kaggle_path

    raise FileNotFoundError(
        f"Could not find {filename}. Checked {data_dir} and ../input."
    )


def load_data(data_dir: Path = None):
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train_path = find_csv_file(data_dir, TRAIN_FILENAME)
    test_path = find_csv_file(data_dir, TEST_FILENAME)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
