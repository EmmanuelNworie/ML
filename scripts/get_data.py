import os
import zipfile
import pathlib
import pandas as pd

COMP = "house-prices-advanced-regression-techniques"
ZIP = f"{COMP}.zip"


def load_house_prices():
    """
    Downloads + loads the Kaggle House Prices competition dataset.

    Requires environment variables:
        KAGGLE_USERNAME
        KAGGLE_KEY

    Returns:
        train_df, test_df
    """

    train_path = pathlib.Path("train.csv")
    test_path = pathlib.Path("test.csv")

    # If already present, just load
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    # Check credentials
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError(
            "Missing Kaggle credentials.\n\n"
            "Fix (Colab): Click 🔑 Secrets (left sidebar) and add:\n"
            "  KAGGLE_USERNAME = your Kaggle username\n"
            "  KAGGLE_KEY      = your Kaggle API key\n\n"
            "Then Runtime -> Restart runtime, and run again.\n"
            "Also ensure you have joined/accepted the competition rules on Kaggle."
        )

    # Download dataset
    os.system(f"kaggle competitions download -c {COMP} -p .")

    # Unzip
    with zipfile.ZipFile(ZIP, "r") as z:
        z.extractall(".")

    return pd.read_csv(train_path), pd.read_csv(test_path)
