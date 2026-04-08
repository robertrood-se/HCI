# Copyright (c) 2026 Mamata Joshi, Afsana Tasnim Juha, Rob Rood
# Licensed under the MIT License


# This file contains data file operations to allow for passing in any dataset file and loading it regardless of column names
from pathlib import Path
import pandas as pd
from typing import List, Optional

def load_csv_data(file_path: Path, columns: Optional[List[str]] = None, nrows: int = 20_000) -> pd.DataFrame:
    """loads the requested dataset CSV file and returns it as a Pandas Dataframe.

    :param ds_file: a Path object pointing to the location of the dataset csv file.
    :param columns: the specific columns that you would like to load from the dataset. Default is to load all columns.
    :param nrows: an int defining how many rows you'd like to load from your dataset. Defaults to 20_000 rows if not specified. This is to pevent kernel crashes.
    
    :returns: a pandas Dataframe of the dataset.
    """
    try:
        # usecols allows pandas to only read the necessary data into memory
        df = pd.read_csv(file_path, usecols=columns, nrows=nrows)
        return df
    
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error: One or more columns might be missing. {e}")
        return pd.DataFrame()

def get_dataset_summary(df: pd.DataFrame):
    """Prints a quick overview of the loaded data."""
    if df.empty:
        print("DataFrame is empty.")
        return
    
    print("--- Dataset Summary ---")
    print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    print(df.head())
    
    