# Copyright (c) 2026 Mamata Joshi, Afsana Tasnim Juha, Rob Rood
# Licensed under the MIT License


# This file contains data file operations to allow for passing in any dataset file and loading it regardless of column names


from pathlib import Path

from sklearn.utils import Bunch


def load_dataset_csv(file: Path, columns: list= None) -> Bunch:
    """loads the requested dataset CSV file and returns it as a scikit-learn Bunch object.

    :param file: a Path object pointing to the location of the dataset csv file.
    :param columns: the specific columns that you would like to load from the dataset. Default is to load all columns.
    """
