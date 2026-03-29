# Copyright (c) 2026 Mamata Joshi, Afsana Tasnim Juha, Rob Rood
# Licensed under the MIT License
from pathlib import Path

import pytest

from eda.data_ops import get_dataset_summary, load_csv_data

# import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("meirnizri/covid19-dataset")
# print("Path to dataset files:", path)

COVID_DATA = Path(__file__).parent / "test_data/CovidData.csv"

class TestEDA:
    def test_load_csv_data(self):
        data = load_csv_data(COVID_DATA)
        assert data.shape == (1048575, 21)
        get_dataset_summary(data)