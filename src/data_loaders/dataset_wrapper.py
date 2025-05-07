from scipy.io import arff
import subprocess
from typing import Any
import numpy as np
from pathlib import Path
import pandas as pd


class DatasetWrapperBase:
    name: str
    _X: np.ndarray
    _Y: np.ndarray

    def __init__(self, name, X, Y):
        self.name = name
        self._X = X
        self._Y = Y

    def __repr__(self):
        return f"Dataset {self.name}"

    def __str__(self):
        return repr(self)

    def get_x_y_tuple(self):
        return self._X, self._Y

    def get_x(self):
        return self._X

    def get_y(self):
        return self._Y


class FileDatasetWrapperBase(DatasetWrapperBase):
    def from_file(self, file: Path):
        return self

    def preprocess_file(self, file: Path):
        pass


class CSVDatasetWrapperBase(FileDatasetWrapperBase):
    def from_file(self, file: Path):
        raw_df = pd.read_csv(file)

        X, Y = self.parse_dataframe(raw_df)
        self._X = X
        self._Y = Y

        return self

    def parse_dataframe(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pass


class CSVNumericalDatasetWrapper(CSVDatasetWrapperBase):
    def __init__(self, name, anomalous_category: str, anomalous_value: Any):
        super().__init__(name, None, None)
        self._anomaly_category = anomalous_category
        self._anomaly_value = anomalous_value

    def parse_dataframe(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self._anomaly_category in df.columns, "The anomalous category must be present in the header"
        X = df.drop(columns=self._anomaly_category).to_numpy()
        Y = (df[self._anomaly_category] == self._anomaly_value).astype(int).to_numpy().reshape(-1, 1)
        return (X, Y)


class TarXZArchiveNumericalDatasetWrapper(CSVNumericalDatasetWrapper):
    def from_file(self, file: Path):
        return super().from_file(file.with_suffix("").with_suffix(".csv"))

    def preprocess_file(self, file: Path):
        subprocess.run(["unxz", file], cwd=file.parent)
        subprocess.run(["tar", "-xvf", file.with_suffix("").with_suffix(".tar")], cwd=file.parent)


class ARFFDatasetWrapper(CSVDatasetWrapperBase):
    # Arf arf
    def __init__(self, name, anomalous_category: str, anomalous_value: Any):
        super().__init__(name, None, None)
        self._anomaly_category = anomalous_category
        self._anomaly_value = anomalous_value

    def from_file(self, file):
        arff_file = arff.loadarff(file)
        df = pd.DataFrame(arff_file[0])

        X, Y = self.parse_dataframe(df)
        self._X = X
        self._Y = Y

        return self

    def parse_dataframe(self, dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        pass


class ARFFCategoricalDatasetWrapper(ARFFDatasetWrapper):
    def parse_dataframe(self, dataframe: pd.DataFrame):
        df_features_only = dataframe.drop(columns=self._anomaly_category)
        one_hot_encoded = pd.get_dummies(df_features_only, drop_first=False)
        X = one_hot_encoded.to_numpy().astype(float)
        Y = (dataframe[self._anomaly_category] == self._anomaly_value).astype(int).to_numpy().reshape(-1, 1)
        return (X, Y)
