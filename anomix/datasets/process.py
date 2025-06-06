import pandas as pd
from datasets import Dataset
from scipy.io import arff
from tqdm import tqdm

from anomix.config import EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR
from anomix.datasets.definition import DatasetDefinition


def process_datasets(definitions: list[DatasetDefinition]) -> None:
    for definition in tqdm(definitions, desc="Processing Datasets"):
        extracted_path = EXTRACTED_DATA_DIR / f"{definition.name}.{definition.format}"
        output_path = PROCESSED_DATA_DIR / f"{definition.name}"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            continue

        if definition.format == "csv":
            df = pd.read_csv(extracted_path)
        elif definition.format == "arff":
            data, _ = arff.loadarff(extracted_path)
            df = pd.DataFrame(data)

            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].str.decode("utf-8")
        else:
            raise ValueError("Unsupported file format")

        assert definition.label_column in df.columns, (
            f"{definition.label_column} not in columns for dataset {definition.name}"
        )
        df["__label__"] = (df[definition.label_column] == definition.anomalous_value).astype(int)
        df.drop(columns=[definition.label_column], inplace=True)

        definition = Dataset.from_pandas(df, preserve_index=False)
        definition.save_to_disk(str(output_path))
