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

        labels = 1 - 2 * (df[definition.label_column] == definition.anomalous_value).astype(int)

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col != "__label__"]
        df_encoded = pd.get_dummies(df[categorical_cols], drop_first=False)

        other_cols = df.drop(columns=categorical_cols + ["__label__"])
        df_final = pd.concat([other_cols, df_encoded, labels], axis=1)

        definition = Dataset.from_pandas(df_final, preserve_index=False)
        definition.save_to_disk(str(output_path))
