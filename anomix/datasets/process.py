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

            cat_cols = df.select_dtypes([object]).columns
            df[cat_cols] = df[cat_cols].apply(lambda col: col.str.decode("utf-8"))
            df = pd.get_dummies(df, columns=cat_cols.difference([definition.label_column]), drop_first=False)
            print(df.info())
        else:
            raise ValueError("Unsupported file format")

        assert definition.label_column in df.columns, (
            f"{definition.label_column} not in columns for dataset {definition.name}"
        )

        df["__label__"] = 1 - 2 * (df[definition.label_column] == definition.anomalous_value).astype(int)
        df.drop(columns=[definition.label_column], inplace=True)

        print(df.info())

        definition = Dataset.from_pandas(df, preserve_index=False)
        definition.save_to_disk(str(output_path))
