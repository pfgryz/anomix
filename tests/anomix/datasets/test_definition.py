import json
from pathlib import Path

from anomix.datasets.definition import load_dataset_definitions


def test_load_dataset_definitions(tmp_path: Path):
    test_data = [
        {
            "name": "donors",
            "url": "https://example.com/donors.csv",
            "format": "csv",
            "label_column": "class",
            "anomalous_value": 1,
        }
    ]

    mock_json_path = tmp_path / "datasets.json"
    mock_json_path.write_text(json.dumps(test_data))

    result = load_dataset_definitions(mock_json_path)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].name == "donors"
    assert result[0].anomalous_value == 1
