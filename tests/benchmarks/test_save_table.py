import json
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarks._core.evaluate.table import BaseTable, BaseTableRow
from benchmarks._core.save_table import to_local
from benchmarks._core.save_table.to_wandb import push_to_wandb


@dataclass(frozen=True)
class DummyRow(BaseTableRow):
    payload: dict[str, object]
    extra: object


def test_save_to_local_handles_empty_table(tmp_path: Path) -> None:
    table = BaseTable(name="empty", rows=[])
    to_local.save_to_local([table], tmp_path, format="jsonl")
    to_local.save_to_local([table], tmp_path, format="csv")

    jsonl_path = tmp_path / "tables" / "empty.jsonl"
    csv_path = tmp_path / "tables" / "empty.csv"
    assert jsonl_path.exists()
    assert csv_path.exists()


def test_save_to_local_serializes_dict_and_objects(tmp_path: Path) -> None:
    row = DummyRow(model_name="m1", payload={"a": 1}, extra={1, 2})
    table = BaseTable(name="serialized", rows=[row])

    to_local.save_to_local([table], tmp_path, format="jsonl")
    jsonl_path = tmp_path / "tables" / "serialized.jsonl"
    record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert record["payload"] == {"a": 1}
    assert record["extra"] == "{1, 2}"

    to_local.save_to_local([table], tmp_path, format="csv")
    csv_path = tmp_path / "tables" / "serialized.csv"
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "payload" in csv_text
    assert "{\"a\": 1}" in csv_text


def test_push_to_wandb_skips_empty_rows() -> None:
    table = BaseTable(name="empty", rows=[])

    with patch("benchmarks._core.save_table.to_wandb.wandb.log") as mock_log:
        push_to_wandb([table])

    mock_log.assert_not_called()


def test_push_to_wandb_serializes_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    wandb_module = types.ModuleType("wandb")
    logged: dict[str, object] = {}

    class DummyTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows: list[list[object]] = []

        def add_data(self, *data):
            self.rows.append(list(data))

    def log(payload):
        logged.update(payload)

    wandb_module.Table = DummyTable
    wandb_module.log = log
    monkeypatch.setitem(sys.modules, "wandb", wandb_module)

    to_wandb = importlib.reload(importlib.import_module("benchmarks._core.save_table.to_wandb"))

    row = DummyRow(model_name="m1", payload={"a": 1}, extra={1, 2})
    table = BaseTable(name="wandb", rows=[row])

    to_wandb.push_to_wandb([table])

    assert "tables/wandb" in logged
    wandb_table = logged["tables/wandb"]
    assert wandb_table.columns == ["model_name", "payload", "extra"]
    assert wandb_table.rows[0][1] == json.dumps({"a": 1}, ensure_ascii=False)
