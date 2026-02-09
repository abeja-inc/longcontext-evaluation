from unittest.mock import patch

from benchmarks._core.evaluate.table import BaseTable
from benchmarks._core.save_table.to_wandb import push_to_wandb


def test_push_to_wandb_skips_empty_rows() -> None:
    table = BaseTable(name="empty", rows=[])

    with patch("benchmarks._core.save_table.to_wandb.wandb.log") as mock_log:
        push_to_wandb([table])

    mock_log.assert_not_called()
