# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import pytest

import pyiceberg.table as table_module
from pyiceberg.conversions import to_bytes
from pyiceberg.expressions import BooleanExpression, EqualTo
from pyiceberg.manifest import DataFile, FileFormat
from pyiceberg.schema import Schema
from pyiceberg.table import ManifestGroupPlanner, Table
from pyiceberg.types import LongType


def _data_file(file_number: int, lower_bound: int | None = None, upper_bound: int | None = None) -> DataFile:
    long_type = LongType()
    return DataFile.from_args(
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition={},
        record_count=100,
        file_size_in_bytes=1,
        value_counts={1: 100},
        null_value_counts={1: 0},
        lower_bounds={1: to_bytes(long_type, lower_bound)} if lower_bound is not None else None,
        upper_bounds={1: to_bytes(long_type, upper_bound)} if upper_bound is not None else None,
    )


def test_build_metrics_evaluator_reuses_one_instance_per_callable(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingMetricsEvaluator:
        def __init__(
            self,
            schema: Schema,
            expr: BooleanExpression,
            case_sensitive: bool = True,
            include_empty_files: bool = False,
        ) -> None:
            self.calls: list[DataFile] = []
            instances.append(self)

        def eval(self, data_file: DataFile) -> bool:
            self.calls.append(data_file)
            return True

    instances: list[CountingMetricsEvaluator] = []
    monkeypatch.setattr(table_module, "_InclusiveMetricsEvaluator", CountingMetricsEvaluator)
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=EqualTo("x", 10))
    first_file = _data_file(1)
    second_file = _data_file(2)

    first_callable = planner._build_metrics_evaluator()
    assert not instances
    assert first_callable(first_file)
    assert first_callable(second_file)

    second_callable = planner._build_metrics_evaluator()
    assert len(instances) == 1
    assert second_callable(first_file)

    assert len(instances) == 2
    assert instances[0].calls == [first_file, second_file]
    assert instances[1].calls == [first_file]


def test_reused_metrics_evaluator_replaces_file_state(table_v2: Table) -> None:
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=EqualTo("x", 10))
    metrics_evaluator = planner._build_metrics_evaluator()
    cannot_match = _data_file(1, lower_bound=0, upper_bound=5)
    might_match = _data_file(2, lower_bound=10, upper_bound=15)
    missing_metrics = _data_file(3)

    assert not metrics_evaluator(cannot_match)
    assert metrics_evaluator(might_match)
    assert metrics_evaluator(missing_metrics)
    assert not metrics_evaluator(cannot_match)
