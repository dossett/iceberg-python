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
"""Benchmark metrics evaluation across the sequential entries in one manifest.

Run with:
    uv run pytest tests/benchmark/test_metrics_evaluator_benchmark.py -v -s -m benchmark
"""

from __future__ import annotations

import statistics
import timeit

import pytest

from pyiceberg.conversions import to_bytes
from pyiceberg.expressions import And, BooleanExpression, EqualTo, GreaterThanOrEqual, LessThanOrEqual, Or
from pyiceberg.manifest import DataFile, FileFormat
from pyiceberg.schema import Schema
from pyiceberg.table import ManifestGroupPlanner, Table
from pyiceberg.types import LongType, NestedField


def _combined_filter() -> BooleanExpression:
    branches: list[BooleanExpression] = []
    for value in range(11):
        branch: BooleanExpression = GreaterThanOrEqual("x", 0)
        for predicate in (
            LessThanOrEqual("x", 10),
            EqualTo("y", value),
            EqualTo("y", value + 1),
            EqualTo("y", value + 2),
            EqualTo("y", value + 3),
        ):
            branch = And(branch, predicate)
        branches.append(branch)

    combined = branches[0]
    for branch in branches[1:]:
        combined = Or(combined, branch)
    return combined


def _data_file(file_number: int) -> DataFile:
    long_type = LongType()
    return DataFile.from_args(
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition={},
        record_count=100,
        file_size_in_bytes=1,
        value_counts={1: 100, 2: 100},
        null_value_counts={1: 0, 2: 0},
        lower_bounds={1: to_bytes(long_type, 0), 2: to_bytes(long_type, 0)},
        upper_bounds={1: to_bytes(long_type, 10), 2: to_bytes(long_type, 10)},
    )


@pytest.mark.benchmark
def test_metrics_evaluator_reuse(table_v2: Table) -> None:
    num_files = 1_000
    schema = Schema(
        NestedField(1, "x", LongType(), required=True),
        NestedField(2, "y", LongType(), required=True),
        *(NestedField(field_id, f"unused_{field_id}", LongType(), required=False) for field_id in range(3, 103)),
        schema_id=table_v2.metadata.current_schema_id,
    )
    metadata = table_v2.metadata.model_copy(update={"schemas": [schema]})
    planner = ManifestGroupPlanner(table_metadata=metadata, io=table_v2.io, row_filter=_combined_filter())
    data_files = [_data_file(file_number) for file_number in range(num_files)]

    def evaluate_files() -> int:
        metrics_evaluator = planner._build_metrics_evaluator()
        return sum(metrics_evaluator(data_file) for data_file in data_files)

    assert evaluate_files() == num_files
    timings = timeit.repeat(evaluate_files, number=1, repeat=3)

    print(
        f"Evaluated metrics for {num_files} files with a 102-column schema and 66-leaf predicate in "
        f"{statistics.mean(timings):.3f}s (best: {min(timings):.3f}s)"
    )
