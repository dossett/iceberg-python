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

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

import pyiceberg.table as table_module
from pyiceberg.expressions import And, BooleanExpression, GreaterThan
from pyiceberg.manifest import DataFile, FileFormat
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import ManifestGroupPlanner, Table, TableProperties
from pyiceberg.table.metadata import TableMetadataV2
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record, StructProtocol
from pyiceberg.types import LongType, NestedField


def _data_file(file_number: int, *partition_values: int) -> DataFile:
    return DataFile.from_args(
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(*partition_values),
        record_count=100,
        file_size_in_bytes=1,
    )


def test_partition_evaluator_prepares_once_per_spec(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator_calls: list[list[int]] = []

    def counting_expression_evaluator(
        schema: Schema, unbound: BooleanExpression, case_sensitive: bool
    ) -> Callable[[StructProtocol], bool]:
        calls: list[int] = []
        evaluator_calls.append(calls)

        def evaluate(struct: StructProtocol) -> bool:
            value = struct[0]
            calls.append(value)
            return value > 5

        return evaluate

    monkeypatch.setattr(table_module, "expression_evaluator", counting_expression_evaluator)
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=GreaterThan("x", 5))
    partition_evaluator = planner._build_partition_evaluator(0)

    assert len(evaluator_calls) == 1
    assert not partition_evaluator(_data_file(1, 1))
    assert not partition_evaluator(_data_file(2, 1))
    assert partition_evaluator(_data_file(3, 10))
    assert partition_evaluator(_data_file(4, 10))
    assert evaluator_calls == [[1, 10]]


def test_partition_evaluator_cache_keys_referenced_partition_fields(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluated_partitions: list[tuple[int | None, ...]] = []

    def counting_expression_evaluator(
        schema: Schema, unbound: BooleanExpression, case_sensitive: bool
    ) -> Callable[[StructProtocol], bool]:
        def evaluate(struct: StructProtocol) -> bool:
            evaluated_partitions.append(tuple(struct[pos] for pos in range(3)))
            return struct[0] > 5 and struct[1] > 5

        return evaluate

    schema = Schema(
        NestedField(1, "x", LongType(), required=True),
        NestedField(2, "y", LongType(), required=True),
        NestedField(3, "partition_hash", LongType(), required=True),
    )
    spec = PartitionSpec(
        PartitionField(1, 1000, IdentityTransform(), "x"),
        PartitionField(2, 1001, IdentityTransform(), "y"),
        PartitionField(3, 1002, IdentityTransform(), "partition_hash"),
        spec_id=0,
    )
    metadata = TableMetadataV2(
        location="s3://bucket/table",
        last_column_id=3,
        schemas=[schema],
        current_schema_id=schema.schema_id,
        partition_specs=[spec],
        default_spec_id=spec.spec_id,
    )

    monkeypatch.setattr(table_module, "expression_evaluator", counting_expression_evaluator)
    planner = ManifestGroupPlanner(
        table_metadata=metadata,
        io=table_v2.io,
        row_filter=And(GreaterThan("x", 5), GreaterThan("y", 5)),
    )
    partition_evaluator = planner._build_partition_evaluator(spec.spec_id)

    assert partition_evaluator(_data_file(1, 10, 10, 1))
    assert not partition_evaluator(_data_file(2, 10, 1, 2))
    assert partition_evaluator(_data_file(3, 10, 10, 3))
    assert evaluated_partitions == [(10, 10, None), (10, 1, None)]


@pytest.mark.parametrize(
    ("cache_size", "expected_evaluations"),
    [
        (0, [1, 1, 10, 1]),
        (1, [1, 10, 1]),
    ],
)
def test_partition_evaluator_cache_configuration(
    table_v2: Table,
    monkeypatch: pytest.MonkeyPatch,
    cache_size: int,
    expected_evaluations: list[int],
) -> None:
    evaluations: list[int] = []

    def counting_expression_evaluator(
        schema: Schema, unbound: BooleanExpression, case_sensitive: bool
    ) -> Callable[[StructProtocol], bool]:
        def evaluate(struct: StructProtocol) -> bool:
            evaluations.append(struct[0])
            return struct[0] > 5

        return evaluate

    monkeypatch.setattr(table_module, "expression_evaluator", counting_expression_evaluator)
    planner = ManifestGroupPlanner(
        table_metadata=table_v2.metadata,
        io=table_v2.io,
        row_filter=GreaterThan("x", 5),
        options={TableProperties.PARTITION_FILTER_CACHE_MAX_SIZE: str(cache_size)},
    )
    partition_evaluator = planner._build_partition_evaluator(0)

    assert not partition_evaluator(_data_file(1, 1))
    assert not partition_evaluator(_data_file(2, 1))
    assert partition_evaluator(_data_file(3, 10))
    assert not partition_evaluator(_data_file(4, 1))
    assert evaluations == expected_evaluations


def test_partition_evaluator_cache_rejects_negative_size(table_v2: Table) -> None:
    planner = ManifestGroupPlanner(
        table_metadata=table_v2.metadata,
        io=table_v2.io,
        row_filter=GreaterThan("x", 5),
        options={TableProperties.PARTITION_FILTER_CACHE_MAX_SIZE: "-1"},
    )

    with pytest.raises(
        ValueError,
        match=f"{TableProperties.PARTITION_FILTER_CACHE_MAX_SIZE} must be a non-negative integer",
    ):
        planner._build_partition_evaluator(0)


def test_partition_evaluator_cache_is_thread_safe(table_v2: Table) -> None:
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=GreaterThan("x", 5))
    partition_evaluator = planner._build_partition_evaluator(0)
    data_files = [_data_file(file_number, file_number % 10) for file_number in range(1_000)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(partition_evaluator, data_files))

    assert results == [file_number % 10 > 5 for file_number in range(1_000)]
