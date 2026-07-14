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

import pytest

import pyiceberg.table as table_module
from pyiceberg.expressions import BooleanExpression, GreaterThan
from pyiceberg.io import FileIO
from pyiceberg.manifest import DataFile, FileFormat, ManifestContent, ManifestEntry, ManifestFile
from pyiceberg.schema import Schema
from pyiceberg.table import ManifestGroupPlanner, Table
from pyiceberg.typedef import Record, StructProtocol


def _data_file(file_number: int, partition_value: int) -> DataFile:
    return DataFile.from_args(
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(partition_value),
        record_count=100,
        file_size_in_bytes=1,
    )


def _manifest_file(file_number: int) -> ManifestFile:
    return ManifestFile.from_args(
        manifest_path=f"s3://bucket/manifest-{file_number}.avro",
        manifest_length=1,
        partition_spec_id=0,
        content=ManifestContent.DATA,
        sequence_number=1,
        min_sequence_number=1,
        added_snapshot_id=1,
    )


def test_partition_evaluator_reuses_instance_per_manifest_callable(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator_calls: list[list[int]] = []

    def counting_expression_evaluator(
        schema: Schema, unbound: BooleanExpression, case_sensitive: bool
    ) -> Callable[[StructProtocol], bool]:
        calls: list[int] = []
        evaluator_calls.append(calls)

        def evaluate(struct: StructProtocol) -> bool:
            calls.append(struct[0])
            return True

        return evaluate

    monkeypatch.setattr(table_module, "expression_evaluator", counting_expression_evaluator)
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=GreaterThan("x", 5))
    first_file = _data_file(1, 1)
    second_file = _data_file(2, 10)

    partition_evaluator_factory = planner._build_partition_evaluator_factory(0)
    first_callable = partition_evaluator_factory()
    assert not evaluator_calls
    assert first_callable(first_file)
    assert first_callable(second_file)

    second_callable = partition_evaluator_factory()
    assert len(evaluator_calls) == 1
    assert second_callable(first_file)

    assert evaluator_calls == [[1, 10], [1]]


def test_manifest_group_planner_creates_partition_evaluator_per_manifest(
    table_v2: Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=GreaterThan("x", 5))
    built_factories: list[int] = []
    built_evaluators: list[Callable[[DataFile], bool]] = []
    opened_evaluators: list[Callable[[DataFile], bool]] = []

    def build_partition_evaluator_factory(spec_id: int) -> Callable[[], Callable[[DataFile], bool]]:
        built_factories.append(spec_id)

        def partition_evaluator_factory() -> Callable[[DataFile], bool]:
            def partition_evaluator(data_file: DataFile) -> bool:
                return True

            built_evaluators.append(partition_evaluator)
            return partition_evaluator

        return partition_evaluator_factory

    def open_manifest(
        io: FileIO,
        manifest: ManifestFile,
        partition_evaluator: Callable[[DataFile], bool],
        metrics_evaluator: Callable[[DataFile], bool],
    ) -> list[ManifestEntry]:
        opened_evaluators.append(partition_evaluator)
        return []

    monkeypatch.setattr(planner, "_build_manifest_evaluator", lambda _: lambda _: True)
    monkeypatch.setattr(planner, "_build_partition_evaluator_factory", build_partition_evaluator_factory)
    monkeypatch.setattr(table_module, "_open_manifest", open_manifest)

    list(planner.plan_manifest_entries([_manifest_file(1), _manifest_file(2)]))

    assert built_factories == [0]
    assert len(built_evaluators) == 2
    assert {id(evaluator) for evaluator in opened_evaluators} == {id(evaluator) for evaluator in built_evaluators}


def test_reused_partition_evaluator_replaces_file_state(table_v2: Table) -> None:
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=GreaterThan("x", 5))
    partition_evaluator = planner._build_partition_evaluator_factory(0)()

    assert not partition_evaluator(_data_file(1, 1))
    assert partition_evaluator(_data_file(2, 10))
    assert not partition_evaluator(_data_file(3, 2))
