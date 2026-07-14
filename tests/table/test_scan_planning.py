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
from pyiceberg.expressions import AlwaysTrue, BooleanExpression, EqualTo
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat, ManifestEntry, ManifestEntryStatus
from pyiceberg.table import ManifestGroupPlanner, Table
from pyiceberg.typedef import Record


class _CountingResidualEvaluator:
    def __init__(self, marker: int) -> None:
        self.marker = marker
        self.calls: list[int] = []

    def residual_for(self, partition: Record) -> BooleanExpression:
        partition_value = partition[0]
        self.calls.append(partition_value)
        return EqualTo("x", self.marker * 10 + partition_value)


def _manifest_entry(file_number: int, spec_id: int, partition: int) -> ManifestEntry:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(partition),
        record_count=1,
        file_size_in_bytes=1,
    )
    data_file.spec_id = spec_id
    return ManifestEntry.from_args(
        status=ManifestEntryStatus.ADDED,
        snapshot_id=1,
        sequence_number=1,
        file_sequence_number=1,
        data_file=data_file,
    )


def test_manifest_group_planner_reuses_residuals_by_spec_and_partition(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        _manifest_entry(0, spec_id=0, partition=1),
        _manifest_entry(1, spec_id=0, partition=1),
        _manifest_entry(2, spec_id=1, partition=1),
        _manifest_entry(3, spec_id=1, partition=1),
        _manifest_entry(4, spec_id=0, partition=2),
    ]
    evaluators = {0: _CountingResidualEvaluator(0), 1: _CountingResidualEvaluator(1)}
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=AlwaysTrue())

    monkeypatch.setattr(planner, "plan_manifest_entries", lambda _: iter([entries]))
    monkeypatch.setattr(planner, "_build_residual_evaluator", lambda spec_id: evaluators[spec_id])

    tasks = list(planner.plan_files([]))

    assert evaluators[0].calls == [1, 2]
    assert evaluators[1].calls == [1]
    assert [task.residual for task in tasks] == [
        EqualTo("x", 1),
        EqualTo("x", 1),
        EqualTo("x", 11),
        EqualTo("x", 11),
        EqualTo("x", 2),
    ]


def test_manifest_group_planner_bounds_residual_cache(table_v2: Table, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        _manifest_entry(0, spec_id=0, partition=1),
        _manifest_entry(1, spec_id=0, partition=2),
        _manifest_entry(2, spec_id=0, partition=3),
        _manifest_entry(3, spec_id=0, partition=1),
    ]
    evaluator = _CountingResidualEvaluator(0)
    planner = ManifestGroupPlanner(table_metadata=table_v2.metadata, io=table_v2.io, row_filter=AlwaysTrue())

    monkeypatch.setattr(table_module, "_RESIDUAL_CACHE_MAX_SIZE", 2)
    monkeypatch.setattr(planner, "plan_manifest_entries", lambda _: iter([entries]))
    monkeypatch.setattr(planner, "_build_residual_evaluator", lambda _: evaluator)

    tasks = list(planner.plan_files([]))

    assert evaluator.calls == [1, 2, 3, 1]
    assert [task.residual for task in tasks] == [
        EqualTo("x", 1),
        EqualTo("x", 2),
        EqualTo("x", 3),
        EqualTo("x", 1),
    ]
