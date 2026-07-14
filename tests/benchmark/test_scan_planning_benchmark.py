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
"""Benchmark residual evaluation with a high-cardinality irrelevant partition field.

Every file has a unique unreferenced partition-hash value. The repeated case
measures reuse by relevant partition values, while the unique case forces misses.

Run with:
    uv run pytest tests/benchmark/test_scan_planning_benchmark.py -v -s -m benchmark
"""

from __future__ import annotations

import statistics
import timeit

import pytest

from pyiceberg.expressions import And, BooleanExpression, EqualTo, GreaterThanOrEqual, LessThanOrEqual, Or
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat, ManifestEntry, ManifestEntryStatus
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.table import ManifestGroupPlanner, Table
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record


def _combined_filter() -> BooleanExpression:
    branches: list[BooleanExpression] = []
    for source in range(11):
        branch: BooleanExpression = GreaterThanOrEqual("x", 0)
        for predicate in (
            LessThanOrEqual("x", 6),
            EqualTo("y", source),
            EqualTo("y", source + 1),
            EqualTo("y", source + 2),
            EqualTo("y", source + 3),
        ):
            branch = And(branch, predicate)
        branches.append(branch)

    combined = branches[0]
    for branch in branches[1:]:
        combined = Or(combined, branch)
    return combined


def _manifest_entry(file_number: int, relevant_partition: int) -> ManifestEntry:
    data_file = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=f"s3://bucket/data-{file_number}.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(relevant_partition, file_number),
        record_count=1,
        file_size_in_bytes=1,
    )
    data_file.spec_id = 0
    return ManifestEntry.from_args(
        status=ManifestEntryStatus.ADDED,
        snapshot_id=1,
        sequence_number=1,
        file_sequence_number=1,
        data_file=data_file,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "num_relevant_partitions",
    [7, 2_000],
    ids=["repeated-relevant-partitions", "unique-relevant-partitions"],
)
def test_residual_planning(table_v2: Table, monkeypatch: pytest.MonkeyPatch, num_relevant_partitions: int) -> None:
    num_files = 2_000
    entries = [_manifest_entry(file_number, file_number % num_relevant_partitions) for file_number in range(num_files)]
    spec = PartitionSpec(
        PartitionField(1, 1000, IdentityTransform(), "x"),
        PartitionField(3, 1001, IdentityTransform(), "partition_hash"),
        spec_id=0,
    )
    metadata = table_v2.metadata.model_copy(update={"partition_specs": [spec]})
    planner = ManifestGroupPlanner(
        table_metadata=metadata,
        io=table_v2.io,
        row_filter=_combined_filter(),
    )

    monkeypatch.setattr(planner, "plan_manifest_entries", lambda _: iter([entries]))

    timings = timeit.repeat(lambda: list(planner.plan_files([])), number=1, repeat=3)

    assert len(list(planner.plan_files([]))) == num_files
    print(
        f"Planned {num_files} files across {num_relevant_partitions} relevant partitions in "
        f"{statistics.mean(timings):.3f}s (best: {min(timings):.3f}s)"
    )
