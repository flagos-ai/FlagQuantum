"""Complete locked tensor report and native deadline-process task acceptance."""

import csv
import hashlib
import multiprocessing
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.core.ir.program import ProgramRecord
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from tests.unit.test_component_tensor_nodes import fixture, h_supplier, order
from tests.unit.test_development_tools import REFERENCE, load_source
from tools.development import tensor_report
from tools.development.process_deadlines import ProcessDeadlineExecutor
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def pipe_task(
    cxn: Connection,
    value: object = 0,
    delay: float = 0,
    fail: bool = False,
    send: bool = True,
) -> None:
    time.sleep(delay)
    if fail:
        raise RuntimeError("injected child failure")
    if send:
        cxn.send(value)
    cxn.close()


def never_exit(cxn: Connection) -> None:
    time.sleep(20)


def test_source_file_blob() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0089")
    raw = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == row["git_blob"]
    )


@pytest.mark.parametrize("max_workers", [1, 2])
def test_complete_original_queue_and_result_semantics(
    max_workers: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = load_source("tensor_report_card")
    # Both implementations use the same real fork primitives. The source file
    # remains untouched; this isolates startup-policy cost from queue semantics.
    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(
        original,
        "multiprocessing",
        types.SimpleNamespace(Pipe=context.Pipe, Process=context.Process),
    )
    expected = original.ExecuteWithTimeout(timeout=2, max_workers=max_workers)
    actual = ProcessDeadlineExecutor(
        timeout=2, max_workers=max_workers, start_method="fork"
    )
    source_kwargs: list[dict[str, object]] = []
    native_kwargs: list[dict[str, object]] = []
    for i in range(5):
        left: dict[str, object] = {"value": i}
        right: dict[str, object] = {"value": i}
        source_kwargs.append(left)
        native_kwargs.append(right)
        expected.submit(pipe_task, left)
        actual.submit(pipe_task, right)
    assert expected.work_to_be_done == actual.work_to_be_done == 5
    source_results = []
    native_results = []
    try:
        while expected.work_to_be_done:
            kwargs, result = expected.next_result()
            source_results.append((kwargs["value"], result))
        while actual.work_to_be_done:
            kwargs, result = actual.next_result()
            native_results.append((kwargs["value"], result))
        assert (
            sorted(source_results)
            == sorted(native_results)
            == [(i, i) for i in range(5)]
        )
        assert all("cxn" in kwargs for kwargs in source_kwargs + native_kwargs)
        assert actual.work_to_be_done == 0
        with pytest.raises(StopIteration):
            actual.next_result()
    finally:
        actual.close()
        for kwargs in source_kwargs:
            connection = kwargs["cxn"]
            assert isinstance(connection, Connection)
            connection.close()
        for task in expected.pending:
            task.p.terminate()
            task.p.join(timeout=2)
            task.recv.close()


@pytest.mark.parametrize("kwargs", [{"fail": True}, {"send": False}, {"value": None}])
def test_exit_failure_missing_payload_and_none(kwargs: dict[str, object]) -> None:
    with ProcessDeadlineExecutor(2, 1, start_method="fork") as executor:
        executor.submit(pipe_task, kwargs)
        _, result = executor.next_result()
        assert result is None
        assert executor.work_to_be_done == 0


def test_timeout_joins_child_and_context_cleans_early_exit() -> None:
    with ProcessDeadlineExecutor(0.05, 1, start_method="fork") as executor:
        executor.submit(never_exit, {})
        executor._fill_workers()
        task = executor.pending[0]
        start = time.monotonic()
        _, result = executor.next_result()
        assert result is None and time.monotonic() - start < 2
        assert task.process.exitcode is not None and not task.process.is_alive()
        assert task.recv.closed
    with (
        pytest.raises(RuntimeError, match="consumer"),
        ProcessDeadlineExecutor(10, 2, start_method="fork") as executor,
    ):
        for _ in range(3):
            executor.submit(never_exit, {})
        executor._fill_workers()
        tasks = list(executor.pending)
        assert executor.work_to_be_done == 3
        raise RuntimeError("consumer")
    assert executor.work_to_be_done == 0
    assert all(not task.process.is_alive() and task.recv.closed for task in tasks)
    executor.close()


def test_large_payload_drains_before_exit() -> None:
    payload = b"x" * (1 << 20)
    with ProcessDeadlineExecutor(3, 1, start_method="fork") as executor:
        executor.submit(pipe_task, {"value": payload})
        _, result = executor.next_result()
        assert result == payload


def test_platform_default_spawn_task(tmp_path: Path) -> None:
    script = tmp_path / "spawn_report.py"
    script.write_text(
        "from tests.unit.test_tensor_report import pipe_task\n"
        "from tools.development.process_deadlines import ProcessDeadlineExecutor\n"
        "if __name__ == '__main__':\n"
        "    with ProcessDeadlineExecutor(10, 1) as executor:\n"
        "        executor.submit(pipe_task, {'value': 'spawn'})\n"
        "        _, result = executor.next_result()\n"
        "        assert result == 'spawn', result\n"
    )
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=25,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "timeout,workers",
    [(0, 1), (-1, 1), (float("nan"), 1), (float("inf"), 1), (1, 0), (1, True)],
)
def test_invalid_executor_budget(timeout: float, workers: int) -> None:
    with pytest.raises(ValueError):
        ProcessDeadlineExecutor(timeout, workers)


def native_nodes(record: ProgramRecord) -> tuple[TensorNetworkNode, ...]:
    return component_tensor_nodes(
        record, order(record), h_supplier, torch.empty((), dtype=torch.complex128)
    )[0]


def receive_report(run: Callable[[Connection], None]) -> dict[str, object]:
    recv, send = multiprocessing.Pipe(duplex=False)
    try:
        run(send)
        result = recv.recv()
        assert isinstance(result, dict)
        return result
    finally:
        recv.close()
        send.close()


@pytest.mark.parametrize("width,shape", [(0, ()), (1, ()), (2, (2,)), (1, (0,))])
def test_original_tensor_report_with_existing_graph_and_nodes(
    width: int, shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.simulation.tensor import cbloq_to_quimb

    original = load_source("tensor_report_card")
    record, source = fixture(width, shape)
    nodes = native_nodes(record)
    upstream = cbloq_to_quimb(source.flatten())
    assert len(nodes) == len(upstream.tensors)
    for node, tensor in zip(nodes, upstream.tensors, strict=True):
        np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
    # Deterministic clock isolates exact field order/width from actual duration.
    times = iter(float(i) for i in range(12))
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    expected = receive_report(
        lambda cxn: original.report_on_tensors("h", "HGraph", source, cxn)
    )
    actual = receive_report(
        lambda cxn: tensor_report.report_on_tensors(
            "h",
            "HGraph",
            record,
            cxn,
            flatten=lambda value: value,
            build_nodes=native_nodes,
        )
    )
    assert actual == expected
    assert tuple(actual) == ("name", "cls", "flat_dur", "tn_dur", "width", "width_dur")
    assert isinstance(actual["width"], float) and actual["width"] >= 0


@pytest.mark.parametrize("failure", ["flatten", "build", "width"])
def test_source_errors_keep_completed_fields(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = load_source("tensor_report_card")
    record, source = fixture(1, ())

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("report-stage failure")

    def flatten(value: ProgramRecord) -> ProgramRecord:
        return value

    build = native_nodes
    if failure == "flatten":
        flatten = fail
        monkeypatch.setattr(type(source), "as_composite_bloq", fail)
    elif failure == "build":
        build = fail
        monkeypatch.setattr(original, "cbloq_to_quimb", fail)
    else:
        import quimb.tensor as qtn

        monkeypatch.setattr(qtn.TensorNetwork, "contraction_width", fail)
        monkeypatch.setattr(tensor_report, "tensor_contraction_width", fail)
    times = iter(float(i) for i in range(12))
    monkeypatch.setattr(time, "perf_counter", lambda: next(times))
    expected = receive_report(
        lambda cxn: original.report_on_tensors("h", "HGraph", source, cxn)
    )
    actual = receive_report(
        lambda cxn: tensor_report.report_on_tensors(
            "h", "HGraph", record, cxn, flatten=flatten, build_nodes=build
        )
    )
    assert actual == expected
    assert actual["err"] == "report-stage failure"


def original_hang_probe(case: str) -> None:
    original = load_source("tensor_report_card")
    context = multiprocessing.get_context("fork")
    setattr(
        original,
        "multiprocessing",
        types.SimpleNamespace(Pipe=context.Pipe, Process=context.Process),
    )
    executor = original.ExecuteWithTimeout(timeout=0.05, max_workers=1)
    if case == "missing-payload":
        executor.submit(pipe_task, {"send": False})
    executor.next_result()


@pytest.mark.parametrize("case", ["empty", "missing-payload"])
def test_original_hang_boundaries_are_reproduced_and_bounded(case: str) -> None:
    process = multiprocessing.get_context("fork").Process(
        target=original_hang_probe, args=(case,)
    )
    process.start()
    try:
        process.join(timeout=0.3)
        assert (
            process.is_alive()
        ), "Expected the untouched source boundary to remain blocking"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
    assert not process.is_alive()


def flatten_record(record: ProgramRecord) -> ProgramRecord:
    from flagquantum.compiler.registers.component_tensor_expansion import (
        expand_components_for_tensor,
    )

    return expand_components_for_tensor(
        record, lambda name: name == "h", full_flatten=True
    )[0]


def test_report_uses_existing_compiler_and_simulation_in_worker() -> None:
    from functools import partial

    record, _ = fixture(2, (2,))
    task = partial(
        tensor_report.report_on_tensors,
        flatten=flatten_record,
        build_nodes=native_nodes,
    )
    with ProcessDeadlineExecutor(5, 1, start_method="fork") as executor:
        executor.submit(task, {"name": "h", "cls_name": "HGraph", "component": record})
        _, report = executor.next_result()
    assert isinstance(report, dict)
    assert "err" not in report
    assert report["name"] == "h" and report["cls"] == "HGraph"
    assert all(report[key] >= 0 for key in ("flat_dur", "tn_dur", "width", "width_dur"))


def test_startup_failure_keeps_task_and_closes_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with ProcessDeadlineExecutor(2, 1, start_method="fork") as executor:
        process_type = multiprocessing.get_context("fork").Process
        start = process_type.start

        def fail_start(self: Any) -> None:
            raise RuntimeError("start failure")

        monkeypatch.setattr(process_type, "start", fail_start)
        arguments: dict[str, object] = {"value": 7}
        executor.submit(pipe_task, arguments)
        with pytest.raises(RuntimeError, match="start failure"):
            executor.next_result()
        assert executor.work_to_be_done == 1 and not executor.pending
        connection = arguments["cxn"]
        assert isinstance(connection, Connection) and connection.closed
        monkeypatch.setattr(process_type, "start", start)
        _, result = executor.next_result()
        assert result == 7 and executor.work_to_be_done == 0
