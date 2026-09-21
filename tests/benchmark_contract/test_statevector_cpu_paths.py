"""Contract tests for the path-decomposed CPU statevector benchmark.

These runs are deliberately small. They pin the payload contract, the reference
agreement, and the fail-closed behavior. Absolute latency belongs to a recorded
result payload, not to a test assertion.

The runtime-statistics assertions are chosen to hold both before and after the
CPU kernel work they measure: they pin which engine path a gate family takes,
not how fast that path currently is. A change that makes a rotation chain route
through the diagonal kernel is a routing defect and should fail here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flagquantum.benchmarking as runners
from flagquantum.benchmarking.statevector_cpu_paths import (
    CASES,
    SCHEMA,
    _direct_marginal_probabilities,
    build_marginal_circuit,
    cpu_kernel_switch_state,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

ROOT = Path(__file__).parents[2]
_CASE_NAMES = tuple(case.name for case in CASES)


def _small_run(**overrides: object) -> dict:
    arguments = {
        "n_wires": 4,
        "layers": 1,
        "batch_size": 1,
        "warmup": 0,
        "iterations": 2,
        "marginal_wires": 3,
    }
    arguments.update(overrides)
    return run_benchmark(**arguments)


def test_cpu_paths_runner_is_registered_and_discoverable():
    assert "statevector_cpu_paths" in runners.names()
    assert callable(runners.resolve("statevector_cpu_paths"))


def test_cpu_paths_payload_is_correct_deterministic_and_fail_closed():
    payload = _small_run()

    assert payload["schema_version"] == SCHEMA
    assert payload["artifact_class"] == "measured_local_run"
    assert payload["claim_evidence_type"] == "local_performance"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["release_gate_allowed"] is False
    assert payload["workload"]["cases"] == list(_CASE_NAMES)

    assert payload["all_cases_correct"] is True
    assert payload["all_cases_deterministic"] is True

    measured = {item["case"] for item in payload["cases"]}
    assert measured == set(_CASE_NAMES)
    for item in payload["cases"]:
        assert item["correctness"]["passed"] is True, (
            f"{item['case']} disagreed with its reference by "
            f"{item['correctness']['max_abs_error']}"
        )
        assert item["sample_count"] == 2
        assert len(item["samples_seconds"]) == 2
        assert item["median_seconds"] > 0
        assert item["p95_seconds"] >= item["median_seconds"]
        assert item["deterministic_across_repeats"] is True
        # The stability gate is a statement about this host, so the contract is
        # the gate's own arithmetic rather than a fixed verdict: a slow machine
        # is allowed to report that it could not resolve the ratio.
        gate = item["stability_gate"]
        assert gate["gated_statistic"] == "speedup_vs_reference"
        assert gate["observed_relative_standard_error"] == (
            item["speedup_timing"]["relative_standard_error"]
        )
        assert gate["passed"] is (
            gate["observed_relative_standard_error"]
            <= gate["max_relative_standard_error"]
            and item["sample_count"] >= gate["minimum_sample_count"]
        )


def test_cpu_paths_attributes_each_gate_family_to_its_engine_path():
    by_case = {item["case"]: item for item in _small_run()["cases"]}

    rotation = by_case["rotation_chain"]["runtime_statistics"]
    assert rotation["diagonal_elementwise_gates"] == 0
    assert rotation["statevector_apply_count"] >= 1

    # A single diagonal gate per wire still reaches the elementwise diagonal
    # kernel; fusing two or more on one wire does not, which is the gap the
    # diagonal-routing work targets. Assert the reachable case here and let the
    # fused case stay a measurement rather than an expectation.
    diagonal_single = by_case["diagonal_chain"]["runtime_statistics"]
    assert diagonal_single["diagonal_elementwise_gates"] >= 1

    cx_chain = by_case["cx_chain"]["runtime_statistics"]
    assert cx_chain["permutation_gates"] >= 1
    assert cx_chain["statevector_apply_count"] == 1
    assert cx_chain["fused_gate_regions"] == 0

    mixed = by_case["mixed_chain"]["runtime_statistics"]
    assert mixed["permutation_gates"] >= 1

    # ``triton_cx_sequence_regions`` is not asserted to be zero: it counts
    # ``_StatevectorCXSequenceStep`` entries in the built program, which is
    # device-independent, so it is non-zero on CPU as well. The Triton dispatch
    # is gated later on ``state.is_cuda`` and no counter in this dict witnesses
    # it. The benchmark therefore reports the field as a plan statistic and
    # makes no claim that a Triton kernel ran.
    for item in _small_run()["cases"]:
        statistics = item["runtime_statistics"]
        assert (
            statistics["statevector_apply_count"] >= 1
        ), f"{item['case']} reports no executed statevector step at all"


def test_fused_rotation_regions_are_attributed_and_reduce_to_one_apply_each():
    by_case = {item["case"]: item for item in _small_run(layers=2, n_wires=6)["cases"]}

    rotation = by_case["rotation_chain"]["runtime_statistics"]
    assert rotation["fused_gate_regions"] == 6
    assert rotation["fused_gate_count"] == 12
    assert rotation["statevector_apply_count"] == 6

    # Fusion swallows a diagonal chain today: it reports zero diagonal
    # elementwise gates and falls back to the dense kernel. This is recorded,
    # not asserted as desirable, so the benchmark can show the change.
    diagonal = by_case["diagonal_chain"]["runtime_statistics"]
    assert diagonal["fused_gate_regions"] == 6
    assert diagonal["diagonal_elementwise_gates"] == 0


def test_cpu_paths_can_select_a_subset_of_cases():
    payload = _small_run(cases=["cx_chain"])
    assert [item["case"] for item in payload["cases"]] == ["cx_chain"]


def test_cpu_paths_rejects_unknown_cases_and_unusable_dimensions():
    with pytest.raises(ValueError, match="unknown cases"):
        _small_run(cases=["not_a_case"])
    with pytest.raises(ValueError, match="iterations must be >= 2"):
        _small_run(iterations=1)
    with pytest.raises(ValueError, match="n_wires >= 3 required"):
        _small_run(n_wires=2)
    with pytest.raises(ValueError, match="marginal_wires must be between"):
        _small_run(marginal_wires=9)
    with pytest.raises(ValueError, match="threads must be >= 1"):
        _small_run(threads=0)
    with pytest.raises(ValueError, match="layers >= 1 required"):
        _small_run(layers=0)
    with pytest.raises(ValueError, match="batch_size >= 1 required"):
        _small_run(batch_size=0)


def test_marginal_reference_matches_the_state_reduction_it_replaces():
    """The reference must agree with an independent reshape-and-sum reduction."""
    circuit = build_marginal_circuit(n_wires=4, layers=2, batch_size=1, seed=4417)
    state = circuit.state()
    for wires in ((0,), (1, 2), (0, 1), (0, 2, 3)):
        reduced = _direct_marginal_probabilities(state, wires)
        assert reduced.shape == (1, 2 ** len(wires))
        assert float(reduced.sum()) == pytest.approx(1.0, abs=1e-5)

    every_wire = _direct_marginal_probabilities(state, (0, 1, 2, 3))
    assert every_wire.shape == (1, 16)
    torch.testing.assert_close(every_wire, torch.abs(state) ** 2)


def test_marginal_case_reference_is_a_probability_reduction_not_a_second_kernel():
    """The marginal reference must not re-enter the kernel it is checking."""
    payload = _small_run(cases=["marginal_probabilities"])
    case = payload["cases"][0]
    assert case["reference"] == (
        "resimulated_state_reduced_over_the_complement_of_the_requested_wires"
    )
    assert case["workload"]["marginal_wires"] == 3
    assert case["correctness"]["passed"] is True


def test_marginal_case_reference_reads_a_freshly_simulated_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """``circuit.state()`` is cached, so a reference that read it would be free.

    The candidate re-runs the program on every call. If the reference reduced
    the cached state instead, the ratio would compare a full simulation against
    a reduction of a state that already existed -- at 20 wires that is 4.4 s
    against 0.6 ms, and the candidate looks arbitrarily bad for no reason.
    Every state the reference reduces must therefore be one it just simulated,
    which is observable as a distinct tensor each time.
    """
    from flagquantum.benchmarking import statevector_cpu_paths as module

    reduced: list[torch.Tensor] = []
    original = module._direct_marginal_probabilities

    def recording(state: torch.Tensor, wires: tuple[int, ...]) -> torch.Tensor:
        reduced.append(state)
        return original(state, wires)

    monkeypatch.setattr(module, "_direct_marginal_probabilities", recording)
    module._run_marginal_case(
        module._CASE_BY_NAME["marginal_probabilities"],
        n_wires=4,
        layers=1,
        batch_size=1,
        seed=4417,
        warmup=0,
        iterations=2,
        marginal_wires=2,
    )

    assert len(reduced) >= 3, "the reference was not exercised by both measures"
    primed = reduced[0]
    for later in reduced[1:]:
        assert later is not primed, "the marginal reference reduced the cached state"


def test_marginal_case_does_not_depend_on_the_circuit_inputs_tensor():
    """``fq.run`` ignores ``circuit_param['inputs']``, so the case must not use it."""
    circuit = build_marginal_circuit(n_wires=4, layers=2, batch_size=1, seed=4417)
    assert circuit.circuit_param.get("inputs") is None


def test_rotation_case_fails_closed_when_a_kernel_result_is_corrupted(monkeypatch):
    """A case whose output stops matching its reference must fail the payload."""
    from flagquantum.benchmarking import statevector_cpu_paths as module

    real_sequential_reference = module.sequential_reference

    def corrupted(circuit: object) -> torch.Tensor:
        return real_sequential_reference(circuit) * 2.0

    monkeypatch.setattr(module, "sequential_reference", corrupted)
    payload = _small_run(cases=["rotation_chain"])
    assert payload["all_cases_correct"] is False
    assert payload["cases"][0]["correctness"]["passed"] is False


def test_cpu_paths_payload_satisfies_the_shared_result_contract(tmp_path):
    """The payload must survive the writer every benchmark result goes through."""
    from flagquantum.benchmarking.contract import validate_payload, write_json_atomic

    payload = _small_run(cases=["cx_chain"])
    validate_payload(payload)
    assert payload["runner"] == "statevector_cpu_paths"
    assert payload["schema"] == SCHEMA

    written = write_json_atomic(tmp_path / "nested" / "cpu_paths.json", payload)
    assert json.loads(written.read_text(encoding="utf-8")) == json.loads(
        json.dumps(payload)
    )


def test_cpu_paths_cli_writes_a_contract_payload_and_reports_its_verdict(tmp_path):
    output = tmp_path / "cpu_paths.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "flagquantum.benchmarking",
            "run",
            "statevector_cpu_paths",
            "--n-wires",
            "3",
            "--layers",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--marginal-wires",
            "2",
            "--json-output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["runner"] == "statevector_cpu_paths"
    assert payload["all_cases_correct"] is True
    assert payload["scalability_claim_allowed"] is False


def test_cpu_paths_runner_reports_a_nonzero_exit_code_when_a_case_is_wrong(
    monkeypatch, capsys
):
    from flagquantum.benchmarking import statevector_cpu_paths as module

    real_sequential_reference = module.sequential_reference
    monkeypatch.setattr(
        module,
        "sequential_reference",
        lambda circuit: real_sequential_reference(circuit) * 2.0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "statevector_cpu_paths",
            "--n-wires",
            "3",
            "--layers",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "2",
            "--marginal-wires",
            "2",
            "--cases",
            "rotation_chain",
        ],
    )
    # The verdict reaches the exit code instead of an exception, which is what
    # makes the runner usable from the CLI at all.
    assert module.main() == 2
    assert json.loads(capsys.readouterr().out)["all_cases_correct"] is False


def test_cpu_paths_payload_declares_its_evidence_limits():
    limits = _small_run()["evidence_limits"]
    assert "release gate" in limits
    assert "scalability" in limits


def test_cpu_paths_records_the_kernel_switches_its_numbers_depend_on(monkeypatch):
    """A payload is only comparable to another one under the same switches."""

    monkeypatch.delenv("FQ_CPU_CX_SEQUENCE_GATHER", raising=False)
    monkeypatch.delenv("FQ_CPU_SINGLE_WIRE_ELEMENTWISE", raising=False)
    flags = _small_run(cases=["rotation_chain"])["execution_flags"]

    assert set(flags) == {
        "FQ_CPU_CX_SEQUENCE_GATHER",
        "FQ_CPU_SINGLE_WIRE_ELEMENTWISE",
    }
    assert flags["FQ_CPU_CX_SEQUENCE_GATHER"] == {
        "effective": True,
        "source": "code_default",
        "raw": None,
    }
    assert flags["FQ_CPU_SINGLE_WIRE_ELEMENTWISE"] == {
        "effective": False,
        "source": "code_default",
        "raw": None,
    }

    monkeypatch.setenv("FQ_CPU_SINGLE_WIRE_ELEMENTWISE", "1")
    flags = _small_run(cases=["rotation_chain"])["execution_flags"]
    assert flags["FQ_CPU_SINGLE_WIRE_ELEMENTWISE"]["effective"] is True
    assert flags["FQ_CPU_SINGLE_WIRE_ELEMENTWISE"]["source"] == "environment"


def test_cpu_paths_switch_state_uses_the_same_reading_as_the_kernels(monkeypatch):
    """The payload must not disagree with the dispatch it describes."""

    from flagquantum.simulation.statevector.operations import (
        _cpu_cx_sequence_gather_enabled,
        _cpu_single_wire_elementwise_enabled,
    )

    for value in ("0", "1", "false", "true", "nonsense"):
        monkeypatch.setenv("FQ_CPU_CX_SEQUENCE_GATHER", value)
        monkeypatch.setenv("FQ_CPU_SINGLE_WIRE_ELEMENTWISE", value)
        flags = cpu_kernel_switch_state()
        assert flags["FQ_CPU_CX_SEQUENCE_GATHER"]["effective"] is (
            _cpu_cx_sequence_gather_enabled()
        )
        assert flags["FQ_CPU_SINGLE_WIRE_ELEMENTWISE"]["effective"] is (
            _cpu_single_wire_elementwise_enabled()
        )
