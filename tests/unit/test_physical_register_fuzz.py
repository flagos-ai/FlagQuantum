"""Compare complete original fuzz paths and expose later-chunk validation gaps."""

import json
import random
from pathlib import Path
from typing import cast

import pytest

from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.testing._physical_register_fuzz import (
    assert_fuzz_testing_acts_like,
    indentation_fix,
)

pytestmark = pytest.mark.unit
_CASES = json.loads(
    (Path(__file__).with_name("data") / "physical_register_fuzz.json").read_text()
)


@pytest.mark.parametrize("case", _CASES)
def test_complete_original_specification_and_diagnostic_paths(
    case: dict[str, object],
) -> None:
    samples = cast(list[dict[str, int]] | None, case.get("samples"))
    position = 0

    def sampler() -> dict[str, int]:
        nonlocal position
        assert samples is not None
        value = samples[position % len(samples)]
        position += 1
        return value

    state = random.getstate()
    random.seed(71)
    try:
        try:
            assert_fuzz_testing_acts_like(
                read_kmx_circuit(str(case["circuit"])),
                str(case["code"]),
                shots=cast(int, case["shots"]),
                context=cast(dict[str, object] | None, case.get("context")),
                input_sampler=sampler if samples is not None else None,
                ignore_phase=bool(case.get("ignore_phase", False)),
            )
            result = "passed"
            exception = None
        except (AssertionError, KeyError, NameError) as error:
            exception, result = type(error).__name__, str(error)
        assert (exception, result) == (case.get("exception"), case["result"])
    finally:
        random.setstate(state)


@pytest.mark.parametrize("shots", [257, 512, 513])
def test_native_rejects_later_faults_missed_by_unchanged_source(shots: int) -> None:
    circuit = read_kmx_circuit('REGISTER r0 "target"\nAPPEND_TO_REGISTER q0 r0\nX q0')
    position = 0

    def sampler() -> dict[str, int]:
        nonlocal position
        value = int(position >= 256)
        position += 1
        return {"target": value}

    with pytest.raises(AssertionError, match="'target' = 1 -> 0 != 1"):
        assert_fuzz_testing_acts_like(
            circuit, "target = 1", shots=shots, input_sampler=sampler
        )
    assert position == min(shots, 512)


def test_unnamed_ports_and_native_identity_diagnostics() -> None:
    with pytest.raises(AssertionError, match="Unnamed register OperandIdentity"):
        assert_fuzz_testing_acts_like(
            read_kmx_circuit("APPEND_TO_REGISTER q0 r0"), "", shots=1
        )


@pytest.mark.parametrize(
    "code", ["", " \t", "\r\n\tpass", "pass", "\n\rpass", "\n    pass"]
)
def test_complete_indentation_state_machine(code: str) -> None:
    result = indentation_fix(code)
    assert result.endswith(
        ("if True:\n" if code in (" \t", "\r\n\tpass", "\n    pass") else "") + code
    )


def test_callback_failures_propagate_and_later_successful_shots_are_validated() -> None:
    circuit = read_kmx_circuit('REGISTER r0 "target"\nAPPEND_TO_REGISTER q0 r0\nX q0')

    def failing_sample() -> dict[str, int]:
        raise RuntimeError("sample stopped")

    with pytest.raises(RuntimeError, match="sample stopped"):
        assert_fuzz_testing_acts_like(
            circuit, "target ^= 1", shots=1, input_sampler=failing_sample
        )
    assert_fuzz_testing_acts_like(
        circuit, "target ^= 1", shots=513, input_sampler=lambda: {"target": 0}
    )
