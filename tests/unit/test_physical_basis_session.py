"""Compare complete interactive physical observation paths with frozen source."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

from flagquantum.compiler.artifacts.operand_identity import OperandIdentity, OperandKind
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.ecosystem._physical_basis_session import PhysicalBasisSession
from tests.unit.fixtures.physical_basis_session_cases import observe_sessions

pytestmark = pytest.mark.unit


def _q(index: int) -> OperandIdentity:
    return OperandIdentity(OperandKind.QUANTUM, index)


def _b(index: int) -> OperandIdentity:
    return OperandIdentity(OperandKind.CLASSICAL, index)


def _r(index: int) -> OperandIdentity:
    return OperandIdentity(OperandKind.REGISTER, index)


def test_complete_source_session_observations() -> None:
    expected = json.loads(
        (Path(__file__).with_name("data") / "physical_basis_session.json").read_text()
    )
    actual = observe_sessions(PhysicalBasisSession, read_kmx_circuit, _q, _b, _r)
    for name, value in expected.items():
        assert actual[name] == value, name
    assert actual.keys() == expected.keys()


@pytest.mark.parametrize("seed", [True, "bad", -1, 1 << 256])
def test_seed_admission(seed: object) -> None:
    error = TypeError if isinstance(seed, (bool, str)) else ValueError
    with pytest.raises(error):
        PhysicalBasisSession(0, seed=seed)


@pytest.mark.parametrize("size", [-1, 1.5, "3"])
def test_original_size_before_coercion(size: int) -> None:
    with pytest.raises((ValueError, TypeError)):
        PhysicalBasisSession(size)


def test_phase_failure_order_and_output_aliases() -> None:
    sim = PhysicalBasisSession(3, debug_print=False, seed=0)
    with pytest.raises(TypeError):
        sim.write_shot_phase(99, None)
    with pytest.raises(IndexError):
        sim.write_shot_phase(99, Fraction(1, 4))
    with pytest.raises(IndexError):
        sim.read_within_shot(_q(0), 3, out=int)
    with pytest.raises(ValueError):
        sim.read_within_shot([], -1, out=int)
    with pytest.raises(ValueError, match="counting"):
        sim.operation_statistics()
    with pytest.raises(TypeError):
        sim.write_across_shots(True, 1)
    with pytest.raises(IndexError):
        sim.read_across_shots("unknown")
    with pytest.raises(IndexError):
        sim.read_across_shots(_r(0))
    with pytest.raises(ValueError, match="np.ndarray"):
        sim.read_across_shots(_q(0), out=np.zeros(3, dtype=np.int64))
    with pytest.raises(ValueError, match="shape"):
        sim.write_across_shots(_q(0), np.zeros((1, 3), dtype=np.bool_))
    readonly = np.zeros(3, dtype=np.bool_)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        sim.read_across_shots(_q(0), out=readonly)


def test_debug_defaults_and_failure_retains_source_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sim = PhysicalBasisSession(257, seed=0, counting_operations=True)
    sim.do(
        read_kmx_circuit(
            "X q0\nBIT_STORE1 b0\nDEBUG_PRINT q0\nDEBUG_PRINT b0\nDEBUG_PRINT"
        )
    )
    assert capsys.readouterr().err.count("11 (phase=") == 2
    sim.debug_print = False
    sim.do(read_kmx_circuit("DEBUG_PRINT"))
    assert capsys.readouterr().err == ""

    def throwing(*args: object) -> None:
        raise RuntimeError("stop")

    sim.debug_print = throwing
    with pytest.raises(RuntimeError, match="stop"):
        sim.do(read_kmx_circuit("NEG\nX q1\nDEBUG_PRINT"))
    assert sim.read_shot_phase(0) == 0
    assert sim.read_across_shots(_q(1), out=int) == (1 << 256) - 1
    assert sim.operation_statistics()["NEG"] == 256
    with pytest.raises(TypeError, match="debug_print"):
        setattr(sim, "debug_print", 3)


@pytest.mark.parametrize(
    "mode",
    [
        "single-read",
        "multi-read",
        "single-write",
        "multi-write",
        "within-read",
        "within-write",
        "phase",
    ],
)
def test_source_empty_numpy_pointer_refusals(mode: str) -> None:
    sim = PhysicalBasisSession(0, debug_print=False, seed=0)
    with pytest.raises(IndexError):
        if mode == "single-read":
            sim.read_across_shots(_q(0))
        elif mode == "multi-read":
            sim.read_across_shots([_q(0)])
        elif mode == "single-write":
            sim.write_across_shots(_q(0), np.zeros(0, dtype=np.bool_))
        elif mode == "multi-write":
            sim.write_across_shots([], np.zeros((0, 0), dtype=np.bool_))
        elif mode == "within-read":
            sim.read_within_shot([], 0)
        elif mode == "within-write":
            sim.write_within_shot([], 0, np.zeros(0, dtype=np.bool_))
        else:
            sim.read_phase_flipped_across_shots()
    assert sim.read_across_shots([], out=int) == []
    assert sim.read_within_shot([], 99, out=int) == 0


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((3, 1), dtype=np.bool_),
        np.zeros(4, dtype=np.bool_),
        [],
        np.zeros(3, dtype=np.uint8),
    ],
)
def test_output_shapes_and_types(value: object) -> None:
    sim = PhysicalBasisSession(3, debug_print=False, seed=0)
    with pytest.raises(ValueError):
        sim.read_across_shots(_q(0), out=value)


@pytest.mark.parametrize(
    "value", [2, [1], [1, 8], np.zeros((2, 2), dtype=np.bool_), np.array([1, 2])]
)
def test_multiple_write_refusals(value: object) -> None:
    sim = PhysicalBasisSession(3, debug_print=False, seed=0)
    with pytest.raises(ValueError):
        sim.write_across_shots([_q(0), _b(0)], value)


def test_native_indices_validation_and_retained_aliases() -> None:
    sim = PhysicalBasisSession(
        np.int64(3), debug_print=False, seed=0, counting_operations=True
    )
    circuit = read_kmx_circuit('APPEND_TO_REGISTER q0 r0\nREGISTER r0 "data"\nX q0')
    sim.do(circuit)
    sim.use_same_registers_as(circuit)
    assert sim.read_across_shots("data", out=int) == [7]
    sim.clear_for_shot()
    sim.write_within_shot("data", np.int64(2), True)
    assert sim.read_within_shot("data", 2, out=int) == 1
    assert sim.read_within_shot(_q(0), 2, out=int) is True
    assert sim.read_across_shots([False, True], out=int) == [0, 7]
    for index in (-1, 3):
        with pytest.raises(IndexError):
            sim.operation_statistics(index)
        with pytest.raises(ValueError if index < 0 else IndexError):
            sim.read_shot_phase(index)
    for invalid_index in (1.5, 1 << 64):
        with pytest.raises(TypeError):
            sim.operation_statistics(invalid_index)
    with pytest.raises(ValueError):
        sim.write_across_shots(_q(0), "bad")
    with pytest.raises(ValueError):
        sim.write_within_shot(_q(0), 0, [True])
    with pytest.raises(ValueError):
        sim.write_within_shot([_q(0)], 0, np.zeros(2, dtype=np.bool_))
    with pytest.raises(IndexError):
        sim.write_within_shot(_q(0), 3, 1)
    with pytest.raises(TypeError):
        sim.read_within_shot(_q(0), "0")
    with pytest.raises(ValueError):
        sim.read_within_shot([], 1 << 64)
    assert PhysicalBasisSession(0, debug_print=False).batch_size == 0
