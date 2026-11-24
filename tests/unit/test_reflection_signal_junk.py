"""Compare retained signal environments with the original physical reflection."""

from importlib.metadata import version
from itertools import product
from math import pi

import pytest

from flagquantum.compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from flagquantum.compiler.registers.primitives.global_phase_component import (
    scalar_phase_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_default_linear_combination_oracles import bind

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def reference_runtime() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")


@pytest.mark.parametrize("weights", ((0.5, 0.5), (0.25, 0.75)))
def test_alias_reflection_retains_signal_environment_and_restores_joint_state(
    weights: tuple[float, float],
) -> None:
    import cirq
    from qualtran import Adjoint
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )

    from flagquantum.compiler.formats.legacy_alias_preparation_factories import (
        alias_preparation_factories,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        resolve_component_program,
    )
    from tests.unit.test_hubbard_hamiltonian_gqsp import _source_sparse_branch
    from tests.unit.test_legacy_arithmetic_factories import _library
    from tests.unit.test_sparse_chemistry_prepare import _branch

    prepare = StatePreparationAliasSampling.from_probabilities(weights, precision=0.4)
    assert prepare.mu == 1
    pair = tuple(
        resolve_component_program(_library(child, 0), 0, alias_preparation_factories())
        for child in (prepare, Adjoint(prepare))
    )
    names = tuple("signal_" + reg.name for reg in prepare.junk_registers)
    record = prepared_reflection_program(
        pair[0],
        pair[1],
        ("selection",),
        global_phase=-1,
        scalar_phase=scalar_phase_program(pi),
        signal_junk_names=names,
    )
    assert not verify_program_record(record)
    bound, logical = bind(record)

    class RecordingManager(cirq.SimpleQubitManager):
        def __init__(self) -> None:
            super().__init__()
            self.allocations: list[list[cirq.Qid]] = []

        def qalloc(self, n: int, dim: int = 2) -> list[cirq.Qid]:
            allocated = super().qalloc(n, dim)
            self.allocations.append(allocated)
            return allocated

    source = ReflectionUsingPrepare(prepare, global_phase=-1)
    named = get_named_qubits(source.signature)
    manager = RecordingManager()
    # Force the physical root, rather than treating its projected tensor as a
    # unitary on selection alone. Source qfree is not a cleanup certificate.
    body = cirq.decompose_once(
        source.on_registers(**named), context=cirq.DecompositionContext(manager)
    )
    signal = tuple(q for group in manager.allocations[1:] for q in group)
    assert len(signal) == sum(reg.total_bits() for reg in prepare.junk_registers)
    visible = (*tuple(named["selection"].flat), *signal)
    assert len(visible) == logical
    ops = list(
        cirq.decompose(
            body,
            keep=lambda op: cirq.is_measurement(op)
            or isinstance(op, cirq.ClassicallyControlledOperation)
            or isinstance(op.gate, cirq.ResetChannel)
            or (len(op.qubits) <= 3 and cirq.has_unitary(op)),
        )
    )
    logical_input = {0: 0.6 + 0j, 1 << (logical - 1): 0.8j}
    expected = _source_sparse_branch(ops, visible, logical_input, 0)
    stride = 1 << (bound.circuit.n_wires - logical)
    initial = {key * stride: value for key, value in logical_input.items()}
    measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert measurements <= 8
    for history in product((0, 1), repeat=measurements):
        actual = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            history,
            initial,
            normalize_measurements=True,
        )
        assert all(key % stride == 0 for key in actual)
        projected = {key // stride: value for key, value in actual.items()}
        for key in expected.keys() | projected.keys():
            assert projected.get(key, 0) == pytest.approx(
                expected.get(key, 0), abs=2e-10
            )
        restored = _branch(
            bound.circuit.instructions,
            bound.circuit.n_wires,
            history,
            actual,
            normalize_measurements=True,
        )
        for key in restored.keys() | initial.keys():
            assert restored.get(key, 0) == pytest.approx(initial.get(key, 0), abs=2e-10)
    assert (
        sum(
            abs(value) ** 2
            for key, value in expected.items()
            if key % (1 << len(signal))
        )
        > 0.1
    )
