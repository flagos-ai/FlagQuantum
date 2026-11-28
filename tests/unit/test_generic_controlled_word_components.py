"""Preserve generic controlled word calls, complete operators and raw histories."""

from collections import Counter
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.registers.primitives.controlled_swap_component import (
    positive_controlled_swap_program,
)
from flagquantum.compiler.registers.primitives.controlled_z_word_component import (
    positive_controlled_z_word_program,
)
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_momentum_source_declarations import frozen_source as frozen_source
from tests.unit.test_multi_control_component import MeasurementSlots, atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_generic_controls(frozen_source: None) -> None:
    import qualtran

    root = Path(qualtran.__file__).parent
    for path, blob in (
        ("_infra/controlled.py", "6fabe760e793f1b39f825d76f80e92b6e5fa5c28"),
        ("bloqs/basic_gates/swap.py", "5c2481f7a1ee0a592d1192999ddf7e6dba8facc3"),
        ("bloqs/mcmt/specialized_ctrl.py", "98b64995010da5863218dbf93b39bbffbd600a2f"),
        (
            "bloqs/mcmt/controlled_via_and.py",
            "3947ddf86be37f4ff3ba58fcc316a8a64922f84f",
        ),
        ("bloqs/chemistry/black_boxes.py", "f9670a2ebaf2fe0973323cf7b532c88142970902"),
    ):
        raw = (root / path).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("kind", ("swap", "phase_one", "phase_two"))
def test_complete_generic_word_controls(
    frozen_source: None, width: int, kind: str
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs

    values = (1,) if kind == "phase_one" else (1, 0)
    source = Controlled(
        CSwap(width) if kind == "swap" else ApplyControlledZs(values, width), CtrlSpec()
    )
    record = (
        positive_controlled_swap_program(width)
        if kind == "swap"
        else positive_controlled_z_word_program(width, values)
    )
    assert verify_program_record(record) == ()
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=8e-11, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)
    ports: dict[str, tuple[int, ...]] = {}
    offset = 0
    for register in source.signature:
        width = int(register.total_bits())
        ports[register.name] = tuple(range(offset, offset + width))
        offset += width
    gates = bound.circuit.instructions
    reference = reference_arithmetic_gates(
        source,
        ports,
        tuple(range(logical, bound.circuit.n_wires)),
        MeasurementSlots(gates),
        keep_pauli_leaves=True,
        preserve_components=True,
    )
    assert_source(gates, reference)
    assert Counter(g.name for g in gates) == Counter(g.name for g in reference)


@pytest.mark.parametrize("kind", ("swap", "phase"))
@pytest.mark.parametrize("strategy", ("trajectory", "batched"))
def test_generic_word_append_preserves_joint_spectator_and_clean_work(
    frozen_source: None, kind: str, strategy: str
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.bloqs.chemistry.black_boxes import ApplyControlledZs

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    source = Controlled(
        CSwap(2) if kind == "swap" else ApplyControlledZs((1, 0), 2), CtrlSpec()
    )
    record = (
        positive_controlled_swap_program(2)
        if kind == "swap"
        else positive_controlled_z_word_program(2, (1, 0))
    )
    bound, logical = bind(record)
    work = bound.circuit.n_wires - logical
    amplitudes = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(amplitudes, amplitudes.flip(0))
    amplitudes = (amplitudes / amplitudes.norm()).reshape(1 << logical, 2)
    initial = torch.zeros(1 << (bound.circuit.n_wires + 1), dtype=torch.complex128)
    initial.reshape(1 << logical, 1 << work, 2)[:, 0, :] = amplitudes
    target = torch.zeros_like(initial)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    target.reshape(1 << logical, 1 << work, 2)[:, 0, :] = matrix @ amplitudes
    circuit = DynamicCircuit(
        bound.circuit.n_wires + 1, inputs=initial, dtype=torch.complex128
    )
    circuit.append(FieldOperation(circuit.n_qubits, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=47, strategy=strategy)
    for state in result.final_states:
        torch.testing.assert_close(state.reshape(-1), target, atol=8e-11, rtol=0)
