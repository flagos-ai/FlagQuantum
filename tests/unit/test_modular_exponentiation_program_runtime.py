"""Execute the restored ModExp Program through normal native append paths."""

import json
import sys
from copy import deepcopy

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_modular_exponentiation_factories import (
    modular_exponentiation_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_source_free_normal_append_program_copy_and_lowered_json(
    strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    library = _library(ModExp(2, 3, 1, 2), 0)
    for module in ("qualtran", "cirq", "kickmix", "quimb"):
        monkeypatch.setitem(sys.modules, module, None)
    record = resolve_component_program(library, 0, modular_exponentiation_factories())
    bound = bind_register_program(
        record,
        ((5,),),
        clean=tuple(wire for wire in range(1, 17) if wire != 5),
        borrowed=(0,),
        allow_workspace_outputs=True,
    )
    copied = bind_register_program(
        deepcopy(record),
        ((5,),),
        clean=tuple(wire for wire in range(1, 17) if wire != 5),
        borrowed=(0,),
        allow_workspace_outputs=True,
    )
    assert copied == bound
    assert (
        CircuitIR.from_dict(json.loads(json.dumps(bound.circuit.to_dict())))
        == bound.circuit
    )
    width = bound.circuit.n_wires
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    amplitudes = torch.tensor(
        [1 + 2j, -2 + 1j, 0.5 - 0.7j, 2 - 0.2j], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    first, last = bound.outputs[1]
    for control in (0, 1):
        for spectator in (0, 1):
            prefix = (spectator << (width - 1)) | (control << (width - 1 - 5))
            initial[prefix] = amplitudes[2 * control + spectator]
            result_bit = first if control else last
            expected[prefix | (1 << (width - 1 - result_bit))] = amplitudes[
                2 * control + spectator
            ]
    assert all(0 not in gate.wires for gate in bound.circuit.instructions)
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(width, bound.circuit.instructions))
    assert circuit.to_ir().instructions == bound.circuit.instructions
    result = run_dynamic(circuit, shots=2, seed=11, strategy=strategy)
    assert result.statistics["measurement_count"] == 96
    for state in result.final_states:
        torch.testing.assert_close(state.flatten(), expected, atol=2e-11, rtol=0)


def test_absolute_complex_branches_match_the_unmodified_source_tensor() -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    source = ModExp(2, 3, 1, 2)
    library = _library(source, 0)
    record = resolve_component_program(library, 0, modular_exponentiation_factories())
    bound = bind_register_program(
        record, ((0,),), clean=tuple(range(1, 16)), allow_workspace_outputs=True
    )
    width = bound.circuit.n_wires
    amplitudes = torch.tensor([1 + 2j, -2 + 1j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    expected = torch.as_tensor(source.tensor_contract()) @ amplitudes
    initial = {i << (width - 1): complex(value) for i, value in enumerate(amplitudes)}
    measurements = sum(g.name == "measure" for g in bound.circuit.instructions)
    target = {
        i << (width - 3): complex(value) * 2 ** (-measurements / 2)
        for i, value in enumerate(expected)
        if abs(value) > 1e-20
    }
    for history in (
        (0,) * measurements,
        (1,) * measurements,
        tuple(i % 2 for i in range(measurements)),
    ):
        actual = _branch(bound.circuit.instructions, width, history, initial.copy())
        assert {k: v for k, v in actual.items() if abs(v) > 1e-20} == pytest.approx(
            target, abs=3e-19, rel=0
        )
