"""The numerical ModExp entry uses shared Programs within caller storage."""

import json
import sys
from typing import NoReturn

import pytest
import torch

import flagquantum.algorithms.primitives.modular_exponentiation as numerical
import flagquantum.compiler.registers.modular.modular_multiplication as flat
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _forbidden(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("The frozen flat modular multiplier must not execute")


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
def test_native_append_replaces_flat_path_with_shared_graph(
    strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in ("qualtran", "cirq", "kickmix", "quimb"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(flat, "controlled_modular_multiply_constant_ir", _forbidden)
    model = numerical.ModularExponentiation(2, 3, 1, 2)
    q, c = model.workspace_requirements()
    assert (q, c) == (13, 4)
    exponent, target = (5,), (3, 1)
    work = tuple(i for i in range(1, 17) if i not in (*exponent, *target))
    slots = (9, 2, 7, 4)
    bound = model.make_on(
        exponent=exponent, target=target, workspace=work, classical_bits=slots
    )
    width = 17
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    amplitudes = torch.tensor(
        [1 + 2j, -2 + 1j, 0.5 - 0.7j, 2 - 0.2j], dtype=torch.complex128
    )
    amplitudes /= amplitudes.norm()
    for control in (0, 1):
        for spectator in (0, 1):
            prefix = (spectator << 16) | (control << 11)
            amplitude = amplitudes[2 * control + spectator]
            initial[prefix] = amplitude
            result_bit = target[0] if control else target[1]
            expected[prefix | (1 << (16 - result_bit))] = amplitude
    circuit = DynamicCircuit(width, inputs=initial, dtype=torch.complex128)
    circuit.append(bound)
    ir = circuit.to_ir()
    assert CircuitIR.from_dict(json.loads(json.dumps(ir.to_dict()))) == ir
    assert all(0 not in gate.wires for gate in ir.instructions)
    assert {
        gate.metadata["classical_bit"]
        for gate in ir.instructions
        if gate.name == "measure"
    } == set(slots)
    result = run_dynamic(circuit, shots=2, seed=11, strategy=strategy)
    assert result.statistics["measurement_count"] == 96
    for state in result.final_states:
        torch.testing.assert_close(state.flatten(), expected, atol=2e-11, rtol=0)


@pytest.mark.ftqc_reference
def test_numerical_append_absolute_branches_match_original_tensor() -> None:
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    source = ModExp(2, 3, 1, 2)
    amplitudes = torch.tensor([1 + 2j, -2 + 1j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    source_output = torch.as_tensor(source.tensor_contract()) @ amplitudes
    model = numerical.ModularExponentiation(2, 3, 1, 2)
    bound = model.make_on(
        exponent=(0,),
        target=(1, 2),
        workspace=tuple(range(3, 16)),
        classical_bits=(13, 8, 3, 21),
    )
    ir = DynamicCircuit(16).append(bound).to_ir()
    measurements = sum(g.name == "measure" for g in ir.instructions)
    assert measurements == 48
    initial = {i << 15: complex(value) for i, value in enumerate(amplitudes)}
    expected = {
        i << 13: complex(value) * 2 ** (-measurements / 2)
        for i, value in enumerate(source_output)
        if abs(value) > 1e-20
    }
    for history in (
        (0,) * measurements,
        (1,) * measurements,
        tuple(i % 2 for i in range(measurements)),
    ):
        actual = _branch(ir.instructions, 16, history, initial.copy())
        assert {k: v for k, v in actual.items() if abs(v) > 1e-20} == pytest.approx(
            expected, abs=3e-19, rel=0
        )
