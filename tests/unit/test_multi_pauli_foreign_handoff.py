"""Complete deprecated generic Pauli-child handoff and wire dispatch."""

import builtins
from collections import Counter
from functools import partial
from typing import Any

import cirq
import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.multi_pauli_catalog import ControlledPauliExample
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.control_symbols import multi_control_wire_symbol
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import SymbolClass, register_score_data
from flagquantum.drawer.rotation_symbols import rotation_wire_symbol
from flagquantum.ecosystem.cirq._gate_metadata import cirq_gate_wire_symbol
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
GATES = (
    cirq.X,
    cirq.Y,
    cirq.Z,
    cirq.S,
    cirq.rx(0.3),
    cirq.H,
    cirq.ZPowGate(exponent=0.37, global_shift=0.29),
    cirq.MatrixGate(cirq.unitary(cirq.rz(0.21)) @ cirq.unitary(cirq.ry(-0.43))),
)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("values", ((), (0,), (1,), (1, 0, 1)))
def test_actual_generic_child_export_full_complex_adjoint_and_owned_append(
    gate: Any, values: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli
    from qualtran.cirq_interop import cirq_gate_to_bloq

    with pytest.warns(DeprecationWarning, match="MultiControlPauli"):
        source = MultiControlPauli(values, cirq_gate_to_bloq(gate))
    emitter = import_register_emitter(source)
    n = len(values) + 2
    logical = n - 1
    ports: dict[str, tuple[int, ...]] = {"target": (logical - 1,)}
    if values:
        ports["controls"] = tuple(range(len(values)))
    # This actual source Cirq entry uses an unmeasured controlled-unitary recipe.
    # Its gate counts differ from the separately accepted typed And/erase path.
    assert emitter.requirements() == (0, 0)
    assert emitter.binding_swaps == (0, 0)
    labels = torch.arange(1 << n, dtype=torch.float64)
    initial = torch.complex(labels + 1, labels.remainder(11) - 5)
    initial /= initial.norm()
    controlled = gate.controlled(control_values=values) if values else gate
    np.testing.assert_allclose(
        cirq.unitary(source), cirq.unitary(controlled), atol=2e-12, rtol=0
    )
    matrix = np.kron(cirq.unitary(controlled), np.eye(2))
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Captured generic control imported {name}")
        return original_import(name, *args, **kwargs)

    streams = tuple(emitter(ports, (), (), a) for a in (False, True))
    for adjoint, gates in enumerate(streams):
        expected = (matrix.conj().T if adjoint else matrix) @ initial.numpy()
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", owned_import)
            artifact = CircuitIR.from_json(CircuitIR(n, gates).to_json())
            circuit = Circuit(n, inputs=initial, dtype=initial.dtype).append(
                FieldOperation(n, artifact.instructions)
            )
            np.testing.assert_allclose(
                circuit.state().numpy().flatten(), expected, atol=2e-12, rtol=0
            )
    restored = Circuit(n, inputs=initial, dtype=initial.dtype)
    for gates in streams:
        restored.append(FieldOperation(n, gates))
    torch.testing.assert_close(restored.state().flatten(), initial, atol=2e-12, rtol=0)
    assert not Counter(g.name for g in streams[0])["measure"]


@pytest.mark.parametrize("values", ((), (0,), (1, 0), (1, 0, 1, 0)))
@pytest.mark.parametrize("gate", GATES)
def test_generic_child_control_and_target_wire_symbols_match_actual_source(
    values: tuple[int, ...], gate: Any
) -> None:
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli
    from qualtran.cirq_interop import cirq_gate_to_bloq

    with pytest.warns(DeprecationWarning, match="MultiControlPauli"):
        source = MultiControlPauli(values, cirq_gate_to_bloq(gate))
    signature = ControlledPauliExample("x", values).signature
    target = Register("q", quantum_bit_type())
    glyph, attributes = cirq_gate_wire_symbol(gate, type(signature)((target,)), target)

    def child_symbol(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        assert register == target and index == ()
        if gate == cirq.rx(0.3):
            return rotation_wire_symbol(step, register, index, axis="x", parameter=0.3)
        if isinstance(gate, (cirq.MatrixGate, cirq.ZPowGate)) and gate not in (
            cirq.Z,
            cirq.S,
        ):
            return partition_wire_symbol(step, register, index)
        return glyph, attributes

    step = RegisterStep(str(source))
    assert multi_control_wire_symbol(
        step,
        None,
        None,
        values=values,
        target_register=target,
        target_symbol=child_symbol,
    ) == ("Text", {"text": str(source), "fontsize": 10})
    for register in signature:
        indices = (
            tuple((i,) for i in range(len(values)))
            if register.name == "controls"
            else ((),)
        )
        foreign = source.signature.get_left(register.name)
        for index in indices:
            actual = multi_control_wire_symbol(
                step,
                register,
                index,
                values=values,
                target_register=target,
                target_symbol=child_symbol,
            )
            expected = source.wire_symbol(foreign, index)
            assert actual[0] == type(expected).__name__
            assert actual[1] == {key: getattr(expected, key) for key in actual[1]}


def test_symbolic_controls_retain_title_and_target_but_refuse_individual_polarities() -> (
    None
):
    import sympy
    from qualtran.bloqs.mcmt.multi_control_pauli import _ccpauli_symb

    source = _ccpauli_symb.make()
    width = sympy.Symbol("n")
    signature = ControlledPauliExample("x", width).signature
    target = Register("q", quantum_bit_type())

    def child_symbol(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        return "ModPlus", {}

    step = RegisterStep(str(source))
    assert multi_control_wire_symbol(
        step,
        None,
        None,
        values=None,
        target_register=target,
        target_symbol=child_symbol,
    ) == ("Text", {"text": str(source), "fontsize": 10})
    assert multi_control_wire_symbol(
        step,
        tuple(signature)[1],
        (),
        values=None,
        target_register=target,
        target_symbol=child_symbol,
    ) == ("ModPlus", {})
    with pytest.raises(ValueError, match="symbolic"):
        source.wire_symbol(source.signature.get_left("controls"), (0,))
    with pytest.raises(ValueError, match="symbolic"):
        multi_control_wire_symbol(
            step,
            tuple(signature)[0],
            (0,),
            values=None,
            target_register=target,
            target_symbol=child_symbol,
        )
    with pytest.raises(ValueError, match="element index"):
        multi_control_wire_symbol(
            step,
            tuple(signature)[0],
            None,
            values=(1,),
            target_register=target,
            target_symbol=child_symbol,
        )


def test_multi_control_glyph_dispatch_uses_normal_native_score_path() -> None:
    values = (1, 0, 1)
    signature = ControlledPauliExample("x", values).signature
    registers = tuple(signature)
    names = ("c0", "c1", "c2", "target")
    step = RegisterStep(
        "C^3X", names, names, input_registers=registers, output_registers=registers
    )
    score = layout_register_score(
        dict.fromkeys(names, 1),
        names,
        (step,),
        input_registers=registers,
        output_registers=registers,
    )
    from flagquantum.drawer.basis_symbols import pauli_basis_wire_symbol

    callback = partial(
        multi_control_wire_symbol,
        values=values,
        target_register=Register("q", quantum_bit_type()),
        target_symbol=partial(pauli_basis_wire_symbol, gate="x"),
    )
    data = register_score_data(score, register_wire_symbol=callback)
    glyphs = [item for item in data["soqs"] if item["seq_x"] == 0]
    assert [item["symb_cls"] for item in glyphs] == [
        "Circle",
        "Circle",
        "Circle",
        "ModPlus",
    ]
    assert [item["symb_attributes"] for item in glyphs[:3]] == [
        {"filled": True},
        {"filled": False},
        {"filled": True},
    ]
    assert data["vlines"][0]["label"] == {"text": "C^3X", "fontsize": 10}


@pytest.mark.parametrize("gate", (cirq.ResetChannel(), cirq.CNOT))
def test_original_nonunitary_and_multiwire_child_failure_boundaries(gate: Any) -> None:
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli
    from qualtran.cirq_interop import cirq_gate_to_bloq

    with pytest.warns(DeprecationWarning, match="MultiControlPauli"):
        source = MultiControlPauli((1, 0), cirq_gate_to_bloq(gate))
    error = TypeError if isinstance(gate, cirq.ResetChannel) else IndexError
    with pytest.raises(error):
        cirq.unitary(source)
    message = (
        "cannot be decomposed"
        if isinstance(gate, cirq.ResetChannel)
        else "too many values to unpack"
    )
    with pytest.raises(ValueError, match=message):
        import_register_emitter(source)


@pytest.mark.parametrize("gate", GATES)
def test_generic_child_measured_predicate_all_branches_and_clean_workspace(
    gate: Any,
) -> None:
    from itertools import product

    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli
    from qualtran.cirq_interop import cirq_gate_to_bloq

    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from flagquantum.compiler.registers.control.controlled_component import (
        controlled_component_ir,
    )
    from tests.unit.measurement_checks import forced_branch

    values = (1, 0, 1)
    with pytest.warns(DeprecationWarning, match="MultiControlPauli"):
        source = MultiControlPauli(values, cirq_gate_to_bloq(gate))
    child = import_register_emitter(source.target_bloq.controlled())
    assert child.requirements() == (0, 0)
    assert tuple(width for _, width in child.layout) == (1, 1)

    def emit(predicate: int) -> tuple[Instruction, ...]:
        return child(
            {child.layout[0][0]: (predicate,), child.layout[1][0]: (3,)}, (), (), False
        )

    gates = controlled_component_ir(
        ((0, 1, 2),), (ControlRegister(1, values),), emit, (4, 5), (0, 1)
    )
    assert Counter(g.name for g in gates)["measure"] == 2
    labels = torch.arange(32, dtype=torch.float64)
    amplitudes = torch.complex(labels + 1, labels.remainder(7) - 3)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    indices = (torch.arange(32) // 2 << 3) | (torch.arange(32) & 1)
    initial[indices] = amplitudes
    expected = torch.zeros_like(initial)
    expected[indices] = (
        torch.from_numpy(np.kron(cirq.unitary(source), np.eye(2)) @ amplitudes.numpy())
        / 2
    )
    for outcomes in product((0, 1), repeat=2):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
        assert torch.count_nonzero(actual[torch.arange(128) & 6 != 0]) == 0
