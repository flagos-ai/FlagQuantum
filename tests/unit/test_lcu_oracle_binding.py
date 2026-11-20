"""Replace PREPARE and SELECT through their existing native callback interfaces."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.gate_matrix import gate_matrix
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_lcu_external_file import (
    frozen_lcu_source_files as frozen_lcu_source_files,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _layout(registers: Any) -> tuple[tuple[str, int], ...]:
    return tuple((reg.name, int(reg.total_bits())) for reg in registers)


def _bind_children(
    source: Any,
) -> tuple[LCUBlockEncoding, CircuitRegisterEmitter, CircuitRegisterEmitter]:
    from qualtran._infra.controlled import CtrlSpec

    local = tuple(reg for reg in source.prepare.junk_registers if reg.side.value == 2)
    through = tuple(reg for reg in source.prepare.junk_registers if reg.side.value == 3)
    prepare = import_register_emitter(
        source.prepare, local_ports=tuple(reg.name for reg in local)
    )
    select = import_register_emitter(source.select)
    signal = BlackBoxPrepare(
        _layout(source.prepare.selection_registers),
        _layout(through),
        prepare,
        source.alpha,
        _layout(local),
    )
    block = SelectBlockEncoding(
        _layout(source.select.selection_registers),
        _layout(source.select.target_registers),
        select,
        signal,
    )
    value = source.control_val
    if value is None:
        return LCUBlockEncoding(block), prepare, select
    controlled = import_register_emitter(source.select.controlled(CtrlSpec(cvs=value)))

    def emit(
        activation: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert activation == value
        return controlled(
            {
                controlled.layout[0][0] if name == "control" else name: wires
                for name, wires in ports.items()
            },
            work,
            slots,
            adjoint,
        )

    return LCUBlockEncoding(block, value, emit), prepare, controlled


def _circuit(source: Any, adjoint: bool) -> tuple[CircuitIR, LCUBlockEncoding]:
    native, prepare, select = _bind_children(source)
    control = int(source.control_val is not None)
    address = native.ancilla_bitsize
    junk = sum(width for _, width in native.junk_layout)
    target = native.system_bitsize
    logical = control + address + junk + target
    pw, pb = prepare.requirements()
    sw, sb = select.requirements()
    local = native.select_block.signal_state.local_junk_width
    ir = (
        Circuit(logical + local + max(pw, sw))
        .append(
            native.make_on(
                control=tuple(range(control)),
                selection=tuple(range(control, control + address)),
                junk=tuple(range(control + address, control + address + junk)),
                target=tuple(range(control + address + junk, logical)),
                prepare_workspace=tuple(range(logical, logical + local + pw)),
                prepare_classical_bits=tuple(range(pb)),
                select_workspace=tuple(range(logical + local, logical + local + sw)),
                select_classical_bits=tuple(range(sb)),
                adjoint=adjoint,
            )
        )
        .to_ir()
    )
    return ir, native


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_hubbard_children_replace_lcu_consumers_and_clean_all_histories(
    packed: bool, control: int | None, adjoint: bool
) -> None:
    from attrs import evolve
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _black_box_lcu_block,
        _lcu_block,
    )

    source = evolve(
        (_black_box_lcu_block if packed else _lcu_block)(), control_val=control
    )
    ir, native = _circuit(source, adjoint)
    assert native.alpha == source.alpha == 72 and native.epsilon == source.epsilon == 0
    logical = sum(reg.total_bits() for reg in source.signature)
    work = tuple(range(logical, ir.n_wires))
    count = certify_erasure_histories(
        ir.instructions, n_wires=ir.n_wires, zero_wires=work
    )
    assert count == (86 if control is None else 89)
    assert len(work) == 8  # Complete exported-frame liveness reserves two extra names.
    counts = Counter(g.name for g in ir.instructions)
    assert counts["t"] + counts["tdg"] == (540 if control is None else 552)
    assert counts["cy"] == 18
    assert len(ir.instructions) == (
        2490 if control is None else 2542 if control == 0 else 2540
    )
    assert CircuitIR.from_json(ir.to_json()).to_json() == ir.to_json()
    # Capture is over: the existing native consumer emits from owned JSON only.
    prepare = native.select_block.signal_state.prepare
    assert isinstance(prepare, CircuitRegisterEmitter)
    assert all(isinstance(s, CircuitIR) for s in prepare.snapshots)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_select_keeps_nonidentity_native_signal_and_excludes_prepare_junk(
    packed: bool, adjoint: bool
) -> None:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _black_box_lcu_block,
        _lcu_block,
    )

    source = (_black_box_lcu_block if packed else _lcu_block)()
    native, prepare, select = _bind_children(source)
    block = native.select_block
    assert block.signal_state.prepare is prepare
    assert block.signal_state.junk_width == 2
    assert block.ancilla_bitsize + block.system_bitsize == 30
    work, slots = select.requirements()
    ir = (
        Circuit(30 + work)
        .append(
            block.make_on(
                selection=tuple(range(12)),
                target=tuple(range(12, 30)),
                workspace=tuple(range(30, 30 + work)),
                classical_bits=tuple(range(slots)),
                adjoint=adjoint,
            )
        )
        .to_ir()
    )
    assert (
        certify_erasure_histories(
            ir.instructions, n_wires=ir.n_wires, zero_wires=tuple(range(30, ir.n_wires))
        )
        == 40
    )
    assert sum(g.name == "cy" for g in ir.instructions) == 18
    assert len(ir.instructions) == 1098


@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_named_children_full_operator_and_complex_state_for_every_small_history(
    control: int | None, adjoint: bool
) -> None:
    import cirq
    from attrs import frozen
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.bloqs.basic_gates.hadamard import Hadamard
    from qualtran.bloqs.basic_gates.s_gate import SGate
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding as SourceLCU,
    )
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class Prepare(PrepareOracle):
        register: Any

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (self.register,)

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (Register("junk", QBit()),)

        @property
        def l1_norm_of_coeffs(self) -> float:
            return 3.0

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            selection, junk = soqs["selection"], soqs["junk"]
            wires = bb.split(selection)
            wires[0] = bb.add(Hadamard(), q=wires[0])
            wires[0] = bb.add(SGate(), q=wires[0])
            wires[0], junk = bb.add(CNOT(), ctrl=wires[0], target=junk)
            return {
                "selection": bb.join(wires, dtype=self.register.dtype),
                "junk": junk,
            }

    select = SelectPauliLCU(
        2,
        1,
        tuple(
            cirq.DensePauliString(
                word.removeprefix("-"), coefficient=-1 if word.startswith("-") else 1
            )
            for word in ("X", "Y", "-Z")
        ),
    )
    source = SourceLCU(
        select, Prepare(select.selection_registers[0]), control_val=control
    )
    ir, _ = _circuit(source, adjoint)
    logical = 4 + int(control is not None)
    clean = torch.zeros(1 << (ir.n_wires - logical), dtype=torch.complex128)
    clean[0] = 1
    original = source.adjoint() if adjoint else source
    matrix = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    data = torch.arange(1 << logical, dtype=torch.float64)
    coherent = torch.complex(data + 1, data % 5 - 2)
    coherent /= coherent.norm()
    count = certify_erasure_histories(
        ir.instructions,
        n_wires=ir.n_wires,
        zero_wires=tuple(range(logical, ir.n_wires)),
    )
    assert count <= 2
    # All basis columns include arbitrary logical junk; it is never a scratch bit.
    for state in (*torch.eye(1 << logical, dtype=torch.complex128), coherent):
        expected = torch.kron(matrix @ state, clean) * 2 ** (-count / 2)
        for history in product((0, 1), repeat=count):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), ir.instructions, history),
                expected,
                atol=5e-12,
                rtol=0,
            )


def test_emitter_admission_and_binding_are_owned_and_exact() -> None:
    from types import SimpleNamespace

    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register, Side, Signature
    from qualtran.bloqs.basic_gates.cnot import CNOT

    external = import_register_emitter(CNOT())
    assert external.layout == (("ctrl", 1), ("target", 1))
    gates = external({"target": (4,), "ctrl": (2,)}, (), (), False)
    assert gates == (Instruction("cx", (2, 4)),)
    for ports in (
        {"ctrl": (2,), "target": (2,)},
        {"q": (2, 4)},
        {"ctrl": (2,), "target": ()},
    ):
        with pytest.raises(ValueError):
            external(ports, (), (), False)
    with pytest.raises(ValueError, match="Boolean"):
        malformed_direction: dict[str, Any] = {"adjoint": 1}
        external({"ctrl": (2,), "target": (4,)}, (), (), **malformed_direction)
    with pytest.raises(ValueError, match="through ports"):
        import_register_emitter(
            SimpleNamespace(
                signature=Signature((Register("out", QBit(), side=Side.RIGHT),))
            )
        )
    with pytest.raises(ValueError, match="distinct"):
        replace(external, layout=(("q", 1), ("q", 1)))
    with pytest.raises(ValueError, match="concrete"):
        from sympy import Symbol

        replace(external, layout=(("q", Symbol("n")),))
    with pytest.raises(ValueError, match="directional"):
        malformed_snapshots: dict[str, Any] = {"snapshots": external.snapshots[:1]}
        replace(external, **malformed_snapshots)
    with pytest.raises(ValueError, match="nonnegative"):
        replace(external, binding_swaps=(-1, 0))
    with pytest.raises(ValueError, match="declared work"):
        external({"ctrl": (2,), "target": (4,)}, (5,), (), False)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_original_hubbard_composed_providers_full_sparse_complex_output(
    packed: bool, control: int | None, adjoint: bool
) -> None:
    import cirq
    from attrs import evolve
    from qualtran._infra.gate_with_registers import get_named_qubits
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _black_box_lcu_block,
        _lcu_block,
    )
    from qualtran.cirq_interop._bloq_to_cirq import BloqAsCirqGate

    from tests.unit.test_hubbard_hamiltonian_gqsp import (
        _modadd_leaf,
        _source_sparse_branch,
    )

    base = evolve(
        (_black_box_lcu_block if packed else _lcu_block)(), control_val=control
    )
    source = base.adjoint() if adjoint else base
    named = get_named_qubits(source.signature)
    visible = tuple(q for reg in source.signature for q in named[reg.name].flat)
    context = cirq.DecompositionContext(cirq.SimpleQubitManager(prefix="source_lcu"))
    source_ops = cirq.decompose(
        (source if isinstance(source, cirq.Gate) else BloqAsCirqGate(source)).on(
            *visible
        ),
        context=context,
        keep=lambda op: _modadd_leaf(op) is not None
        or (
            len(op.qubits) <= 3
            and (
                cirq.is_measurement(op)
                or cirq.has_unitary(op)
                or isinstance(op, cirq.ClassicallyControlledOperation)
                or isinstance(op.gate, cirq.ResetChannel)
            )
        ),
    )
    ir, _ = _circuit(base, adjoint)
    work = ir.n_wires - len(visible)
    state = {0: 1 / sqrt(3), 1 << 17: 1j / sqrt(3), (1 << 18) + (1 << 4): -1 / sqrt(3)}
    if control is not None:
        mask = 1 << (len(visible) - 1)
        state = {
            index | bit: value * phase
            for index, value in state.items()
            for bit, phase in ((0, 1 / sqrt(2)), (mask, 1j / sqrt(2)))
        }
    expected = _source_sparse_branch(source_ops, visible, state, 0)
    # Reuse the sparse interpreter; only local unsupported unitary gates become
    # their canonical native matrices. No dense whole-block contraction occurs.
    supported = {
        "h",
        "measure",
        "reset",
        "x",
        "cx",
        "ccx",
        "s",
        "sdg",
        "z",
        "cz",
        "t",
        "tdg",
        "phase",
        "cphase",
        "rz",
        "swap",
        "cswap",
        "any",
    }
    gates = tuple(
        (
            gate
            if gate.name in supported
            else Instruction(
                "any",
                gate.wires,
                matrix=gate_matrix(
                    gate, bsz=1, device="cpu", dtype=torch.complex128
                ).reshape(1 << len(gate.wires), 1 << len(gate.wires)),
                metadata=gate.metadata,
            )
        )
        for gate in ir.instructions
    )
    executable = replace(ir, instructions=gates)
    for seed in (7, 19):
        actual = execute_sparse_register(
            executable,
            {index << work: value for index, value in state.items()},
            seed=seed,
        )
        assert all(index % (1 << work) == 0 for index in actual)
        logical_output = {index >> work: value for index, value in actual.items()}
        # The two independent sparse interpreters prune at different numerical
        # floors. Compare the union, including amplitudes absent from either.
        assert (
            max(
                abs(logical_output.get(index, 0) - expected.get(index, 0))
                for index in logical_output.keys() | expected.keys()
            )
            < 9e-12
        )
