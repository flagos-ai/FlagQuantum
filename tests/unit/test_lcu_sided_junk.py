"""Preparation-local outputs survive SELECT and are consumed by the source adjoint."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_lcu_external_file import (
    frozen_lcu_source_files as frozen_lcu_source_files,
)
from tests.unit.test_lcu_oracle_binding import _bind_children, _circuit

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _canonical_fixture_stream(
    gates: tuple[Instruction, ...], fixed_ports: int
) -> tuple[Instruction, ...]:
    """Rename scratch and measurement versions without reordering any gate.

    The source exporter can retain idle allocation names; native stages can
    reuse a classical slot. Logical qubits stay fixed; private identities are renamed consistently.
    This fixture has only direct measurement records and conjunction feedback.
    """
    wires = {i: i for i in range(fixed_ports)}
    versions: dict[int, int] = {}
    events = 0
    result = []
    for gate in gates:
        operands = tuple(wires.setdefault(q, len(wires)) for q in gate.wires)
        metadata = dict(gate.metadata)
        assert metadata.keys() <= {"classical_bit", "conditions", "is_dynamic"}
        if "conditions" in metadata:
            metadata["conditions"] = tuple(
                (versions[bit], value) for bit, value in metadata["conditions"]
            )
        if gate.name == "measure":
            versions[metadata["classical_bit"]] = events
            metadata["classical_bit"] = events
            events += 1
        result.append(
            Instruction(
                gate.name, operands, gate.params, matrix=gate.matrix, metadata=metadata
            )
        )
    return tuple(result)


def _source(
    control: int | None = None,
    *,
    shaped: bool = True,
    left: bool = False,
    normalization: Any = 3.0,
    local_select: bool = False,
) -> Any:
    import cirq
    import numpy as np
    from attrs import frozen
    from qualtran import BQUInt, QBit, Register, Side
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, SGate
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.mcmt import And
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
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
            return (
                Register("through", QBit()),
                Register(
                    "local",
                    QBit(),
                    shape=(2,) if shaped else (),
                    side=Side.LEFT if left else Side.RIGHT,
                ),
            )

        @property
        def l1_norm_of_coeffs(self) -> Any:
            return normalization

        def build_composite_bloq(
            self, bb: Any, selection: Any, through: Any, **soqs: Any
        ) -> dict[str, Any]:
            wires = bb.split(selection)
            if left:
                # The original LCU excludes LEFT junk from its input signature.
                # Requiring the declared input must retain that source diagnostic.
                local = soqs["local"]
                if shaped:
                    for bit in local:
                        bb.free(bit)
                else:
                    bb.free(local)
                return {
                    "selection": bb.join(wires, dtype=self.register.dtype),
                    "through": through,
                }
            wires[0] = bb.add(Hadamard(), q=wires[0])
            wires[1] = bb.add(SGate(), q=wires[1])
            wires, flag = bb.add(And(), ctrl=wires)
            flag, through = bb.add(CNOT(), ctrl=flag, target=through)
            if shaped:
                copy = bb.allocate(dtype=QBit())
                wires[0], copy = bb.add(CNOT(), ctrl=wires[0], target=copy)
                local = np.asarray([flag, copy], dtype=object)
            else:
                local = flag
            return {
                "selection": bb.join(wires, dtype=self.register.dtype),
                "through": through,
                "local": local,
            }

    select = SelectPauliLCU(
        2, 1, tuple(cirq.DensePauliString(w) for w in ("X", "Y", "Z"))
    )

    @frozen
    class SelectLocal(SelectOracle):
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("local", QBit()),)

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("target", QBit()),)

        def build_composite_bloq(
            self, bb: Any, local: Any, target: Any
        ) -> dict[str, Any]:
            local, target = bb.add(CNOT(), ctrl=local, target=target)
            return {"local": local, "target": target}

    if local_select:
        return LCUBlockEncoding(
            SelectLocal(),
            Prepare(Register("selection", BQUInt(2, 3))),
            control_val=control,
        )
    return LCUBlockEncoding(
        select, Prepare(select.selection_registers[0]), control_val=control
    )


@pytest.mark.parametrize("shaped", (False, True))
@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_local_junk_complete_operator_each_history_and_independent_cost(
    shaped: bool, control: int | None, adjoint: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    base = _source(control, shaped=shaped)
    source = base.adjoint() if adjoint else base
    ir, native = _circuit(base, adjoint)
    logical = 4 + int(control is not None)
    local_width = 2 if shaped else 1
    work = tuple(range(logical, ir.n_wires))
    raw, swaps = _snapshot(
        source, tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
    )
    assert swaps == 0
    assert _canonical_fixture_stream(
        ir.instructions, logical
    ) == _canonical_fixture_stream(raw.instructions, logical)
    assert tuple(reg.name for reg in base.junk_registers) == ("through",)
    assert native.junk_layout == (("through", 1),)
    assert native.resource_bitsize == base.resource_bitsize == 1 + local_width
    assert native.select_block.signal_state.local_junk_width == local_width
    counts = Counter(g.name for g in ir.instructions)
    events = certify_erasure_histories(
        ir.instructions, n_wires=ir.n_wires, zero_wires=work
    )
    assert events == counts["measure"]
    assert (
        certify_erasure_histories(
            raw.instructions,
            n_wires=raw.n_wires,
            zero_wires=tuple(range(logical, raw.n_wires)),
        )
        == events
    )
    assert events == (2 if control is None else 3)
    clean = torch.zeros(1 << len(work), dtype=torch.complex128)
    clean[0] = 1
    expected_matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    grid = torch.arange(1 << logical, dtype=torch.float64)
    coherent = torch.complex(grid + 1, grid % 7 - 3)
    coherent /= coherent.norm()
    restored = CircuitIR.from_json(ir.to_json())
    assert restored.to_json() == ir.to_json()
    for state in (*torch.eye(1 << logical, dtype=torch.complex128), coherent):
        expected = torch.kron(expected_matrix @ state, clean) * 2 ** (-events / 2)
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), restored.instructions, history),
                expected,
                atol=8e-12,
                rtol=0,
            )
    select = (
        base.select
        if control is None
        else base.select.controlled(CtrlSpec(cvs=control))
    )
    select = select.adjoint() if adjoint else select

    def cost(operation: Any) -> ComponentCost:
        return ComponentCost(
            "Independent source",
            int(get_cost_value(operation, QubitCount())),
            GateCost(**get_cost_value(operation, QECGatesCost()).asdict()),
        )

    assert_source_model(
        native.resource_model(
            cost(select),
            cost(base.prepare),
            cost(base.prepare.adjoint()),
            adjoint=adjoint,
        ),
        source,
    )
    # Both columns are independently derived; emission is not a source model.
    declared = get_cost_value(source, QECGatesCost())
    assert events == declared.measurement
    assert counts["t"] + counts["tdg"] == declared.total_t_count()
    print(
        "local-resource",
        shaped,
        control,
        adjoint,
        ir.n_wires,
        len(ir.instructions),
        counts["t"] + counts["tdg"],
        events,
        int(get_cost_value(source, QubitCount())),
    )


def test_local_junk_lifetime_admission_and_source_left_diagnostic() -> None:
    base = _source()
    native, prepare, select = _bind_children(base)
    with pytest.raises(ValueError, match="through ports"):
        import_register_emitter(base.prepare)
    with pytest.raises(ValueError, match="RIGHT outputs"):
        import_register_emitter(base.prepare.adjoint(), local_ports=("local",))
    with pytest.raises(ValueError, match="distinct declared"):
        import_register_emitter(base.prepare, local_ports=("local", "local"))
    with pytest.raises(ValueError, match="through ports"):
        import_register_emitter(base.prepare, local_ports=("unknown",))
    with pytest.raises(ValueError, match="one-sided"):
        _snapshot(base.prepare, prepare.layout, local_ports=("selection",))
    wrong = tuple(
        (name, width + int(name == "local")) for name, width in prepare.layout
    )
    with pytest.raises(ValueError, match="Local port widths"):
        _snapshot(base.prepare, wrong, local_ports=("local",))
    args = dict(selection=(0, 1), junk=(2,), target=(3,), select_classical_bits=(0,))
    with pytest.raises(ValueError, match="persistent local junk"):
        native.make_on(**args)
    with pytest.raises(ValueError, match="cannot overlap"):
        native.make_on(**args, prepare_workspace=(4, 5), select_workspace=(5,))
    for source in (_source(left=True), _source(left=True).adjoint()):
        ports = tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
        with pytest.raises(KeyError, match="local"):
            _snapshot(source, ports)


def test_prepare_only_local_output_still_executes_without_selection() -> None:
    calls: list[bool] = []

    def emit(
        ports: Any, work: tuple[int, ...], slots: tuple[int, ...], adjoint: bool
    ) -> tuple[Instruction, ...]:
        assert ports == {"local": (0,)} and not work and not slots
        calls.append(adjoint)
        return (Instruction("x", ports["local"]),)

    prepare = BlackBoxPrepare((), (), emit, 1.0, (("local", 1),))
    circuit = Circuit(1).append(prepare.make_on(selection=(), junk=(), workspace=(0,)))
    circuit.append(prepare.make_on(selection=(), junk=(), workspace=(0,), adjoint=True))
    assert calls == [False, True]
    assert circuit.to_ir().instructions == (Instruction("x", (0,)),) * 2


@pytest.mark.parametrize("adjoint", (False, True))
def test_consumed_local_input_can_return_as_another_logical_port(adjoint: bool) -> None:
    import cirq
    import numpy as np
    from attrs import evolve, frozen
    from qualtran import BQUInt, Register, Side
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class MoveSelection(PrepareOracle):
        inverse: bool = False

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("selection", BQUInt(2, 3)),)

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (
                Register(
                    "local",
                    BQUInt(2, 3),
                    side=Side.LEFT if self.inverse else Side.RIGHT,
                ),
            )

        @property
        def l1_norm_of_coeffs(self) -> float:
            return 3.0

        def adjoint(self) -> Any:
            return evolve(self, inverse=not self.inverse)

        def build_composite_bloq(
            self, bb: Any, selection: Any, **soqs: Any
        ) -> dict[str, Any]:
            if self.inverse:
                bb.free(selection)
                return {"selection": soqs["local"]}
            return {"selection": bb.allocate(dtype=BQUInt(2, 3)), "local": selection}

        def as_cirq_op(
            self, qubit_manager: Any, **quregs: Any
        ) -> tuple[None, dict[str, Any]]:
            # The external oracle explicitly returns its physical output aliases.
            if self.inverse:
                qubit_manager.qfree(quregs["selection"].flat)
                return None, {"selection": quregs["local"]}
            return None, {
                "selection": np.asarray(qubit_manager.qalloc(2), dtype=object),
                "local": quregs["selection"],
            }

    select = SelectPauliLCU(
        2, 1, tuple(cirq.DensePauliString(w) for w in ("X", "Y", "Z"))
    )
    source = LCUBlockEncoding(select, MoveSelection())
    ir, native = _circuit(source, adjoint)
    owner = native.select_block.signal_state.prepare
    assert owner.binding_swaps == (4, 2)
    assert sum(g.name == "swap" for g in ir.instructions) == 6
    events = certify_erasure_histories(
        ir.instructions, n_wires=ir.n_wires, zero_wires=tuple(range(3, ir.n_wires))
    )
    assert events == 1
    clean = torch.zeros(1 << (ir.n_wires - 3), dtype=torch.complex128)
    clean[0] = 1
    original = source.adjoint() if adjoint else source
    matrix = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    for state in torch.eye(8, dtype=torch.complex128):
        for history in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), ir.instructions, history),
                torch.kron(matrix @ state, clean) * 2 ** (-events / 2),
                atol=8e-12,
                rtol=0,
            )


def test_symbolic_normalization_is_metadata_not_a_gate_parameter() -> None:
    from sympy import Symbol

    from flagquantum.algorithms.primitives.black_box_prepare import (
        PackedPrepareDeclaration,
    )
    from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
    from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.value_math import SymbolicReal

    norm = 2 * Symbol("lambda", positive=True) + 1
    source = _source(normalization=norm)
    ir, native = _circuit(source, False)
    assert native.alpha == native.select_block.alpha == source.alpha == norm
    assert QubitizationWalk(native).alpha == norm
    logical = 4
    state = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128)
    state /= state.norm()
    clean = torch.zeros(1 << (ir.n_wires - logical), dtype=torch.complex128)
    clean[0] = 1
    expected = torch.kron(
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128) @ state, clean
    )
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(torch.kron(state, clean), ir.instructions, history),
            expected / 2,
            atol=8e-12,
            rtol=0,
        )

    class Declaration(PrepareOracle):
        @property
        def selection_registers(self) -> tuple[Register, ...]:
            return (Register("q", quantum_bit_type()),)

        @property
        def l1_norm_of_coeffs(self) -> SymbolicReal:
            return norm

    packed = PackedPrepareDeclaration(Declaration()).bind(lambda *_: ())
    assert packed.normalization == norm


@pytest.mark.parametrize("adjoint", (False, True))
def test_select_can_use_a_declared_preparation_local_output(adjoint: bool) -> None:
    source = _source(shaped=False, local_select=True)
    ir, native = _circuit(source, adjoint)
    assert native.select_block.selection_layout == (("local", 1),)
    assert native.selection_layout == (("selection", 2),)
    assert native.junk_layout == (("through", 1),)
    assert native.resource_bitsize == 2
    events = certify_erasure_histories(
        ir.instructions, n_wires=ir.n_wires, zero_wires=tuple(range(4, ir.n_wires))
    )
    assert events == 1
    clean = torch.zeros(1 << (ir.n_wires - 4), dtype=torch.complex128)
    clean[0] = 1
    original = source.adjoint() if adjoint else source
    matrix = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    for state in torch.eye(16, dtype=torch.complex128):
        for history in ((0,), (1,)):
            torch.testing.assert_close(
                forced_branch(torch.kron(state, clean), ir.instructions, history),
                torch.kron(matrix @ state, clean) * 2 ** (-events / 2),
                atol=8e-12,
                rtol=0,
            )


@pytest.mark.parametrize("adjoint", (False, True))
def test_actual_zero_selection_oracles_with_local_output(adjoint: bool) -> None:
    from attrs import frozen
    from qualtran import QBit, Register, Side
    from qualtran.bloqs.basic_gates import SGate, XGate
    from qualtran.bloqs.block_encoding.lcu_block_encoding import LCUBlockEncoding
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class Prepare(PrepareOracle):
        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (Register("local", QBit(), side=Side.RIGHT),)

        @property
        def l1_norm_of_coeffs(self) -> float:
            return 1.0

        def build_composite_bloq(self, bb: Any) -> dict[str, Any]:
            return {"local": bb.add(XGate(), q=bb.allocate(dtype=QBit()))}

    @frozen
    class Select(SelectOracle):
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("target", QBit()),)

        def build_composite_bloq(self, bb: Any, target: Any) -> dict[str, Any]:
            return {"target": bb.add(SGate(), q=target)}

    source = LCUBlockEncoding(Select(), Prepare())
    ir, native = _circuit(source, adjoint)
    assert native.selection_layout == native.junk_layout == ()
    assert native.resource_bitsize == 1
    assert [g.name for g in ir.instructions] == ["x", "sdg" if adjoint else "s", "x"]
    assert (
        certify_erasure_histories(ir.instructions, n_wires=ir.n_wires, zero_wires=(1,))
        == 0
    )
    original = source.adjoint() if adjoint else source
    matrix = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    state = torch.tensor([1 + 2j, -2 + 1j], dtype=torch.complex128)
    state /= state.norm()
    clean = torch.tensor([1, 0], dtype=torch.complex128)
    torch.testing.assert_close(
        forced_branch(torch.kron(state, clean), ir.instructions, ()),
        torch.kron(matrix @ state, clean),
        atol=8e-12,
        rtol=0,
    )


def test_whole_source_symbolic_typed_declarations_without_physical_coercion() -> None:
    from attrs import frozen
    from qualtran import BQUInt, QAny, QUInt, Register, Side
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding,
        SelectBlockEncoding,
    )
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle
    from sympy import symbols

    from flagquantum.algorithms.primitives.lcu_block_encoding import (
        source_lcu_block_signature,
    )
    from flagquantum.algorithms.primitives.select_block_encoding import (
        source_select_block_signature,
    )
    from flagquantum.core.ir.program.register_types import (
        bounded_unsigned_register_type,
        opaque_register_type,
        unsigned_register_type,
    )
    from flagquantum.core.ir.program.registers import (
        Register as NativeRegister,
    )
    from flagquantum.core.ir.program.registers import (
        RegisterSide,
    )
    from tests.unit.test_sparse_chemistry_prepare import _assert_sparse_signature

    n, m, k, ell, t, bound, alpha = symbols(
        "n m k ell t L alpha", integer=True, positive=True
    )

    @frozen
    class Prepare(PrepareOracle):
        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return (Register("selection", BQUInt(n, bound), shape=(2,)),)

        @property
        def junk_registers(self) -> tuple[Any, ...]:
            return (
                Register("through", QAny(m), shape=(3,)),
                Register("local", QAny(k), side=Side.RIGHT),
                Register("left", QAny(ell), side=Side.LEFT),
            )

        @property
        def l1_norm_of_coeffs(self) -> Any:
            return alpha

    @frozen
    class Select(SelectOracle):
        @property
        def control_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def selection_registers(self) -> tuple[Any, ...]:
            return Prepare().selection_registers

        @property
        def target_registers(self) -> tuple[Any, ...]:
            return (Register("target", QUInt(t), shape=(2, 2)),)

    selection = (
        NativeRegister(
            "selection", bounded_unsigned_register_type(n, bound), shape=(2,)
        ),
    )
    junk = (
        NativeRegister("through", opaque_register_type(m), shape=(3,)),
        NativeRegister("local", opaque_register_type(k), side=RegisterSide.RIGHT),
        NativeRegister("left", opaque_register_type(ell), side=RegisterSide.LEFT),
    )
    target = (NativeRegister("target", unsigned_register_type(t), shape=(2, 2)),)
    select = SelectBlockEncoding(Select(), Prepare())
    _assert_sparse_signature(
        source_select_block_signature(selection, target), select.signature
    )
    for control in (None, 0, 1):
        lcu = LCUBlockEncoding(Select(), Prepare(), control_val=control)
        _assert_sparse_signature(
            source_lcu_block_signature(
                selection, junk, target, controlled=control is not None
            ),
            lcu.signature,
        )
        assert (
            lcu.ancilla_bitsize,
            lcu.resource_bitsize,
            lcu.system_bitsize,
            lcu.alpha,
            lcu.epsilon,
        ) == (2 * n, 3 * m + k + ell, 4 * t, alpha, 0)
    assert select.alpha == alpha and select.junk_registers == ()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(Select())
