"""Full frozen qubitization-QPE composition and independent child ownership."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha1
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Any

import cirq
import numpy as np
import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.luis_perina_state import LuisPerinaState
from flagquantum.algorithms.primitives.phase_estimation_window import (
    RectangularWindowState,
)
from flagquantum.algorithms.primitives.qubitization_phase_estimation import (
    QubitizationPhaseEstimation,
)
from flagquantum.algorithms.primitives.select_block_encoding import (
    ControlledSelectEmitter,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module")
def source_walk() -> Any:
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU
    from qualtran.bloqs.qubitization import QubitizationWalkOperator
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity

    select = SelectPauliLCU(
        1, 1, (cirq.DensePauliString("X"), cirq.DensePauliString("Z"))
    )
    return QubitizationWalkOperator(
        SelectBlockEncoding(select, PrepareIdentity(select.selection_registers))
    )


def native_qpe(
    source_walk: Any, width: int, measured: bool = False
) -> QubitizationPhaseEstimation:
    from qualtran import CtrlSpec

    walk = import_register_emitter(source_walk)
    controlled = import_register_emitter(source_walk.controlled())
    reflection = import_register_emitter(
        source_walk.reflect.controlled(CtrlSpec(cvs=0))
    )

    def walk_body(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return walk(ports, work, (), adjoint)

    def controlled_walk(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return controlled({"ctrl": (control,), **ports}, work, slots, adjoint)

    def zero_reflection(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return reflection({"control": (control,), **ports}, (), (), adjoint)

    selection = Register("selection", bounded_unsigned_register_type(1, 2))
    target = Register("target", opaque_register_type(1))
    window = LuisPerinaState(width) if measured else RectangularWindowState(width)
    return QubitizationPhaseEstimation(
        Signature((selection, target)),
        Signature((selection,)),
        window,
        walk_body,
        controlled_walk,
        zero_reflection,
        window.bind(),
    )


@pytest.mark.parametrize(
    "width,measured", [(1, False), (2, False), (3, False), (1, True), (2, True)]
)
def test_complete_schedule_source_children_complex_amplitudes_all_histories_and_inverse(
    source_walk: Any, width: int, measured: bool
) -> None:
    from qualtran.bloqs.basic_gates import Power
    from qualtran.bloqs.phase_estimation import QubitizationQPE
    from qualtran.bloqs.phase_estimation import RectangularWindowState as Window
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

    native = native_qpe(source_walk, width, measured)
    original = QubitizationQPE(
        source_walk, LPResourceState(width) if measured else Window(width)
    )
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in original.signature
    ]
    work = width + 4 if measured else 1
    slots = width + 1 if measured else 1
    phase = tuple(range(width))
    target = (width, width + 1)
    spectator = width + 2
    workspace = tuple(range(width + 3, width + 3 + work))
    prep_slots = tuple(range(width + 1)) if measured else ()
    prep_workspace = workspace if measured else ()
    forward = native.make_on(
        phase=phase,
        target=target,
        preparation_workspace=prep_workspace,
        preparation_classical_bits=prep_slots,
        walk_workspace=workspace[:1],
        walk_classical_bits=(slots - 1,),
    )
    reverse = native.make_on(
        phase=phase,
        target=target,
        preparation_workspace=prep_workspace,
        preparation_classical_bits=prep_slots,
        walk_workspace=workspace[:1],
        walk_classical_bits=(slots - 1,),
        adjoint=True,
    )
    circuit = DynamicCircuit(width + 3 + work, dtype=torch.complex128).append(forward)
    gates = circuit.to_ir().instructions
    q = cirq.LineQubit.range(width + 2)
    stages = tuple(
        cirq.flatten_op_tree(
            original.decompose_from_registers(
                cirq.DecompositionContext(cirq.SimpleQubitManager()),
                qpe_reg=np.array(q[:width]),
                selection=np.array(q[width : width + 1]),
                target=np.array(q[width + 1 :]),
            )
        )
    )
    assert len(stages) == 3 * width
    assert stages[1].qubits == (q[width - 1], q[width], q[width + 1])
    for index in range(width - 1):
        before, body, after = stages[2 + 3 * index : 5 + 3 * index]
        assert before == after and before.qubits == (q[width - 2 - index], q[width])
        assert isinstance(body, cirq.Operation)
        assert isinstance(body.gate, Power) and body.gate.power == 2 ** (index + 1)
        captured = import_register_emitter(body.gate)
        source_body = captured(
            {"selection": (target[0],), "target": (target[1],)},
            workspace[:1],
            (),
            False,
        )
        native_body = (
            native.walk(
                {"selection": (target[0],), "target": (target[1],)},
                workspace[:1],
                (),
                False,
            )
            * body.gate.power
        )
        assert len(native_body) == len(source_body)
        for wire in (*target, *workspace[:1]):
            assert_source_stream(
                CircuitIR(
                    width + 3 + work, tuple(g for g in native_body if wire in g.wires)
                ),
                CircuitIR(
                    width + 3 + work, tuple(g for g in source_body if wire in g.wires)
                ),
            )
    rng = np.random.default_rng(13 + width)
    logical = rng.normal(size=4) + 1j * rng.normal(size=4)
    logical /= np.linalg.norm(logical)
    observer = np.array([sqrt(0.3), 1j * sqrt(0.7)])
    zero, clean = np.zeros(2**width), np.zeros(2**work)
    zero[0] = clean[0] = 1
    initial = torch.tensor(np.kron(np.kron(np.kron(zero, logical), observer), clean))
    expected = torch.tensor(
        np.kron(np.kron(original.tensor_contract() @ logical, observer), clean)
    )
    events = Counter(g.name for g in gates)["measure"]
    assert events == 1 + (width + 1 if measured else 0)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**events),
            atol=2e-12,
            rtol=0,
        )
    for history in product((0, 1), repeat=2 * events):
        torch.testing.assert_close(
            forced_branch(initial, (*gates, *reverse.instructions), history),
            initial / 2**events,
            atol=4e-12,
            rtol=0,
        )
    assert spectator not in {wire for g in gates for wire in g.wires}


def test_symbolic_graph_and_invalid_reflection_signature(source_walk: Any) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.phase_estimation import QubitizationQPE
    from qualtran.bloqs.phase_estimation import RectangularWindowState as Window
    from qualtran.resource_counting import SympySymbolAllocator

    native = native_qpe(source_walk, 2)
    for width in (1, 2, 5, sympy.Symbol("m", integer=True, positive=True)):
        current = replace(native, ctrl_state_prep=RectangularWindowState(width))
        original = QubitizationQPE(source_walk, Window(width))
        graph = current.source_graph(
            root="root",
            preparation=original.ctrl_state_prep,
            controlled_walk=source_walk.controlled(),
            zero_controlled_reflection=source_walk.reflect.controlled(CtrlSpec(cvs=0)),
            walk=source_walk,
            inverse_qft=original.qft_inv,
        )
        expected = original.build_call_graph(SympySymbolAllocator())
        assert graph.leaf_counts == expected
        assert str(current) == f"QubitizationPhaseEstimation[{width}]"
    with pytest.raises(ValueError, match="unchanged walk"):
        replace(native, reflection_signature=Signature.build(unknown=1))
    with pytest.raises(ValueError, match="signature"):
        replace(native, qft_inv_signature=Signature.build(q=3))

    def escaped(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("x", (3,)),)

    replacement: ControlledSelectEmitter = escaped
    with pytest.raises(ValueError, match="partition"):
        replace(native, zero_controlled_reflection=replacement).make_on(
            phase=(0, 1), target=(2, 3), walk_workspace=(4,), walk_classical_bits=(0,)
        )


def test_complete_frozen_source_and_original_tests() -> None:
    from qualtran.bloqs.phase_estimation import qubitization_qpe

    root = Path(qubitization_qpe.__file__).parent
    for name, blob in (
        ("qubitization_qpe.py", "39b608ea5708eba219a7fa4f4365176170c40bf5"),
        ("qubitization_qpe_test.py", "301228e1f11e677993a2330a1c1dc64f2ec61c6b"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("width", [1, 2, 4])
def test_independent_composed_resource_categories_and_peak(
    source_walk: Any, width: int
) -> None:
    from qualtran.bloqs.phase_estimation import QubitizationQPE
    from qualtran.bloqs.phase_estimation import RectangularWindowState as Window

    from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
    from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
        SelectPauliLCU,
    )
    from flagquantum.algorithms.primitives.prepared_state_reflection import (
        PreparedStateReflection,
    )
    from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
    from flagquantum.algorithms.primitives.select_block_encoding import (
        SelectBlockEncoding,
    )
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
    )
    from flagquantum.compiler.registers.control.control_predicate import ControlRegister
    from tests.unit.test_first_quantization_resource_model import assert_source_model

    native = native_qpe(source_walk, width)
    selection = tuple(native.reflection_signature)
    identity = IdentityPrepare(selection).bind()
    selector = SelectPauliLCU(1, 1, ("X", "Z"))

    def unused(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        raise AssertionError("A declared resource model must not emit a circuit")

    walk = QubitizationWalk(
        SelectBlockEncoding((("selection", 1),), (("target", 1),), unused, identity)
    )
    select_cost = selector.make_bounded_on(
        selection=(0,), target=(1,), workspace=(), classical_bits=()
    ).resource_model()
    controlled_select = (
        replace(selector, control_val=1)
        .make_bounded_on(
            selection=(1,),
            target=(2,),
            control=(0,),
            workspace=(3,),
            classical_bits=(0,),
        )
        .resource_model()
    )
    empty = ComponentCost("IdentityPrepare", 1)
    uncontrolled = walk.resource_model(select_cost, empty, empty)
    controlled = walk.resource_model(
        controlled_select, empty, empty, control_registers=(ControlRegister(1, (1,)),)
    )
    reflection = PreparedStateReflection(
        IdentityPrepare(selection), control_value=0, global_phase=-1
    ).resource_model(empty, empty)
    model = native.resource_model(
        ComponentCost("Window", width, GateCost(clifford=width)),
        controlled,
        reflection,
        uncontrolled,
    )
    assert_source_model(model, QubitizationQPE(source_walk, Window(width)))


def test_complete_original_factory_declarations_and_accuracy_formulas() -> None:
    import json
    import os
    import subprocess
    import sys

    # Source factories import original marked test helpers. Execute unchanged
    # outside this repository's strict-marker pytest session.
    script = """
import json
import torch
from qualtran.bloqs.phase_estimation import qubitization_qpe as q
names = ["ising", "hubbard_model_small", "hubbard_model_large", "chem_thc", "sparse_chem"]
def signature(registers):
    return [dict(name=r.name, kind=type(r.dtype).__name__, width=r.bitsize,
                 bound=getattr(r.dtype, "iteration_length", None),
                 shape=list(r.shape), side=r.side.name) for r in registers]
examples = []
for name in names:
    source = getattr(q, "_qubitization_qpe_" + name)()
    examples.append(dict(walk=signature(source.walk.signature),
                         reflect=signature(source.walk.reflect.signature),
                         signature=signature(source.signature), width=source.m_bits,
                         alpha=float(source.walk.block_encoding.alpha)))
print(json.dumps(examples))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    examples = json.loads(result.stdout)

    def signature(registers: list[dict[str, Any]]) -> Signature:
        ports = []
        for reg in registers:
            if reg["kind"] == "BQUInt":
                dtype = bounded_unsigned_register_type(reg["width"], reg["bound"])
            else:
                assert reg["kind"] == "QAny"
                dtype = opaque_register_type(reg["width"])
            ports.append(Register(reg["name"], dtype, tuple(reg["shape"])))
        return Signature(tuple(ports))

    def unresolved(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        raise NotImplementedError(
            "Large factory fixtures only supply declaration metadata"
        )

    def controlled_unresolved(
        control: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return unresolved(ports, work, slots, adjoint)

    for source, error, expected_width in zip(
        examples,
        (None, 2 / 100, 20 / 100, 0.0016, 0.0016),
        (4, 14, 21, 40, 21),
        strict=True,
    ):
        window = (
            RectangularWindowState(4)
            if error is None
            else LuisPerinaState.from_standard_deviation_eps(
                error / (source["alpha"] * np.sqrt(2))
            )
        )
        native = QubitizationPhaseEstimation(
            signature(source["walk"]),
            signature(source["reflect"]),
            window,
            unresolved,
            controlled_unresolved,
            controlled_unresolved,
        )
        assert native.m_bits == source["width"] == expected_width
        assert [
            (r.name, r.bitsize, r.shape, r.side.name) for r in native.signature
        ] == [
            (r["name"], r["width"], tuple(r["shape"]), r["side"])
            for r in source["signature"]
        ]
        # These five metadata examples do not claim large physical simulation.
        with pytest.raises(ValueError, match="PREPARE"):
            native.make_on(
                phase=tuple(range(expected_width)),
                target=tuple(range(expected_width, int(native.signature.n_qubits()))),
            )


def test_owned_walk_replacement_preserves_consumer_complex_phase_and_clean_work(
    source_walk: Any,
) -> None:
    from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
    from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
        SelectPauliLCU,
    )
    from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
    from flagquantum.algorithms.primitives.select_block_encoding import (
        SelectBlockEncoding,
    )

    native = native_qpe(source_walk, 2)
    selector = SelectPauliLCU(1, 1, ("X", "Z"))

    def select(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return selector.make_bounded_on(
            selection=ports["selection"],
            target=ports["target"],
            workspace=(),
            classical_bits=(),
        ).emit()

    owned = QubitizationWalk(
        SelectBlockEncoding(
            (("selection", 1),),
            (("target", 1),),
            select,
            IdentityPrepare(tuple(native.reflection_signature)).bind(),
        )
    )

    def walk_body(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return owned.make_on(
            selection={"selection": ports["selection"]},
            target=ports["target"],
            reflection_workspace=work,
            adjoint=adjoint,
        ).instructions

    baseline = native.make_on(
        phase=(0, 1), target=(2, 3), walk_workspace=(5,), walk_classical_bits=(0,)
    )
    replaced = replace(native, walk=walk_body).make_on(
        phase=(0, 1), target=(2, 3), walk_workspace=(5,), walk_classical_bits=(0,)
    )
    logical = np.array([1, 2j, -1j, 2]) / sqrt(10)
    initial = torch.tensor(
        np.kron(
            np.kron(np.kron([1, 0, 0, 0], logical), [sqrt(0.3), 1j * sqrt(0.7)]), [1, 0]
        )
    )
    for history in ((0,), (1,)):
        expected = forced_branch(initial, baseline.instructions, history)
        torch.testing.assert_close(
            forced_branch(initial, replaced.instructions, history),
            expected,
            atol=2e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            expected.numpy().reshape(32, 2)[:, 1], 0, atol=1e-13, rtol=0
        )
