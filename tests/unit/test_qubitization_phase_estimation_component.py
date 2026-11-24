"""Typed walk-QPE stages retain source powers, zero controls and clean storage."""

from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum.algorithms.primitives.phase_estimation_window import (
    rectangular_window_program,
)
from flagquantum.compiler.formats.legacy_qubitization_qpe_factories import (
    ExactPair,
    WalkDependencies,
    qubitization_qpe_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.bits.on_each import on_each_word_program
from flagquantum.compiler.registers.component_adjoint import adjoint_component_calls
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_power import power_word_program
from flagquantum.compiler.registers.component_wire_graph import wire_component_program
from flagquantum.compiler.registers.phase.qubitization_phase_estimation_component import (
    qubitization_phase_estimation_program,
)
from flagquantum.compiler.registers.phase.textbook_qft_component import (
    textbook_qft_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_qubitization_phase_estimation import source_walk as source_walk
from tests.unit.test_register_component_calls import SourceCalls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _snapshot_program(
    source: Any, ports: tuple[Register, ...], inverse: bool = False
) -> ProgramRecord:
    """Wrap the already accepted finite SDK snapshot, with its private work.

    This is a test replacement child, not a constructor implementation or a
    claim that the source's registerless scalar graph is natively admitted.
    """
    emitter = import_register_emitter(source)
    circuit = emitter.snapshots[int(inverse)]
    bit = quantum_bit_type()
    leaf = ProgramRecord(
        (
            bit_gate_function(
                "snapshot",
                tuple(Register(f"q{i}", bit) for i in range(circuit.n_wires)),
                circuit,
            ),
        )
    )
    wires = {}
    cursor = 0
    for reg in ports:
        width = int(reg.total_bits())
        wires[reg.name] = tuple(range(cursor, cursor + width))
        cursor += width
    return wire_component_program(
        Signature(ports),
        wires,
        wires,
        ((leaf, {f"q{i}": (i,) for i in range(circuit.n_wires)}),),
    )


def _dependencies(
    walk: Any, width: int
) -> tuple[ExactPair, WalkDependencies, ExactPair, tuple[Register, ...]]:
    from qualtran import CtrlSpec

    selection = Register("selection", bounded_unsigned_register_type(1, 2))
    target = Register("target", opaque_register_type(1))
    bit = quantum_bit_type()
    ports = (selection, target)
    ordinary = tuple(
        _rename_program(_snapshot_program(walk, ports, inverse), "walk")
        for inverse in (False, True)
    )
    control_ports = (Register("ctrl", bit), *ports)
    controlled = tuple(
        _snapshot_program(walk.controlled(), control_ports, inverse)
        for inverse in (False, True)
    )
    reflect_ports = (Register("control", bit), selection)
    reflection = tuple(
        _snapshot_program(
            walk.reflect.controlled(CtrlSpec(cvs=0)), reflect_ports, inverse
        )
        for inverse in (False, True)
    )
    powers = tuple(
        power_word_program(
            ordinary[0].functions[0], 2**i, definitions=ordinary[0].functions[1:]
        )
        for i in range(1, width)
    )
    reverse_powers = tuple(
        power_word_program(
            ordinary[1].functions[0], 2**i, definitions=ordinary[1].functions[1:]
        )
        for i in range(1, width)
    )
    prep = rectangular_window_program(width)
    h = bit_gate_function(
        "hadamard", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
    )
    each = on_each_word_program(width, h)
    unprep = adjoint_component_calls(prep, ((each, each),))
    qft = textbook_qft_program(width, adjoint=True)
    return (
        (prep, unprep),
        (
            (controlled[0], controlled[1]),
            (reflection[0], reflection[1]),
            tuple(zip(powers, reverse_powers, strict=True)),
        ),
        (qft, textbook_qft_program(width)),
        ports,
    )


def _programs(walk: Any, width: int) -> tuple[ProgramRecord, ProgramRecord]:
    prep, (control, reflection, powers), qft, ports = _dependencies(walk, width)
    return _programs_from_pairs(prep, (control, reflection, powers), qft, ports)


@pytest.mark.parametrize("width", (1, 2, 3))
def test_source_root_calls_complete_complex_isometry_and_all_small_histories(
    source_walk: Any, width: int
) -> None:
    from qualtran.bloqs.phase_estimation import QubitizationQPE, RectangularWindowState

    source = QubitizationQPE(source_walk, RectangularWindowState(width))
    forward, inverse = _programs(source_walk, width)
    for record, original in ((forward, source), (inverse, source.adjoint())):
        assert not verify_program_record(record)
        assert _native_trace(record) == _source_trace(
            SourceCalls, original.decompose_bloq()
        )
    logical = width + 2
    scratch = (logical, logical + 1)
    bound = bind_register_program(
        forward,
        ((width,), (width + 1,)),
        clean=(*range(width), *scratch),
        allow_workspace_outputs=True,
    )
    erased = bind_register_program(
        inverse,
        (tuple(range(width)), (width,), (width + 1,)),
        clean=scratch,
        input_ownership=tuple((i, False) for i in range(width)),
    )
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical)[:, :4], expected, atol=4e-11, rtol=0
    )
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    seed = torch.tensor((0.5, 0.5j, -0.5, -0.5j), dtype=torch.complex128)
    initial[:16:4] = seed
    gates = (*bound.circuit.instructions, *erased.circuit.instructions)
    measurements = sum(g.name == "measure" for g in gates)
    histories = (
        product((0, 1), repeat=measurements) if width <= 2 else ((0,) * measurements,)
    )
    for history in histories:
        restored = forced_branch(initial, gates, history) * sqrt(2**measurements)
        torch.testing.assert_close(restored, initial, atol=4e-11, rtol=0)


def test_walk_qpe_refuses_missing_powers_and_control_context(source_walk: Any) -> None:
    forward, _ = _programs(source_walk, 2)
    assert forward.functions[0].registers is not None
    prep = rectangular_window_program(2)
    qft = textbook_qft_program(2, adjoint=True)
    bit = quantum_bit_type()
    wrong = ProgramRecord(
        (
            bit_gate_function(
                "wrong", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
            ),
        )
    )
    with pytest.raises(ValueError, match="explicit powers"):
        qubitization_phase_estimation_program(
            prep, wrong, wrong, (), qft, (Register("target", bit),)
        )
    with pytest.raises(ValueError, match="control port"):
        qubitization_phase_estimation_program(
            prep, wrong, wrong, (wrong,), qft, (Register("target", bit),)
        )


@pytest.mark.parametrize("default_qft", (False, True))
def test_constructor_uses_exact_owned_context_and_inverse(
    source_walk: Any, default_qft: bool
) -> None:
    import sys

    from flagquantum.core.ir.program import FunctionRecord

    preparation, walk, fourier, ports = _dependencies(source_walk, 2)
    prep_child = FunctionRecord(
        "prepare", (), registers=preparation[0].functions[0].registers
    )
    walk_child = FunctionRecord("walk", (), registers=ports)
    qft_child = FunctionRecord(
        "fourier", (), registers=fourier[0].functions[0].registers
    )

    def prep_owner(child: FunctionRecord) -> ExactPair:
        assert child == prep_child
        return preparation

    def walk_owner(child: FunctionRecord, width: int) -> WalkDependencies:
        assert child == walk_child and width == 2
        return walk

    def qft_owner(child: FunctionRecord) -> ExactPair:
        assert not default_qft and child == qft_child
        return fourier

    factories = qubitization_qpe_factories(
        preparation_pair=prep_owner,
        walk_dependencies=walk_owner,
        fourier_pair=qft_owner,
    )
    forward_factory = factories[
        "qualtran.bloqs.phase_estimation.qubitization_qpe.QubitizationQPE"
    ]
    arguments = {"walk": walk_child, "ctrl_state_prep": prep_child}
    if not default_qft:
        arguments["qft_inv"] = qft_child
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        forward = forward_factory("main", None, arguments)
        inverse = factories["qualtran._infra.adjoint.Adjoint"](
            "main",
            None,
            {
                "subbloq": (
                    forward.functions[0]
                    if isinstance(forward, ProgramRecord)
                    else forward
                )
            },
        )
        assert isinstance(forward, ProgramRecord) and isinstance(inverse, ProgramRecord)
        accepted = _programs_from_pairs(preparation, walk, fourier, ports)
        for actual, expected in zip((forward, inverse), accepted, strict=True):
            inverse_direction = actual is inverse
            inputs = ((0, 1), (2,), (3,)) if inverse_direction else ((2,), (3,))
            # Parameterize bindings explicitly so ownership remains a typed
            # input contract, rather than a mutable numerical test side effect.
            if inverse_direction:
                a = bind_register_program(
                    actual,
                    inputs,
                    clean=(4, 5),
                    input_ownership=((0, False), (1, False)),
                )
                b = bind_register_program(
                    expected,
                    inputs,
                    clean=(4, 5),
                    input_ownership=((0, False), (1, False)),
                )
            else:
                a = bind_register_program(
                    actual, inputs, clean=(0, 1, 4, 5), allow_workspace_outputs=True
                )
                b = bind_register_program(
                    expected, inputs, clean=(0, 1, 4, 5), allow_workspace_outputs=True
                )
            assert a.circuit.to_json() == b.circuit.to_json()
    from flagquantum.errors import SerializationError

    with pytest.raises(SerializationError):
        forward_factory("bad", None, {})


def _programs_from_pairs(
    preparation: ExactPair,
    walk: WalkDependencies,
    fourier: ExactPair,
    ports: tuple[Register, ...],
) -> tuple[ProgramRecord, ProgramRecord]:
    control, reflection, powers = walk
    result = []
    for inverse in (False, True):
        result.append(
            qubitization_phase_estimation_program(
                preparation[0],
                control[0],
                reflection[0],
                tuple(p[0] for p in powers),
                fourier[0],
                ports,
                adjoint=inverse,
                inverse_pairs=(preparation, control, reflection, *powers, fourier),
            )
        )
    return result[0], result[1]


def test_native_library_resolution_retains_owned_helper_closures(
    source_walk: Any,
) -> None:
    """Exercise the resolver naming boundary without claiming source wire success."""
    import sys
    from collections.abc import Mapping

    from flagquantum import Circuit
    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation
    from flagquantum.compiler.formats.legacy_component_arguments import (
        ComponentArgument,
        ComponentReference,
    )
    from flagquantum.compiler.formats.legacy_component_library import (
        ComponentDeclaration,
        ComponentLibrary,
        ComponentLibraryEntry,
    )
    from flagquantum.compiler.formats.legacy_component_resolution import (
        ComponentFactory,
        ResolvedArgument,
        resolve_component_program,
    )
    from flagquantum.core.ir.program import FunctionRecord

    preparation, walk, fourier, ports = _dependencies(source_walk, 2)
    ordinary = _snapshot_program(source_walk, ports)
    owned: dict[FunctionRecord, str] = {}

    def wrap(kind: str, program: ProgramRecord) -> ComponentFactory:
        def factory(
            name: str,
            registers: tuple[Register, ...] | None,
            arguments: Mapping[str, ResolvedArgument],
        ) -> ProgramRecord:
            assert not arguments and registers == program.functions[0].registers
            result = _rename_program(program, name)
            owned[result.functions[0]] = kind
            return result

        return factory

    def prep_owner(child: FunctionRecord) -> ExactPair:
        assert owned[child] == "prepare"
        return preparation

    def walk_owner(child: FunctionRecord, width: int) -> WalkDependencies:
        assert owned[child] == "walk" and width == 2
        return walk

    def qft_owner(child: FunctionRecord) -> ExactPair:
        assert owned[child] == "fourier"
        return fourier

    factories = {
        **qubitization_qpe_factories(
            preparation_pair=prep_owner,
            walk_dependencies=walk_owner,
            fourier_pair=qft_owner,
        ),
        "fixture.prepare": wrap("prepare", preparation[0]),
        "fixture.walk": wrap("walk", ordinary),
        "fixture.fourier": wrap("fourier", fourier[0]),
    }
    root = ComponentDeclaration(
        "qualtran.bloqs.phase_estimation.qubitization_qpe.QubitizationQPE",
        tuple(
            ComponentArgument(key, ComponentReference(index))
            for key, index in (("ctrl_state_prep", 1), ("walk", 2), ("qft_inv", 3))
        ),
    )
    entries = (
        ComponentLibraryEntry(0, root),
        *tuple(
            ComponentLibraryEntry(
                index,
                ComponentDeclaration(
                    "fixture." + kind, registers=program.functions[0].registers
                ),
            )
            for index, kind, program in (
                (1, "prepare", preparation[0]),
                (2, "walk", ordinary),
                (3, "fourier", fourier[0]),
            )
        ),
    )
    library = ComponentLibrary("native fixture", entries)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        patch.setitem(sys.modules, "cirq", None)
        record = resolve_component_program(library, 0, factories)
        assert not verify_program_record(record)
        bound = bind_register_program(
            record, ((2,), (3,)), clean=(0, 1, 4, 5), allow_workspace_outputs=True
        )
        circuit = Circuit(bound.circuit.n_wires)
        circuit.append(
            _BoundBlockOperation(bound.circuit.instructions, bound.circuit.n_wires)
        )
        assert (
            circuit.to_ir().to_dict()["instructions"]
            == bound.circuit.to_dict()["instructions"]
        )
    direct = _programs_from_pairs(preparation, walk, fourier, ports)[0]
    expected = bind_register_program(
        direct, ((2,), (3,)), clean=(0, 1, 4, 5), allow_workspace_outputs=True
    )
    assert bound.circuit.to_json() == expected.circuit.to_json()
