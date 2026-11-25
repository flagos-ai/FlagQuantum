"""Resolve original allocation factories through existing typed lifetime owners."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
import sympy
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.register_source_workspace import SourceWorkspace
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def resolve_original(source: Any, depth: int) -> ProgramRecord:
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert bloqs_from_proto(proto)[0] == source

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None and set(arguments) == {"dtype", "dirty"}
        dtype, dirty = arguments["dtype"], arguments["dirty"]
        assert isinstance(dtype, RegisterType) and dirty in (0, 1)
        helper = (
            allocate_word_function
            if registers[0].side.name == "RIGHT"
            else free_word_function
        )
        return helper(name, dtype, bool(dirty))

    resolved = resolve_component_program(
        library, 0, {entry.declaration.name: factory for entry in library.entries}
    )
    assert resolved.functions[0].registers == tuple(native_signature(source.signature))
    assert verify_program_record(resolved) == ()
    return resolved


@pytest.mark.parametrize("kind", ("allocate", "free"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_symbolic_factories_complete_wire_and_native_declaration(
    kind: str, depth: int
) -> None:
    from qualtran.bloqs.bookkeeping.allocate import _alloc
    from qualtran.bloqs.bookkeeping.free import _free

    source = (_alloc if kind == "allocate" else _free).make()
    resolved = resolve_original(source, depth)
    assert isinstance(source.dtype.num_qubits, sympy.Expr)
    operation = resolved.functions[0].blocks[0].operations[0]
    assert operation.opcode == kind + "_register"
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        bind_register_program(resolved, ())


@pytest.mark.parametrize(
    "kind",
    (
        "bit",
        "opaque",
        "unsigned",
        "signed",
        "fixed",
        "signed_fixed",
        "signed_fixed_negative_fraction",
        "empty_fixed",
        "empty_fixed_negative_fraction",
        "ones_complement",
        "bounded",
        "montgomery",
    ),
)
@pytest.mark.parametrize("dirty", (False, True))
def test_concrete_foreign_lifetimes_roundtrip_supported_wire_dtypes(
    kind: str, dirty: bool
) -> None:
    from qualtran import (
        BQUInt,
        QAny,
        QBit,
        QFxp,
        QInt,
        QIntOnesComp,
        QMontgomeryUInt,
        QUInt,
    )
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dtype = {
        "bit": QBit(),
        "opaque": QAny(3),
        "unsigned": QUInt(3),
        "signed": QInt(3),
        "fixed": QFxp(3, -2),
        "signed_fixed": QFxp(3, 1, signed=True),
        "signed_fixed_negative_fraction": QFxp(3, -2, signed=True),
        "empty_fixed": QFxp(0, 0),
        "empty_fixed_negative_fraction": QFxp(0, -2),
        "ones_complement": QIntOnesComp(3),
        "bounded": BQUInt(3, 5),
        "montgomery": QMontgomeryUInt(3),
    }[kind]
    allocation, release = Allocate(dtype, dirty), Free(dtype, dirty)
    functions = tuple(
        resolve_original(source, 1).functions[0] for source in (allocation, release)
    )
    # Both original roots resolve as main. Give the existing definitions distinct
    # names before composing their ordinary calls.
    functions = tuple(
        replace(f, name=n) for f, n in zip(functions, ("allocate", "free"), strict=True)
    )
    builder = ComponentProgramBuilder(Signature(()), functions)
    (word,) = builder.call_tuple("allocate", {})
    builder.call_tuple("free", {"reg": word})
    width = dtype.num_qubits
    wires = tuple(reversed(range(width)))
    bound = bind_register_program(
        builder.finalize({}),
        (),
        clean=() if dirty else wires,
        borrowed=(width,) if width == 0 else wires if dirty else (),
    )
    assert bound.circuit.instructions == () and bound.outputs == ()
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("allocate", wires, dirty),
        ("free", wires, dirty),
    ]
    if width:
        source_pool = SourceWorkspace(() if dirty else wires, wires if dirty else ())
        operation, outputs = allocation.as_cirq_op(source_pool)
        assert operation is None and release.as_cirq_op(source_pool, **outputs) == (
            None,
            {},
        )
        assert not source_pool.live
        assert source_pool.events == [
            (event.kind, event.wires, event.dirty) for event in bound.events
        ]
    else:
        import cirq

        # An empty qfree carries no qubits from which a tracking pool could
        # recover its dirty flag. Use the actual SDK manager for this profile.
        manager = cirq.SimpleQubitManager()
        operation, outputs = allocation.as_cirq_op(manager)
        assert operation is None and outputs["reg"].shape == (0,)
        assert release.as_cirq_op(manager, **outputs) == (None, {})
    assert allocation.as_pl_op(wires=wires) is None
    assert release.as_pl_op(wires=wires) is None
    # Dirty storage may be coherently entangled with the spectator. Clean storage
    # is initialized to zero. An empty lifetime emits no operation in either case.
    generator = torch.Generator().manual_seed(59)
    initial = torch.randn(2 ** (width + 1), dtype=torch.complex128, generator=generator)
    if not dirty:
        initial[2:] = 0
    initial /= initial.norm()
    circuit = fq.Circuit(width + 1, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(width + 1, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    for source in (allocation, release):
        assert get_cost_value(source, QECGatesCost()).asdict() == {}
        assert get_cost_value(source, QubitCount()) == width
    if width:
        assert allocation.on_classical_vals()["reg"] == 0
    else:
        with pytest.raises(ValueError, match="invalid literal for int"):
            allocation.on_classical_vals()
    assert release.on_classical_vals(reg=0) == {}


@pytest.mark.parametrize("kind", ("allocate", "free"))
@pytest.mark.parametrize("polynomial", (False, True))
def test_original_foreign_wire_refuses_field_descriptors(
    kind: str, polynomial: bool
) -> None:
    from qualtran import QGF, QGFPoly
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.serialization.bloq import bloqs_to_proto

    dtype = QGFPoly(1, QGF(2, 2)) if polynomial else QGF(2, 2)
    source = (Allocate if kind == "allocate" else Free)(dtype)
    with pytest.raises(TypeError, match="is not recognized"):
        bloqs_to_proto(source, max_depth=1)
