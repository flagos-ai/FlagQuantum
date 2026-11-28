"""Frozen DF top-level graph capture, resource consumers and source boundaries."""

import builtins
from collections.abc import Mapping
from dataclasses import asdict, replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_encoding_cost,
    source_double_factorization_one_body_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.formats.legacy_component_export import (
    component_entry_from_program,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import (
    FunctionRecord,
    program_to_circuit,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from flagquantum.ecosystem.cirq.models import CirqConversionError
from tests._reference_dependencies import unavailable_reference

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,depth,adjoint,stored",
    tuple(product(("one", "block"), (0, 1, 2), (False, True), (False, True))),
)
@pytest.mark.parametrize("spin", (6, 10))
def test_actual_top_level_constructor_and_stored_graph_model_consumers(
    kind: str,
    depth: int,
    adjoint: bool,
    stored: bool,
    spin: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from attrs import fields
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding,
        DoubleFactorizationOneBody,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = (
        DoubleFactorizationOneBody(50, spin, 250, 12, num_bits_rot=8)
        if kind == "one"
        else DoubleFactorizationBlockEncoding(
            spin,
            50,
            250,
            num_bits_state_prep=12,
            num_bits_rot=8,
            num_bits_rot_aa_outer=3,
            num_bits_rot_aa_inner=8,
        )
    )
    declaration = (
        deserialize_component_library(
            bloqs_to_proto(source, max_depth=0).SerializeToString()
        )
        .entries[0]
        .declaration
    )
    arguments = {argument.name: argument.value for argument in declaration.arguments}
    # The frozen source constructors accept dimensions and precisions only.
    # External PREPARE/SELECT injection is not a field of either source class.
    assert set(arguments) == {field.name for field in fields(type(source))}
    assert all(type(value) is int for value in arguments.values())

    def integer(name: str) -> int:
        value = arguments[name]
        assert type(value) is int
        return value

    common = (
        integer("num_spin_orb"),
        integer("num_eig"),
        integer("num_bits_state_prep"),
        integer("num_bits_rot"),
    )
    model = (
        source_double_factorization_one_body_cost(*common, integer("num_bits_rot_aa"))
        if kind == "one"
        else source_double_factorization_block_encoding_cost(
            integer("num_aux"), *common, integer("num_bits_rot_aa_inner")
        )
    )
    direction = source.adjoint() if adjoint else source
    root = direction.decompose_bloq() if stored else direction
    proto = bloqs_to_proto(root, max_depth=depth)
    raw = proto.SerializeToString()
    library = deserialize_component_library(raw)
    encoded = serialize_component_library(library)
    assert type(proto).FromString(encoded) == proto
    assert (
        serialize_component_library(deserialize_component_library(encoded)) == encoded
    )
    assert library.entries[0].declaration.registers == declaration.registers
    restored = bloqs_from_proto(type(proto).FromString(encoded))[0]
    if stored and depth == 0:
        # A source depth-zero CompositeBloq export is a declaration-only shell.
        assert not library.entries[0].connections
        assert restored != root
    else:
        assert restored == root
    if depth:
        assert len(library.entries[0].connections) == (45 if kind == "one" else 55)
    assert declaration.registers is not None
    assert model.qubits == sum(
        register.total_bits() for register in declaration.registers
    )
    expected = get_cost_value(direction, QECGatesCost()).asdict()
    peak = get_cost_value(direction, QubitCount())
    assert source.adjoint().adjoint() == source
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(
                f"Owned DF encoding model retained SDK dependency {name}"
            )
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        estimate = estimate_component_resources(model)
    assert {
        name: count for name, count in asdict(estimate.gates).items() if count
    } == expected
    assert estimate.peak_logical_qubits == peak


@pytest.mark.parametrize("kind", ("one", "block"))
def test_source_failure_stages_and_actual_signal_prepare_append(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        import cirq
        import numpy as np
        from qualtran.bloqs.chemistry.df.double_factorization import (
            DoubleFactorizationBlockEncoding,
            DoubleFactorizationOneBody,
        )
        from qualtran.serialization.bloq import bloqs_to_proto
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    source = (
        DoubleFactorizationOneBody(50, 6, 250, 12, num_bits_rot=8)
        if kind == "one"
        else DoubleFactorizationBlockEncoding(
            6, 50, 250, num_bits_state_prep=12, num_bits_rot=8
        )
    )
    for property_name in ("alpha", "epsilon"):
        with pytest.raises(NotImplementedError):
            getattr(source, property_name)
    if kind == "block":
        with pytest.raises(NotImplementedError, match="Factory method not implemented"):
            DoubleFactorizationBlockEncoding.build_from_coeffs(
                np.eye(3), np.ones((1, 3, 3))
            )
    graph = source.decompose_bloq().flatten()
    with pytest.raises(ValueError, match="numpy.uint8"):
        bloqs_to_proto(graph, max_depth=1)
    # Live graph capture no longer serializes the source first. It reaches
    # the original opaque SDK leaf and must reject that unsupported operation.
    with pytest.raises(CirqConversionError) as rejected:
        import_register_graph(source)
    report = rejected.value.report
    assert report.adapter == "cirq" and report.direction == "from_cirq"
    assert len(report.issues) == 1
    assert report.issues[0].code == "unsupported_operation"
    assert report.issues[0].operation_name == "BloqAsCirqGate"
    exported = source.decompose_bloq().to_cirq_circuit()
    with pytest.raises(TypeError, match="Opaque source leaf"):
        cirq.decompose(
            exported,
            keep=lambda operation: type(operation.gate).__module__.startswith("cirq."),
            on_stuck_raise=TypeError("Opaque source leaf"),
        )
    assert source.ancilla_bitsize == sum(
        register.total_bits() for register in source.selection_registers
    )
    assert source.resource_bitsize == sum(
        register.total_bits() for register in source.junk_registers
    )
    assert source.system_bitsize == 6
    signal = source.signal_state
    assert signal.selection_bitsize == source.ancilla_bitsize
    assert signal.junk_bitsize == 0
    with pytest.raises(NotImplementedError):
        _ = signal.l1_norm_of_coeffs
    emitter = import_register_emitter(signal)
    assert emitter.layout == (("selection", source.ancilla_bitsize),)
    assert emitter.requirements() == (0, 0)
    assert emitter.binding_swaps == (0, 0)
    width = source.ancilla_bitsize + 1
    ports = {"selection": tuple(reversed(range(source.ancilla_bitsize)))}
    raw = torch.arange(1, (1 << width) + 1, dtype=torch.float64)
    initial = torch.complex(raw, 3 - raw)
    initial /= initial.norm()
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError(f"Owned DF signal retained SDK dependency {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        for adjoint in (False, True):
            gates = emitter(ports, (), (), adjoint)
            assert [(gate.name, gate.wires) for gate in gates] == [
                ("i", (1,)),
                ("i", (0,)),
            ]
            circuit = Circuit(width, inputs=initial, dtype=torch.complex128)
            circuit.append(FieldOperation(width, gates))
            artifact = CircuitIR.from_json(circuit.to_ir().to_json())
            torch.testing.assert_close(
                circuit.state().flatten(), initial, atol=1e-12, rtol=0
            )
            assert artifact.instructions == gates


@pytest.mark.parametrize(
    "kind,spin,adjoint", tuple(product(("one", "block"), (6, 10), (False, True)))
)
def test_actual_stored_graph_reaches_owned_typed_calls_without_invented_bodies(
    kind: str, spin: int, adjoint: bool
) -> None:
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding,
        DoubleFactorizationOneBody,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    source = (
        DoubleFactorizationOneBody(50, spin, 250, 12, num_bits_rot=8)
        if kind == "one"
        else DoubleFactorizationBlockEncoding(
            spin, 50, 250, num_bits_state_prep=12, num_bits_rot=8
        )
    )
    direction = source.adjoint() if adjoint else source
    root = direction.decompose_bloq()
    library = deserialize_component_library(
        bloqs_to_proto(root, max_depth=1).SerializeToString()
    )

    def declared(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None
        return FunctionRecord(name, (), registers=registers)

    program = resolve_component_program(
        library,
        0,
        {entry.declaration.name: declared for entry in library.entries[1:]},
    )
    assert verify_program_record(program) == ()
    assert len(program.functions[0].blocks[0].operations) == (9 if kind == "one" else 7)
    assert all(not function.blocks for function in program.functions[1:])
    identifiers = {
        function.name: (
            0 if function.name == "main" else int(function.name.rsplit("_", 1)[1])
        )
        for function in program.functions
    }
    instance_ids = tuple(instance.i for instance, _, _ in root.iter_bloqsoqs())
    exported = component_entry_from_program(
        program,
        "main",
        identifiers,
        library.entries[0].declaration,
        counts=library.entries[0].counts,
        instance_ids=instance_ids,
    )
    # This compares every ordered producer, consumer, shaped-register index,
    # leaf identity and occurrence, including bypassed source control wires.
    assert exported == library.entries[0]
    assert program_to_circuit(program).state == "unsupported_with_diagnostics"
    spoofed = replace(program, canonical_payload=CircuitIR(6, ()).to_dict())
    assert program_to_circuit(spoofed).state == "unsupported_with_diagnostics"
