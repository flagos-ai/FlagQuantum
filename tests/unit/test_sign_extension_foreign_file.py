"""Complete signed-prefix factories use native interchange and SDK ownership."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.bits.sign_extension_component import (
    sign_extension_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_optree
from flagquantum.ecosystem.cirq._register_program_export import (
    export_bound_register_cirq,
)
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")
cirq = pytest.importorskip("cirq")


@pytest.mark.parametrize("factory_name", ("_sign_extend", "_sign_truncate"))
def test_actual_factories_preserve_full_object_library(factory_name: str) -> None:
    from qualtran.bloqs.arithmetic.conversions import sign_extension
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto
    from qualtran.testing import assert_bloq_example_serializes

    factory = getattr(sign_extension, factory_name)
    source = factory.make()
    assert_bloq_example_serializes(factory)
    for depth in (0, 1, 2):
        original = bloqs_to_proto(source, max_depth=depth)
        wire = original.SerializeToString(deterministic=True)
        native = deserialize_component_library(wire)
        assert serialize_component_library(native) == wire
        assert (
            deserialize_component_library(serialize_component_library(native)) == native
        )
        assert source in bloqs_from_proto(original)
        root = native.entries[0].declaration
        assert root.registers is not None
        assert all(reg.dtype.encoding == "signed" for reg in root.registers)
        assert tuple(reg.bitsize for reg in root.registers) == tuple(
            reg.bitsize for reg in source.signature
        )


@pytest.mark.parametrize("ones", (False, True))
@pytest.mark.parametrize("truncate", (False, True))
def test_symbolic_declarations_preserve_original_type_graph(
    ones: bool, truncate: bool
) -> None:
    from qualtran import QInt, QIntOnesComp
    from qualtran.bloqs.arithmetic.conversions.sign_extension import SignExtend
    from qualtran.serialization.bloq import bloqs_to_proto
    from sympy import Symbol

    n = Symbol("n", integer=True, positive=True)
    dtype = QIntOnesComp if ones else QInt
    source: Any = SignExtend(dtype(n), dtype(n + 2))
    if truncate:
        source = source.adjoint()
    wire = bloqs_to_proto(source, max_depth=0).SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    declaration = library.entries[0].declaration
    assert serialize_component_library(library) == wire

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        assert registers is not None
        assert arguments["inp_dtype"] == registers[0].dtype
        assert arguments["out_dtype"] == registers[1].dtype
        return FunctionRecord(name, (), registers=registers)

    resolved = resolve_component_program(library, 0, {declaration.name: factory})
    assert resolved.functions[0].registers == declaration.registers
    encoding = "ones_complement" if ones else "signed"
    with pytest.raises(ValueError, match="concrete"):
        sign_extension_program(
            RegisterType(encoding, n), RegisterType(encoding, n + 2), adjoint=truncate
        )


@pytest.mark.parametrize("encoding", ("signed", "ones_complement"))
@pytest.mark.parametrize("width,extra", ((2, 1), (2, 3), (3, 2), (8, 8)))
@pytest.mark.parametrize("adjoint", (False, True))
def test_complete_sided_sdk_roundtrip_stream_aliases_and_complex_prefix(
    encoding: str,
    width: int,
    extra: int,
    adjoint: bool,
) -> None:
    graph = sign_extension_program(
        RegisterType(encoding, width),
        RegisterType(encoding, width + extra),
        adjoint=adjoint,
    )
    signature = Signature(graph.functions[0].registers or ())
    incoming = tuple(range(width + extra if adjoint else width))
    workspace = tuple(range(width, width + extra))
    ownership = tuple((wire, False) for wire in range(extra)) if adjoint else ()
    kwargs: dict[str, Any] = dict(
        clean=() if adjoint else workspace,
        allow_workspace_outputs=not adjoint,
        input_ownership=ownership,
    )
    bound = bind_register_program(graph, (incoming,), **kwargs)
    manager = cirq.GreedyQubitManager(prefix="signed_prefix", maximize_reuse=True)
    inputs = np.asarray(tuple(reversed(manager.qalloc(len(incoming)))), dtype=object)
    exported, outputs = export_bound_register_cirq(
        bound, signature, input_quregs={"x": inputs}, qubit_manager=manager
    )
    imported = import_cirq_register_optree(
        exported.circuit,
        signature=signature,
        in_quregs={"x": inputs},
        out_quregs=outputs,
    )
    restored = bind_register_program(imported, (incoming,), **kwargs)
    # Fresh output storage follows first use on import. The logical port map
    # independently identifies every SDK wire across that allocation choice.
    wire_map = dict(zip(incoming, incoming, strict=True))
    wire_map.update(zip(bound.outputs[0], restored.outputs[0], strict=True))
    assert set(wire_map) == set(wire_map.values()) == set(range(width + extra))
    assert len(set(wire_map.values())) == width + extra
    assert all(wire_map[wire] == wire for wire in incoming)
    mapped_gates = tuple(
        replace(gate, wires=tuple(wire_map[wire] for wire in gate.wires))
        for gate in bound.circuit.instructions
    )
    assert len(restored.circuit.instructions) == len(mapped_gates)
    # The SDK schedules independent fanout gates into moments. Every wire's
    # complete ordered operations preserve all gate dependencies and operands.
    for wire in range(width + extra):
        assert tuple(
            gate for gate in restored.circuit.instructions if wire in gate.wires
        ) == tuple(gate for gate in mapped_gates if wire in gate.wires)
    if extra <= 3:
        assert restored.circuit.instructions == mapped_gates
    assert restored.circuit.n_wires == bound.circuit.n_wires == width + extra
    assert CircuitIR.from_json(restored.circuit.to_json()) == restored.circuit
    assert all(gate.name == "cx" for gate in restored.circuit.instructions)
    assert len(restored.circuit.instructions) == 2 * extra - 1
    expected_qids = (
        inputs[extra:] if adjoint else np.concatenate((outputs["y"][:extra], inputs))
    )
    np.testing.assert_array_equal(outputs["y"], expected_qids)
    # All input Qids were leased by this manager, including consumed prefixes.
    if not adjoint:
        assert set(outputs["y"][:extra]).isdisjoint(inputs)
    manager.qfree(outputs["y"])
    reclaimed = manager.qalloc(width + extra)
    assert set(reclaimed) == set(inputs) | set(outputs["y"])
    manager.qfree(reclaimed)
    if width > 3:
        return
    total = width + extra + 1
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    data = torch.arange(1, (1 << (width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(data, 5 - 3 * data)
    amplitudes /= amplitudes.norm()
    for raw, spectator in product(range(1 << width), (0, 1)):
        sign = raw >> (width - 1)
        prefix = ((1 << extra) - 1) if sign else 0
        narrow = ((raw << extra) << 1) | spectator
        wide = (((prefix << width) | raw) << 1) | spectator
        extended = (((raw << extra) | prefix) << 1) | spectator
        if adjoint:
            initial[wide] = amplitudes[2 * raw + spectator]
            expected[(raw << 1) | spectator] = amplitudes[2 * raw + spectator]
        else:
            initial[narrow] = amplitudes[2 * raw + spectator]
            expected[extended] = amplitudes[2 * raw + spectator]
    indices = torch.arange(1 << total)
    mapped = indices & 1
    for wire, physical in wire_map.items():
        mapped |= ((indices >> (total - wire - 1)) & 1) << (total - physical - 1)
    mapped_initial = torch.zeros_like(initial)
    mapped_expected = torch.zeros_like(expected)
    mapped_initial[mapped] = initial
    mapped_expected[mapped] = expected
    actual = forced_branch(mapped_initial, restored.circuit.instructions, ())
    torch.testing.assert_close(actual, mapped_expected, rtol=0, atol=3e-12)
    if adjoint:
        assert torch.count_nonzero(actual.reshape(1 << extra, -1)[1:]) == 0
