"""Exercise identity/exchange library constructors through native append."""

from collections import Counter
from collections.abc import Mapping
from importlib import import_module
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledSwapCall,
    identity_cost_key,
)
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_exchange_factories import exchange_factories
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_legacy_fixed_gate_factories import _library
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
CASES = (
    ("Identity", 1),
    ("Identity", 3),
    ("TwoBitSwap", 1),
    ("TwoBitCSwap", 1),
    ("Swap", 1),
    ("Swap", 2),
    ("CSwap", 1),
    ("CSwap", 2),
)


@pytest.mark.parametrize("example", ("_identity", "_identity_n", "_identity_symb"))
def test_complete_original_identity_factories_and_serialized_depths(
    example: str,
) -> None:
    identity = import_module("qualtran.bloqs.basic_gates.identity")
    codec = import_module("qualtran.serialization.bloq")
    source = getattr(identity, example).make()
    for depth in (0, 1, 2):
        library = deserialize_component_library(
            codec.bloqs_to_proto(source, max_depth=depth).SerializeToString()
        )
        program = resolve_component_program(library, 0, exchange_factories())
        assert verify_program_record(program) == ()
        if example == "_identity_symb":
            assert program.functions[0].blocks == ()
            with pytest.raises(ValueError):
                bind_register_program(program, ((0,),))
        else:
            ports = (tuple(reversed(range(source.bitsize))),)
            bound = bind_register_program(program, ports)
            assert bound.outputs == ports and not bound.events
            assert tuple(
                (g.name, g.wires) for g in bound.circuit.instructions
            ) == tuple(("i", (wire,)) for wire in ports[0])


def _source(kind: str, width: int) -> Any:
    gates = import_module("qualtran.bloqs.basic_gates")
    constructor = getattr(gates, kind)
    return constructor() if kind.startswith("TwoBit") else constructor(width)


@pytest.mark.parametrize("kind,width", CASES)
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_complete_complex_operator_round_trip_work_and_resources(
    kind: str, width: int, dtype: torch.dtype
) -> None:
    source = _source(kind, width)
    program = resolve_component_program(_library(source), 0, exchange_factories())
    assert verify_program_record(program) == ()
    size = source.signature.n_qubits()
    wires = tuple(reversed(range(1, size + 1)))
    offset = 0
    ports = []
    for reg in source.signature:
        ports.append(wires[offset : offset + reg.total_bits()])
        offset += reg.total_bits()
    bound = bind_register_program(program, tuple(ports))
    assert bound.outputs == tuple(ports) and not bound.events
    values = torch.arange(1 << (size + 1), dtype=torch.float64)
    amplitudes = torch.complex(1 + values, 2 - 3 * values).to(dtype)
    initial = torch.zeros(1 << (size + 2), dtype=dtype)
    initial[::2] = amplitudes / amplitudes.norm()
    circuit = Circuit(size + 2, inputs=initial, dtype=dtype)
    circuit.append(FieldOperation(size + 2, bound.circuit.instructions))
    cirq = import_module("cirq")
    np = import_module("numpy")
    qubits = tuple(cirq.LineQubit.range(size + 2))
    offset = 0
    quregs = {}
    for reg in source.signature:
        count = reg.total_bits()
        quregs[reg.name] = np.array(
            [qubits[w] for w in wires[offset : offset + count]]
        ).reshape((*reg.shape, reg.bitsize))
        offset += count
    operation, _ = source.as_cirq_op(cirq.SimpleQubitManager(), **quregs)
    result = cirq.Simulator(
        dtype=np.complex64 if dtype == torch.complex64 else np.complex128
    ).simulate(
        cirq.Circuit(operation), initial_state=initial.numpy(), qubit_order=qubits
    )
    tolerance = 5e-7 if dtype == torch.complex64 else 3e-12
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.from_numpy(result.final_state_vector),
        rtol=0,
        atol=tolerance,
    )
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    circuit.append(FieldOperation(size + 2, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().flatten(), initial, rtol=0, atol=tolerance
    )
    plain_ports = []
    offset = 0
    for reg in source.signature:
        plain_ports.append(tuple(range(offset, offset + reg.total_bits())))
        offset += reg.total_bits()
    plain = bind_register_program(program, tuple(plain_ports))
    operator = Circuit(size, dtype=torch.complex128).append(
        FieldOperation(size, plain.circuit.instructions)
    )
    torch.testing.assert_close(
        get_unitary(operator),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        rtol=0,
        atol=3e-12,
    )
    counts = Counter(g.name for g in plain.circuit.instructions)
    if "CSwap" in kind:
        assert counts == {
            "cx": 8 * width,
            "h": 2 * width,
            "t": 4 * width,
            "tdg": 3 * width,
        }
    else:
        assert counts == {"i" if kind == "Identity" else "swap": width}
    if kind != "Identity":
        emitter = import_register_emitter(source)
        for snapshot in emitter.snapshots:
            assert_source(plain.circuit.instructions, snapshot.instructions)
    assert counts["measure"] == counts["reset"] == 0
    resource = import_module("qualtran.resource_counting")
    key = (
        identity_cost_key(width)
        if kind == "Identity"
        else (
            ControlledSwapCall(width)
            if "CSwap" in kind
            else CliffordCall("swap_bit", 2)
        )
    )
    native_model = query_qec_gate_cost(key)
    original_model = resource.get_cost_value(source, resource.QECGatesCost())
    factor = width if kind in ("Swap", "TwoBitSwap") else 1
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert factor * getattr(native_model, field) == getattr(original_model, field)


@pytest.mark.parametrize("kind", ("Identity", "Swap", "CSwap"))
def test_defaults_symbolic_metadata_and_physical_width_diagnostics(kind: str) -> None:
    key = f"qualtran.bloqs.basic_gates.{'identity' if kind == 'Identity' else 'swap'}.{kind}"
    factory = exchange_factories()[key]
    if kind == "Identity":
        assert isinstance(factory("default", None, {}), FunctionRecord)
    else:
        with pytest.raises(SerializationError, match="requires bitsize"):
            factory("missing", None, {})
    for value in (sympy.Symbol("n"), sympy.Integer(2)):
        declaration = factory("symbolic", None, {"bitsize": value})
        assert isinstance(declaration, FunctionRecord) and declaration.blocks == ()
    for value in (0, -1):
        with pytest.raises(ValueError):
            factory("invalid", None, {"bitsize": value})
    for value in (True, 1.5, "2"):
        with pytest.raises(SerializationError, match="bitsize"):
            factory("invalid", None, {"bitsize": value})


@pytest.mark.parametrize("kind,width", CASES)
def test_explicit_factory_signature_validation_and_replacement(
    kind: str, width: int
) -> None:
    source = _source(kind, width)
    library = _library(source)
    mapping = exchange_factories()
    key = library.entries[0].declaration.name
    factory = mapping[key]
    arguments: dict[str, ResolvedArgument] = (
        {} if kind.startswith("TwoBit") else {"bitsize": width}
    )
    with pytest.raises(SerializationError, match="Unexpected constructor"):
        factory("bad", None, {"unknown": 1})
    with pytest.raises(SerializationError, match="Invalid register signature"):
        factory("bad", tuple(Signature.build(other=1)), arguments)

    def replacement(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        native = factory("replacement_leaf", registers, arguments)
        graph = (
            native if isinstance(native, ProgramRecord) else ProgramRecord((native,))
        )
        root = graph.functions[0]
        assert root.registers is not None
        builder = ComponentProgramBuilder(
            Signature(root.registers), graph.functions, name=name
        )
        return builder.finalize(builder.call_dict(root.name, builder.inputs))

    expected = resolve_component_program(library, 0, mapping)
    mapping[key] = replacement
    actual = resolve_component_program(library, 0, mapping)
    ports = []
    offset = 0
    for reg in source.signature:
        ports.append(tuple(range(offset, offset + reg.total_bits())))
        offset += reg.total_bits()
    assert bind_register_program(actual, tuple(ports)) == bind_register_program(
        expected, tuple(ports)
    )
    mapping.clear()
    assert len(exchange_factories()) == 5


def test_multiple_word_libraries_namespace_helpers_without_consumer_changes() -> None:
    q = import_module("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    builder, inputs = q.BloqBuilder.from_signature(
        q.Signature.build(ctrl=1, a=2, b=2, c=3, d=3)
    )
    ctrl, a, b, c, d = (inputs[k] for k in ("ctrl", "a", "b", "c", "d"))
    a, b = builder.add(gates.Swap(2), x=a, y=b)
    c, d = builder.add(gates.Swap(3), x=c, y=d)
    ctrl, a, b = builder.add(gates.CSwap(2), ctrl=ctrl, x=a, y=b)
    c = builder.add(gates.Identity(3), q=c)
    original = builder.finalize(ctrl=ctrl, a=a, b=b, c=c, d=d)
    codec = import_module("qualtran.serialization.bloq")
    factories = {
        **bookkeeping_factories(),
        **fixed_gate_factories(),
        **exchange_factories(),
    }
    previous = None
    for depth in (1, 2, 3):
        library = deserialize_component_library(
            codec.bloqs_to_proto(original, max_depth=depth).SerializeToString()
        )
        program = resolve_component_program(library, 0, factories)
        assert verify_program_record(program) == ()
        bound = bind_register_program(
            program, ((0,), (1, 2), (3, 4), (5, 6, 7), (8, 9, 10))
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts == {"swap": 5, "cx": 16, "h": 4, "t": 8, "tdg": 6, "i": 3}
        assert not bound.events
        if previous is not None:
            assert bound.circuit.instructions == previous
        previous = bound.circuit.instructions
