"""Actual QROM base factories retain typed data, calls and physical ownership."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import product

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
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.lookup.qrom_table import QromTable
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.classical_lookup import lookup_classical_values
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


def table_program(table: QromTable, name: str) -> ProgramRecord:
    """Use the shared builder and existing physical table emitter."""
    builder = ComponentProgramBuilder(table.signature, name=name)
    words = [
        item for value in builder.inputs.values() for item in np.asarray(value).flat
    ]
    assert all(isinstance(word, RegisterBinding) for word in words)
    scratch = [builder.allocate() for _ in range(table.workspace_size)]
    all_words = [*words, *scratch]
    bits = np.concatenate([builder.split(word) for word in all_words])
    n_ports = sum(int(word.register.bitsize) for word in words)
    count = n_ports + len(scratch)
    gates = table.bind_packed(
        tuple(range(n_ports)), tuple(range(n_ports, count)), tuple(range(len(scratch)))
    )
    leaf = bit_gate_function(
        "lookup",
        (Register("bits", quantum_bit_type(), shape=(count,)),),
        CircuitIR(count, gates),
    )
    builder.define(leaf)
    returned = np.asarray(builder.call_dict("lookup", {"bits": bits})["bits"])
    restored = []
    offset = 0
    for word in all_words:
        size = int(word.register.bitsize)
        restored.append(
            builder.join(returned[offset : offset + size], word.register.dtype)
        )
        offset += size
    for word in restored[len(words) :]:
        builder.free(word)
    outputs = {}
    cursor = 0
    for register in table.signature:
        cells = int(np.prod(register.shape))
        output = np.asarray(restored[cursor : cursor + cells], dtype=object).reshape(
            register.shape
        )
        outputs[register.name] = output if register.shape else output[()]
        cursor += cells
    return builder.finalize(outputs)


@pytest.mark.parametrize("depth", (0, 1, 2))
@pytest.mark.parametrize("grouped,controls", tuple(product((False, True), (0, 1))))
def test_real_foreign_table_factory_binds_existing_emitter_and_all_branches(
    depth: int, grouped: bool, controls: int
) -> None:
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    data = np.array([[0, 1], [1, 0]]) if grouped else np.array([1, 0])
    shapes = ((2,),) if grouped else ((),)
    source = QROM.build_from_data(data, target_shapes=shapes, num_controls=controls)
    table = QromTable.from_data(data, target_shapes=shapes, num_controls=controls)
    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert bloqs_from_proto(proto)[0] == source
    captured = []

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        arrays = arguments["data_or_shape"]
        widths = arguments["target_bitsizes"]
        targets = arguments["target_shapes"]
        selection = arguments["selection_bitsizes"]
        assert isinstance(arrays, np.ndarray) and isinstance(widths, np.ndarray)
        assert isinstance(targets, np.ndarray) and isinstance(selection, np.ndarray)
        control_count = arguments["num_controls"]
        assert isinstance(control_count, int)
        recovered = QromTable.from_data(
            *arrays,
            target_bitsizes=tuple(map(int, widths)),
            target_shapes=tuple(tuple(map(int, row)) for row in targets),
            num_controls=control_count,
        )
        recovered = replace(recovered, selection_bitsizes=tuple(map(int, selection)))
        assert recovered == table and tuple(recovered.signature) == registers
        captured.append(recovered)
        return table_program(recovered, name)

    record = resolve_component_program(
        library, 0, {library.entries[0].declaration.name: factory}
    )
    assert captured == [table]
    assert record.functions[0].registers == tuple(native_signature(source.signature))
    n_ports = int(table.signature.n_qubits())
    sizes = tuple(int(reg.bitsize) for reg in table.signature for _ in reg.all_idxs())
    starts = np.cumsum((0, *sizes)).tolist()
    inputs = tuple(tuple(range(starts[i], starts[i + 1])) for i in range(len(sizes)))
    work = tuple(range(n_ports, n_ports + table.workspace_size))
    bound = bind_register_program(record, inputs, clean=work)
    assert bound.outputs == inputs
    measured = [g for g in bound.circuit.instructions if g.name == "measure"]
    slots = {g.wires[0]: g.metadata["classical_bit"] for g in measured}
    source_ports = {}
    offset = 0
    for reg in source.signature:
        size = int(reg.total_bits())
        source_ports[reg.name] = tuple(range(offset, offset + size))
        offset += size
    expected = reference_arithmetic_gates(
        source, source_ports, work, slots, cirq_style=True
    )
    assert_source(bound.circuit.instructions, expected)
    cost = get_cost_value(source, QECGatesCost())
    assert len(measured) == cost.measurement
    assert sum(g.name in {"t", "tdg"} for g in expected) == cost.total_t_count()
    # Every physical input has a nonzero complex coefficient. This simultaneously
    # exercises all controls, valid addresses and dirty target values.
    raw = torch.arange(1, (1 << n_ports) + 1, dtype=torch.float64)
    amplitude = torch.complex(raw, 2 - raw)
    amplitude /= amplitude.norm()
    initial = torch.zeros(1 << bound.circuit.n_wires, dtype=torch.complex128)
    initial[:: 1 << len(work)] = amplitude
    for history in product((0, 1), repeat=len(measured)):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        original = forced_branch(initial, expected, history)
        torch.testing.assert_close(actual, original, atol=3e-12, rtol=0)
        assert torch.count_nonzero(actual.reshape(-1, 1 << len(work))[:, 1:]) == 0
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


@pytest.mark.parametrize("depth", (0, 1, 2))
def test_original_shape_only_library_rejects_object_array(depth: int) -> None:
    from qualtran.bloqs.data_loading.qrom import QROM
    from qualtran.serialization.bloq import bloqs_to_proto

    source = QROM.build_from_bitsize(4, 2)
    with pytest.raises(ValueError, match="Object arrays cannot be saved"):
        bloqs_to_proto(source, max_depth=depth)


def test_data_factory_retains_symbolic_widths_until_explicit_physical_binding() -> None:
    from qualtran.bloqs.data_loading.qrom import QROM
    from sympy import symbols

    width, controls = symbols("b c", integer=True, positive=True)
    native = QromTable.from_data(
        [1, 0, 1, 0], target_bitsizes=(width,), num_controls=controls
    )
    source = QROM.build_from_data(
        [1, 0, 1, 0], target_bitsizes=(width,), num_controls=controls
    )
    assert source.has_data() and source.is_symbolic()
    assert native.datasets == ((1, 0, 1, 0),)
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in source.signature
    ]
    with pytest.raises(ValueError, match="Python integer dimensions"):
        native.bind_packed((), (), ())
    assert lookup_classical_values(source.data, (4,), {"selection": 3})["target0_"] == 0


@pytest.mark.parametrize(
    "case",
    ("inconsistent_rows", "wrong_target_axis", "scalar_sentinel", "negative_control"),
)
def test_constructor_dispositions_keep_unclosed_source_admission_boundaries(
    case: str,
) -> None:
    from qualtran.bloqs.data_loading.qrom import QROM

    if case == "inconsistent_rows":
        source = QROM.build_from_data([1, 2], [1, 2, 3])
        with pytest.raises(ValueError, match="same shape"):
            _ = source.data_shape
        with pytest.raises(ValueError, match="common selection"):
            QromTable.from_data([1, 2], [1, 2, 3])
    elif case == "wrong_target_axis":
        data = np.arange(6).reshape(3, 2)
        source = QROM.build_from_data(data, target_shapes=((3,),))
        assert source.data_shape == (3,) and source.target_shapes == ((3,),)
        with pytest.raises(ValueError, match="declared target axes"):
            QromTable.from_data(data, target_shapes=((3,),))
    elif case == "scalar_sentinel":
        source = QROM.build_from_data(1, [1, 2])
        assert source.data_shape == (2,) and source.selection_registers == ()
        with pytest.raises(ValueError, match="common selection"):
            QromTable.from_data(1, [1, 2])
    else:
        source = QROM.build_from_data([1, 0], num_controls=-1)
        assert source.control_registers[0].bitsize == -1
        with pytest.raises(ValueError, match="nonnegative widths"):
            QromTable.from_data([1, 0], num_controls=-1)
