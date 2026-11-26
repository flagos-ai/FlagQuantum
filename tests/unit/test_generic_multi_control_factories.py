"""Generic source control constructors retain target and exact inverse ownership."""

from collections import Counter
from dataclasses import asdict, fields
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
)
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledRotationCall,
    PhaseCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_multi_control_factories import (
    multi_control_factories,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
VALUES = ((), (0,), (1,), (1, 1), (1, 0), (0, 1, 0))


@pytest.mark.parametrize("axis", ("X", "Y", "Z", "H", "S", "T"))
@pytest.mark.parametrize("values", VALUES)
@pytest.mark.parametrize("adjoint_depth", (0, 1, 2, 3))
def test_generic_fixed_targets_keep_source_call_graph_and_all_complex_branches(
    axis: str,
    values: tuple[int, ...],
    adjoint_depth: int,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cirq
    from qualtran._infra.adjoint import Adjoint

    gates = import_module("qualtran.bloqs.basic_gates")
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli

    resource = import_module("qualtran.resource_counting")

    with pytest.warns(DeprecationWarning, match="MultiControlPauli"):
        source = MultiControlPauli(
            values,
            getattr(
                gates,
                {
                    "X": "XGate",
                    "Y": "YGate",
                    "Z": "ZGate",
                    "H": "Hadamard",
                    "S": "SGate",
                    "T": "TGate",
                }[axis],
            )(),
        )
    for _ in range(adjoint_depth):
        source = Adjoint(source)
    count = len(values)
    logical = count + 2
    controls = tuple(range(2, logical))
    clean = tuple(range(logical, logical + max(count - 1, 0)))
    ports = {"target": (0,)} | ({"controls": controls} if count else {})
    inputs = (*((q,) for q in controls), (0,))
    observed = []
    for depth in (0, 1, 2):
        library = _library(source, depth)
        base = _factories()
        factories = {
            **base,
            **multi_control_factories(
                adjoint_factory=base["qualtran._infra.adjoint.Adjoint"]
            ),
        }
        with monkeypatch.context() as patch:
            patch.setitem(__import__("sys").modules, "qualtran", None)
            program = resolve_component_program(library, 0, factories)
            assert verify_program_record(program) == ()
            assert len(program.functions[0].blocks[0].operations) == 1
            bound = bind_register_program(program, inputs, clean=clean, borrowed=(1,))
        observed.append(bound.circuit.to_json())
        reference = reference_arithmetic_gates(
            source,
            ports,
            clean,
            MeasurementSlots(bound.circuit.instructions),
            keep_pauli_leaves=True,
            keep_exchange_leaves=True,
        )
        assert_source(bound.circuit.instructions, reference)
        for wire in range(bound.circuit.n_wires):
            assert [
                g.params for g in bound.circuit.instructions if wire in g.wires
            ] == [g.params for g in reference if wire in g.wires]
    assert observed[0] == observed[1] == observed[2]
    actual = Counter(op.name for op in bound.circuit.instructions)
    assert actual["measure"] == max(count - 1, 0)
    # The explicit array predicate is not the target's Toffoli shortcut.
    if count == 2:
        assert actual["measure"] == 1 and actual["t"] + actual["tdg"] == 4
    gate = getattr(cirq, axis)
    if adjoint_depth % 2:
        gate = cirq.inverse(gate)
    operation = gate.on(cirq.LineQubit(0))
    if values:
        operation = operation.controlled_by(
            *(cirq.LineQubit(q) for q in controls), control_values=values
        )
    expected = torch.tensor(
        cirq.Circuit(operation).unitary(qubit_order=cirq.LineQubit.range(logical)),
        dtype=torch.complex128,
    )
    measured = max(count - 1, 0)
    stride = 1 << len(clean)
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        appended = Circuit(bound.circuit.n_wires, dtype=torch.complex128).append(
            FieldOperation(bound.circuit.n_wires, bound.circuit.instructions)
        )
        restored = CircuitIR.from_json(appended.to_ir().to_json())
        for column in range(1 << logical):
            state = torch.zeros(1 << restored.n_wires, dtype=torch.complex128)
            state[column * stride] = 1
            target = torch.zeros_like(state)
            target[::stride] = expected[:, column] / 2 ** (measured / 2)
            for history in product((0, 1), repeat=measured):
                torch.testing.assert_close(
                    forced_branch(state, restored.instructions, history),
                    target,
                    atol=4e-12,
                    rtol=0,
                )
    if count:
        positive_key = (
            CONTROLLED_HADAMARD_COST
            if axis == "H"
            else (
                ControlledRotationCall(
                    PhaseCall(
                        "s" if axis == "S" else "t", adjoint=bool(adjoint_depth % 2)
                    ),
                    (1,),
                )
                if axis in ("S", "T")
                else {
                    "X": CliffordCall("cx", 2),
                    "Y": CliffordCall("cy", 2),
                    "Z": CliffordCall("cz", 2),
                }[axis]
            )
        )
        positive = query_qec_gate_cost(positive_key)
        model = controlled_component_cost(
            ComponentCost("positive", 2, GateCost(**asdict(positive))),
            (ControlRegister(1, values),),
            target_width=1,
        )
        native = query_qec_gate_cost(model)
    else:
        native = query_qec_gate_cost(
            PhaseCall("s" if axis == "S" else "t", adjoint=bool(adjoint_depth % 2))
            if axis in ("S", "T")
            else {
                "X": CliffordCall("x", 1),
                "Y": CliffordCall("y", 1),
                "Z": CliffordCall("z", 1),
                "H": CliffordCall("h", 1),
            }[axis]
        )
    expected_cost = resource.get_cost_value(source, resource.QECGatesCost())
    assert asdict(native) == {
        field.name: getattr(expected_cost, field.name) for field in fields(QECGateCost)
    }


@pytest.mark.parametrize("values", VALUES)
@pytest.mark.parametrize("inverse", (False, True))
def test_explicit_target_owner_keeps_collision_private_work_and_exact_inverse(
    values: tuple[int, ...], inverse: bool, locked_source: Any
) -> None:

    upstream = import_module("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlPauli

    from flagquantum.compiler.formats.legacy_word_factories import _rename_program
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.compiler.registers.component_control import (
        _stored_control_adder,
        control_component_program,
    )
    from flagquantum.compiler.registers.primitives.controlled_rotation import (
        controlled_z_power_ir,
    )
    from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
    from flagquantum.core.ir.program.register_types import quantum_bit_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature
    from tests.unit.test_component_control import (
        add_h,
        add_phase,
        body,
        fixture_program,
    )
    from tests.unit.test_multiplexed_source_binding import logical_operator

    template, _ = fixture_program(1)
    definitions = (
        *body(
            "cp",
            (Register("q", quantum_bit_type(), (2,)),),
            CircuitIR(
                3,
                tuple(
                    Instruction("t", op.wires) if op.name == "phase" else op
                    for op in controlled_z_power_ir(0, 1, 0.25, 2, 0)
                ),
            ),
        ),
        *body(
            "p",
            (Register("q", quantum_bit_type()),),
            CircuitIR(1, (Instruction("t", (0,)),)),
        ),
        *(
            f
            for f in template.functions[1:]
            if f.name not in ("p", "cp") and not f.name.startswith(("p_", "cp_"))
        ),
    )
    bit = quantum_bit_type()
    signature = Signature((Register("ctrl", bit),))
    forward = ComponentProgramBuilder(signature, definitions, name="target_body")
    value = forward.inputs["ctrl"]
    for callee in ("h", "p"):
        value = forward.call_dict(callee, {"q": value})["q"]
    module = forward.finalize({"ctrl": value})
    child = module.functions[0]
    inverse_defs = (
        *body(
            "p_inverse",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("tdg", (0,)),)),
        ),
        *body(
            "cp",
            (Register("q", bit, (2,)),),
            CircuitIR(
                3,
                tuple(
                    Instruction("tdg", op.wires) if op.name == "phase" else op
                    for op in controlled_z_power_ir(0, 1, -0.25, 2, 0)
                ),
            ),
        ),
        *(f for f in definitions if f.name != "cp" and not f.name.startswith("cp_")),
    )
    backward = ComponentProgramBuilder(signature, inverse_defs, name="target_inverse")
    value = backward.inputs["ctrl"]
    for callee in ("p_inverse", "h"):
        value = backward.call_dict(callee, {"q": value})["q"]
    inverse_module = backward.finalize({"ctrl": value})

    def owner(function: FunctionRecord) -> tuple[ProgramRecord, Any]:
        record = module if function == child else inverse_module
        assert function == record.functions[0]
        controlled = _rename_program(
            control_component_program(
                record,
                {"h": add_h, "p" if record == module else "p_inverse": add_phase},
                entry=function.name,
            ),
            function.name + "_positive",
        )
        return controlled, _stored_control_adder(controlled.functions[0])

    def exact_inverse(function: FunctionRecord) -> ProgramRecord:
        assert function == child
        return inverse_module

    factories = multi_control_factories(
        control_system=owner,
        target_adjoint=exact_inverse,
        adjoint_factory=_factories()["qualtran._infra.adjoint.Adjoint"],
    )
    program = factories["qualtran.bloqs.mcmt.multi_control_pauli.MultiControlPauli"](
        "wrapper", None, {"cvs": values, "target_bloq": child}
    )
    assert isinstance(program, ProgramRecord)
    if inverse:
        program = factories["qualtran._infra.adjoint.Adjoint"](
            "inverse", None, {"subbloq": program.functions[0]}
        )
        assert isinstance(program, ProgramRecord)
    else:
        program = ProgramRecord((*program.functions, *module.functions))
    assert verify_program_record(program) == ()
    count = len(values)
    inputs = (*((q,) for q in range(count)), (count,))
    clean = tuple(range(count + 1, count + 1 + (count if count else 0)))
    bound = bind_register_program(
        program, inputs, clean=clean, entry=program.functions[0].name
    )
    source_builder, soqs = upstream.BloqBuilder.from_signature(
        upstream.Signature.build(ctrl=1)
    )
    source_value = soqs["ctrl"]
    sequence = (
        (gates.ZPowGate(-0.25), gates.Hadamard())
        if inverse
        else (gates.Hadamard(), gates.ZPowGate(0.25))
    )
    for gate in sequence:
        source_value = source_builder.add(gate, q=source_value)
    target = source_builder.finalize(ctrl=source_value)
    with pytest.warns(DeprecationWarning):
        source = MultiControlPauli(values, target)
    reference = reference_arithmetic_gates(
        source,
        {"target": (count,)} | ({"controls": tuple(range(count))} if count else {}),
        clean,
        MeasurementSlots(bound.circuit.instructions),
        keep_pauli_leaves=True,
    )
    assert_source(bound.circuit.instructions, reference)
    observed = logical_operator(bound.circuit, count + 1)
    expected = torch.tensor(
        source.tensor_contract().reshape(1 << (count + 1), -1), dtype=torch.complex128
    )
    torch.testing.assert_close(observed, expected, atol=8e-12, rtol=0)
    assert sum(op.name == "measure" for op in bound.circuit.instructions) == (
        count if count else 0
    )


def test_generic_constructor_rejects_unowned_targets_and_keeps_fallback_scoped() -> (
    None
):
    from dataclasses import replace

    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )
    from flagquantum.compiler.registers.component_gate_function import bit_gate_function
    from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
    from flagquantum.core.ir.program.registers import Register, RegisterSide
    from flagquantum.core.ir.program.signatures import Signature
    from flagquantum.errors import SerializationError

    key = "qualtran.bloqs.mcmt.multi_control_pauli.MultiControlPauli"
    inverse_key = "qualtran._infra.adjoint.Adjoint"
    leaf = fixed_gate_factories()["qualtran.bloqs.basic_gates.x_basis.XGate"](
        "x", None, {}
    )
    assert isinstance(leaf, FunctionRecord) and leaf.registers is not None
    delegated = []

    def fallback(name: str, registers: Any, arguments: Any) -> FunctionRecord:
        delegated.append((name, registers, arguments))
        return replace(leaf, name=name)

    catalogue = multi_control_factories(adjoint_factory=fallback)
    for arguments in (
        {},
        {"cvs": (1,), "target_bloq": 7},
        {"cvs": (2,), "target_bloq": leaf},
    ):
        with pytest.raises(SerializationError):
            catalogue[key]("invalid", None, arguments)
    for register in (
        Register(leaf.registers[0].name, leaf.registers[0].dtype, (1,)),
        Register(
            leaf.registers[0].name, leaf.registers[0].dtype, side=RegisterSide.RIGHT
        ),
    ):
        with pytest.raises(SerializationError, match="scalar quantum-bit"):
            catalogue[key](
                "invalid",
                None,
                {"cvs": (1,), "target_bloq": replace(leaf, registers=(register,))},
            )
    unknown = bit_gate_function(
        "unknown",
        tuple(Signature.build(q=1)),
        CircuitIR(1, (Instruction("rx", (0,), {"theta": 0.3}),)),
    )
    with pytest.raises(SerializationError, match="positive-control provider"):
        catalogue[key]("invalid", None, {"cvs": (1,), "target_bloq": unknown})
    # Zero controls can call an arbitrary owned body; inversion still needs an owner.
    root = catalogue[key]("zero", None, {"cvs": (), "target_bloq": unknown})
    assert isinstance(root, ProgramRecord)
    with pytest.raises(SerializationError, match="exact target inverse"):
        catalogue[inverse_key]("inverse", None, {"subbloq": root.functions[0]})
    assert not delegated
    result = multi_control_factories(adjoint_factory=fallback)[inverse_key](
        "foreign", None, {"subbloq": root.functions[0]}
    )
    assert isinstance(result, FunctionRecord) and result.name == "foreign"
    assert len(delegated) == 1
    assert inverse_key not in multi_control_factories()
