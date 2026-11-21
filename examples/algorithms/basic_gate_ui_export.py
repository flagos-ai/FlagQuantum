# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 Basic Gates notebook catalog.

"""Export native basic-gate interfaces using existing execution and drawing owners.

Run ``python -m examples.algorithms.basic_gate_ui_export``. The catalog covers
nine complete notebook entries. Its graph is the atomic interface view. QEC
categories and emitted instructions are reported independently of that view.
"""

import sys
from collections import Counter
from collections.abc import Mapping
from functools import partial
from pathlib import Path

from sympy import Symbol

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
    global_phase_cost_key,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
    MeasurementCall,
    PhaseCall,
    ResourceCall,
    ZeroCostCall,
    identity_cost_key,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.control.multi_control import multi_control_h_ir
from flagquantum.compiler.registers.primitives.global_phase import global_phase_ir
from flagquantum.compiler.registers.primitives.identity import register_identity_ir
from flagquantum.compiler.registers.primitives.measurement import (
    measure_x_clear_ir,
    measure_z_clear_ir,
)
from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development.component_ui_export import (
    component_filename,
    export_component_ui,
)
from tools.development.notebook_autogen import NotebookSpec


def _example(name: str, call: ResourceCall) -> ComponentExample[ResourceCall]:
    return ComponentExample(partial(_identity, call), name, type(call))


def _identity(value: ResourceCall) -> ResourceCall:
    return value


# Group by actual native notebook ownership. Multiple operations deliberately
# share resource-call classes; the UI identity comes from their domain meaning.
DOCUMENTS = (
    ("T Gate", "register_t_gate", (_example("t_gate", PhaseCall("t")),)),
    ("Toffoli", "register_toffoli", (_example("toffoli", FixedTCall("toffoli")),)),
    ("Hadamard", "register_hadamard", (_example("hadamard", CliffordCall("h", 1)),)),
    (
        "Hadamard",
        "register_hadamard",
        (_example("chadamard", CONTROLLED_HADAMARD_COST),),
    ),
    ("CNOT", "register_cnot", (_example("cnot", CliffordCall("cx", 2)),)),
    (
        "Z, S, and CZ",
        "register_diagonal_gates",
        (_example("zgate", CliffordCall("z", 1)),),
    ),
    ("Z, S, and CZ", "register_diagonal_gates", (_example("s_gate", PhaseCall("s")),)),
    (
        "Z, S, and CZ",
        "register_diagonal_gates",
        (_example("cz", CliffordCall("cz", 2)),),
    ),
    ("Y Gate", "register_y_gate", (_example("y_gate", CliffordCall("y", 1)),)),
    ("Y Gate", "register_y_gate", (_example("cy_gate", CliffordCall("cy", 2)),)),
    (
        "Measurement",
        "register_measurement",
        (_example("meas_z", MeasurementCall("z")),),
    ),
    (
        "Measurement",
        "register_measurement",
        (_example("meas_x", MeasurementCall("x")),),
    ),
    (
        "Global Phase",
        "register_global_phase",
        (_example("global_phase", global_phase_cost_key(0.5)),),
    ),
    (
        "Identity Gate",
        "register_identity",
        (
            _example("identity_symb", identity_cost_key(Symbol("n"))),
            _example("identity", identity_cost_key(1)),
            _example("identity_n", identity_cost_key(4)),
        ),
    ),
)


_OPCODE_NAMES = {
    "t": "TGate",
    "s": "SGate",
    "h": "Hadamard",
    "cx": "CNOT",
    "y": "YGate",
    "cy": "CYGate",
    "z": "ZGate",
    "cz": "CZ",
    "toffoli": "Toffoli",
    "global_phase": "GlobalPhase",
    "identity": "Identity",
}


def name(component: object) -> str:
    """Give each native operation a domain name independent of its record class."""
    if component == CONTROLLED_HADAMARD_COST:
        return "CHadamard"
    if isinstance(component, MeasurementCall):
        return "MeasureX" if component.basis == "x" else "MeasureZ"
    if isinstance(component, (PhaseCall, CliffordCall, FixedTCall, ZeroCostCall)):
        return _OPCODE_NAMES[component.opcode]
    raise TypeError("Basic-gate UI requires a registered native gate identity")


def attributes(component: object) -> Mapping[str, object]:
    """Retain the frozen examples' explicit parameter vocabulary."""
    name(component)
    if isinstance(component, PhaseCall):
        return {"is_adjoint": component.adjoint}
    if isinstance(component, ZeroCostCall):
        if component.opcode == "identity":
            return {"bitsize": component.parameters[0]}
        return {"exponent": component.parameters[0], "eps": component.parameters[1]}
    return {}


def filename(component: object) -> str:
    return component_filename(attributes(component))


def signature(component: object) -> Signature:
    """Describe logical ports separately from physical measurement clearing."""
    label = name(component)
    if isinstance(component, MeasurementCall):
        return Signature(
            (
                Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
                Register("c", classical_bit_type(), side=RegisterSide.RIGHT),
            )
        )
    if label == "GlobalPhase":
        return Signature(())
    if isinstance(component, ZeroCostCall):
        width = component.parameters[0]
        return Signature((Register("q", opaque_register_type(width)),))
    if label == "Toffoli":
        return Signature(
            (
                Register("ctrl", quantum_bit_type(), (2,)),
                Register("target", quantum_bit_type()),
            )
        )
    if label in ("CNOT", "CYGate", "CHadamard"):
        return Signature.build(ctrl=1, target=1)
    if label == "CZ":
        return Signature.build(q1=1, q2=1)
    return Signature.build(q=1)


def operation(component: object) -> FieldOperation:
    """Bind concrete catalog examples to the existing append protocol.

    The scalar phase uses an arbitrary carrier wire. Measurement retains the
    classical outcome and physically clears its consumed quantum wire. Neither
    operation needs clean auxiliary quantum storage. Symbolic identity is a
    metadata/model example and requires a concrete width for execution.
    """
    label = name(component)
    if isinstance(component, MeasurementCall):
        emit = measure_x_clear_ir if component.basis == "x" else measure_z_clear_ir
        return FieldOperation(1, emit(0, 0))
    if isinstance(component, ZeroCostCall):
        if component.opcode == "global_phase":
            exponent = component.parameters[0]
            if not isinstance(exponent, (int, float)):
                raise ValueError("Global phase example requires a concrete exponent")
            return FieldOperation(1, global_phase_ir(exponent, anchor=0))
        width = component.parameters[0]
        if not isinstance(width, int):
            raise ValueError("Symbolic identity requires a concrete execution width")
        return FieldOperation(width, register_identity_ir(tuple(range(width))))
    if label == "CHadamard":
        return FieldOperation(2, multi_control_h_ir((0,), (1,), 1, (), ()))
    if isinstance(component, FixedTCall):
        return FieldOperation(3, pauli_gate_ir("x", 2, controls=(0, 1), values=(1, 1)))
    if isinstance(component, (PhaseCall, CliffordCall)):
        width = 2 if label in ("CNOT", "CYGate", "CZ") else 1
        if isinstance(component, PhaseCall) and component.adjoint:
            raise ValueError("This catalog contains forward phase examples")
        return FieldOperation(
            width, (Instruction(component.opcode, tuple(range(width))),)
        )
    raise TypeError("Basic-gate execution requires a registered native gate identity")


def score(component: object) -> object:
    """Draw fresh native interface lanes, including consumed quantum ports."""
    ports = signature(component)
    leaf = FunctionRecord(name(component), (), registers=tuple(ports))
    builder = ComponentProgramBuilder(ports, (leaf,), name="BasicGateInterface")
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))
    return register_score_data(component_register_score(record, "BasicGateInterface"))


def documentation_name(documentation: ComponentDocumentation[object]) -> str:
    return name(documentation.examples[0].make())


def documentation_text(documentation: ComponentDocumentation[object]) -> str:
    component = documentation.examples[0].make()
    label = name(component)
    return (
        f"# {label}\n\n"
        f"Native register interface `{signature(component)}`.\n\n"
        "Concrete examples bind through `operation(component)` and `circuit.append(op)`. "
        "Compiler resource keys own model estimates. Drawer owns interface layout. "
        "The exported graph contains atomic interfaces.\n\n"
        "Measurement consumes q, records c and clears the physical q wire. "
        "Global phase uses an arbitrary carrier. Symbolic identity is model-only "
        "until its width is bound. These examples require no clean auxiliary wires.\n"
    )


def sections() -> tuple[tuple[str, tuple[NotebookSpec, ...]], ...]:
    grouped: dict[str, NotebookSpec] = {}
    for title, stem, examples in DOCUMENTS:
        if title not in grouped:
            grouped[title] = NotebookSpec(
                title=title,
                module=sys.modules[__name__],
                component_specs=[],
                directory=str(Path(__file__).resolve().parents[1] / "notebooks"),
                path_stem=stem,
            )
        grouped[title].component_specs.append(
            ComponentDocumentation(
                component_type=examples[0].component_type,
                examples=tuple(
                    ComponentExample[object](
                        example.factory, example.name, example.component_type
                    )
                    for example in examples
                ),
                import_line="from examples.algorithms.basic_gate_ui_export import operation",
            )
        )
    return (("Basic Gates", tuple(grouped.values())),)


def export_example(destination: Path) -> None:
    """Publish every catalog example using native instance providers."""
    calls = [example.make() for _, _, examples in DOCUMENTS for example in examples]
    graph: dict[object, tuple[object, ...]] = dict.fromkeys(calls, ())
    export_component_ui(
        destination,
        sections(),
        graph,
        name=name,
        attributes=attributes,
        filename=filename,
        score=score,
        documentation_name=documentation_name,
        documentation_text=documentation_text,
    )
    for _, _, examples in DOCUMENTS:
        for example in examples:
            call = example.make()
            if example.name == "identity_symb":
                print(
                    example.name, "symbolic model", query_qec_gate_cost(call).asdict()
                )
                continue
            observed = dict(Counter(gate.name for gate in operation(call).instructions))
            print(example.name, "measured primitives", observed)
            print(example.name, "model categories", query_qec_gate_cost(call).asdict())


if __name__ == "__main__":
    export_example(Path("basic_gate_ui_export"))
