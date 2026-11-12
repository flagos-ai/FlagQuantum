"""Wire symbols for the native classical/tensor walkthrough's diagrams."""

from dataclasses import replace

from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function_data,
)
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    PortSide,
    SymbolClass,
    WireSymbolSupplier,
    _port_symbol,
    register_score_data,
)


def style(function: FunctionRecord, step: RegisterStep) -> RegisterStep:
    gate = {"add_cx": "cx", "add_and": "and", "add_and_adjoint": "and"}.get(
        function.name
    )
    return replace(step, gate=gate, control_values=(1, 1) if gate == "and" else None)


def symbols(
    record: ProgramRecord, vectors: dict[str, tuple[int, bool]], kind: str | None = None
) -> WireSymbolSupplier:
    functions = {f.name: f for f in record.functions}

    def supply(
        step: RegisterStep, register: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register is None:
            return "Text", {
                "text": "X" if step.name == "main" and kind == "xgate" else "",
                "fontsize": 10,
            }
        assert side is not None
        if step.name in vectors:
            value, state = vectors[step.name]
            return ("LarrowTextBox" if state else "RarrowTextBox"), {"text": str(value)}
        if step.name == "main" and kind in {"xgate", "cnot", "toffoli"}:
            gate = {"xgate": "x", "cnot": "cx", "toffoli": "ccx"}[kind]
            return _port_symbol(replace(step, gate=gate), register, side)
        if step.name == "main" and kind == "add":
            return "TextBox", {"text": "a" if register == step.inputs[0] else "a+b"}
        function = functions[step.name]
        if all(
            op.opcode in {"split_register", "join_register"}
            for op in function.blocks[0].operations
        ):
            ports = Signature(function.registers or ())
            selected = ports.lefts() if side == "LEFT" else ports.rights()
            names = step.inputs if side == "LEFT" else step.outputs
            indexed = [(reg, idx) for reg in selected for idx in reg.all_idxs()]
            reg, index = indexed[names.index(register)]
            text = "[" + ", ".join(map(str, index)) + "]" if reg.shape else " "
            return ("RarrowTextBox" if side == "LEFT" else "LarrowTextBox"), {
                "text": text
            }
        return _port_symbol(step, register, side)

    return supply


def crosscheck_score_data(record: ProgramRecord, kind: str) -> MusicalScoreData:
    """Compose the closed example's native layout and declared port symbols."""
    vectors = {
        function.name: value
        for function in record.functions
        if (value := integer_basis_function_data(function)) is not None
    }
    return register_score_data(
        component_register_score(record, "classical_vectors", style=style),
        wire_symbol=symbols(record, vectors, kind),
    )
