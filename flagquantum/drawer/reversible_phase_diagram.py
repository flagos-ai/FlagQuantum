# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Circuit::write_text_diagram_to.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/circuit/circuit_diagram.cc

"""Render physical reversible-phase records as a text circuit diagram."""

from ..compiler.formats.kmb import KmxRecord
from ..compiler.formats.kmx_circuit import KmxCircuit, read_kmx_circuit

_UNCERTAIN = b"\x01"
_INLINE_GUARD = {"X", "Z", "CCX", "CCZ"}
_MERGED_CONTROLS = {"CX", "CZ"}
_META_NAMES = {"PUSH_CONDITION": "push_cond", "POP_CONDITION": "pop_cond", "NEG": "neg"}
_BIT_UPDATES = {"BIT_INVERT": "^=1", "BIT_STORE0": "=0", "BIT_STORE1": "=1"}
_GLYPHS = {"=": "═", "-": "─", "@": "●", ">": "⟩"}


def _wire_row(qubit: int) -> int:
    return qubit * 2 + 2


class _TextDiagram:
    """Track source column occupancy and the current state of each wire."""

    def __init__(self, qubits: int) -> None:
        self.rows = [bytearray() for _ in range(qubits * 2 + 2)]
        self.fillers = [b" "] * len(self.rows)
        for qubit in range(qubits):
            index = _wire_row(qubit)
            self.rows[index].extend(f"q{qubit}: ".encode() + _UNCERTAIN)
            self.fillers[index] = _UNCERTAIN
        width = max(map(len, self.rows))
        for row in self.rows:
            row[:0] = b" " * (width - len(row))
        self.used_rows: set[int] = set()
        self.read_bits: set[int] = set()
        self.written_bits: set[int] = set()
        self.next_meta: int | None = None
        self.control_group: list[KmxRecord] = []
        self.control_targets: set[int] = set()

    def flush_column(self) -> None:
        width = max(map(len, self.rows))
        for row, filler in zip(self.rows, self.fillers, strict=True):
            row.extend(filler * (width - len(row) + 1))
        self.used_rows.clear()
        self.read_bits.clear()
        self.written_bits.clear()

    def reserve_wires(self, wires: tuple[int, ...], condition: int | None) -> None:
        low, high = _wire_row(min(wires)), _wire_row(max(wires))
        reserved = range(low - 2 if condition is not None else low, high + 1)
        if not self.used_rows.isdisjoint(reserved):
            self.flush_column()
        self.used_rows.update(reserved)
        for row in self.rows[low : high + 1]:
            row.extend(b"|")
        if condition is not None:
            self.rows[low - 1].extend(f"if(b{condition})".encode())

    def flush_controls(self) -> None:
        if not self.control_group:
            return
        control = self.control_group[0].qubits[0]
        self.reserve_wires((control, *self.control_targets), None)
        self.rows[_wire_row(control)][-1:] = b"@"
        for record in self.control_group:
            row = _wire_row(record.qubits[-1])
            if record.opcode == "CX":
                self.fillers[row] = b"-"
            self.rows[row][-1:] = record.opcode[-1].encode()
            if record.condition_bit is not None:
                self.rows[row].extend(f"**b{record.condition_bit}".encode())
        self.control_group.clear()
        self.control_targets.clear()

    def end_meta(self) -> None:
        if self.next_meta is not None:
            self.next_meta = None
            self.flush_column()

    def add_meta(self, label: str) -> None:
        self.flush_controls()
        index = self.next_meta
        if index is None or index >= len(self.rows):
            self.flush_column()
            index = 1
        self.rows[index].extend(label.encode())
        self.next_meta = index + 2

    def add_registers(self, circuit: KmxCircuit) -> None:
        for index, register in enumerate(circuit.registers):
            name = register.name or f"reg{index}"
            for member, (kind, wire) in enumerate(register.members):
                label = f"{name}[{member}]"
                if kind == "b":
                    if wire in self.read_bits:
                        self.flush_column()
                    self.add_meta(f"{label}=b{wire}")
                else:
                    self.end_meta()
                    self.flush_controls()
                    self.reserve_wires((wire,), None)
                    self.rows[_wire_row(wire)][-1:] = label.encode()

    def add_operation(self, record: KmxRecord) -> None:
        if record.tag is not None:
            return
        opcode = record.opcode
        target_bit, condition = record.result_bit, record.condition_bit
        if condition in self.written_bits or target_bit in self.read_bits:
            self.flush_column()
        if not record.qubits:
            label = _META_NAMES.get(opcode, "UNEXPECTED_NON_QUBIT_OPERATION")
            if opcode in _BIT_UPDATES:
                label = f"b{target_bit}{_BIT_UPDATES[opcode]}"
            if condition is not None:
                label += f" if b{condition}"
            self.add_meta(label)
            return

        self.end_meta()
        if opcode in _MERGED_CONTROLS:
            control, target = record.qubits
            if target in self.control_targets or (
                self.control_group and self.control_group[0].qubits[0] != control
            ):
                self.flush_controls()
            self.control_group.append(record)
            self.control_targets.add(target)
            return

        self.flush_controls()
        self.reserve_wires(
            record.qubits, None if opcode in _INLINE_GUARD else condition
        )
        if target_bit is not None:
            self.written_bits.add(target_bit)
            self.read_bits.add(target_bit)
        if condition is not None:
            self.read_bits.add(condition)
        self.draw_quantum(record)

    def draw_quantum(self, record: KmxRecord) -> None:
        opcode = record.opcode
        row_index = _wire_row(record.qubits[-1])
        row = self.rows[row_index]
        if opcode == "R":
            if self.fillers[row_index] == _UNCERTAIN:
                row[:] = row.replace(_UNCERTAIN, b" ")
            self.fillers[row_index] = b"-"
            row[-1:] = b"|0>"
        elif opcode == "HMR":
            self.fillers[row_index] = b" "
            row[-1:] = f"HMR=b{record.result_bit}".encode()
        elif opcode in _INLINE_GUARD:
            for control in record.qubits[:-1]:
                self.rows[_wire_row(control)][-1:] = b"@"
            if opcode.endswith("X"):
                self.fillers[row_index] = b"-"
            row[-1:] = opcode[-1].encode()
            if record.condition_bit is not None:
                row.extend(f"**b{record.condition_bit}".encode())
        elif opcode == "Z_POW":
            assert record.angle is not None
            row[-1:] = f"Z^{record.angle.to_decimal_half_turns()}".encode()
        elif opcode in {"SWAP", "CSWAP"}:
            if opcode == "CSWAP":
                self.rows[_wire_row(record.qubits[0])][-1:] = b"@"
            for target in record.qubits[-2:]:
                index = _wire_row(target)
                self.fillers[index] = b"-"
                self.rows[index][-1:] = b"SWAP"
        elif opcode != "DEBUG_PRINT_Q":
            raise ValueError(f"Unsupported diagram operation {opcode!r}")

    def render(self, use_unicode: bool) -> str:
        self.flush_controls()
        self.flush_column()
        lines = [
            row.replace(_UNCERTAIN, b"-").rstrip(b" ").decode() for row in self.rows
        ]
        while lines and not lines[-1]:
            lines.pop()
        first = next((index for index, line in enumerate(lines) if line), len(lines))
        lines = lines[first:]
        if not use_unicode:
            return "\n".join(lines)

        rendered: list[str] = []
        was_wire = False
        for line in lines:
            glyphs: list[str] = []
            for char in line:
                if char == "|":
                    glyphs.append("┼" if was_wire else "│")
                else:
                    glyphs.append(_GLYPHS.get(char, char))
                was_wire = char == "-"
            rendered.append("".join(glyphs))
        return "\n".join(rendered)


def render_reversible_phase_text(kmx: str, *, use_unicode: bool = False) -> str:
    """Render the physical circuit with the source ASCII or Unicode glyphs."""
    return render_circuit_text(read_kmx_circuit(kmx), use_unicode=use_unicode)


def render_circuit_text(circuit: KmxCircuit, *, use_unicode: bool = False) -> str:
    """Render an already parsed physical circuit."""
    diagram = _TextDiagram(circuit.qubits)
    diagram.add_registers(circuit)
    for record in circuit.records:
        diagram.add_operation(record)
    return diagram.render(use_unicode)
