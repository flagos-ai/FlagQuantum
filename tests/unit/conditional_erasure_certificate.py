"""Prove guarded X-erasure channels without enumerating measurement histories.

Quantum basis functions and exact sign phases share the existing reduced Boolean
decision graph. Each active X-basis effect has Kraus factor 1/sqrt(2); inactive
effects have only identity outcome zero. Classical guards must depend only on
earlier outcomes, so the product of these factors is common to every logical
input. This preserves absolute complex amplitudes, not just probabilities.
"""

from collections.abc import Mapping
from functools import cache

from flagquantum.compiler.registers.primitives.global_phase import ccz_ir
from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.classical_dataflow import parse_classical_xor
from flagquantum.core.ir import Instruction
from tests.unit.measurement_certificate import ONE, ZERO, _BooleanSupport


class _ConditionalSupport(_BooleanSupport):
    def node(self, variable: int, low: int, high: int) -> int:
        assert len(self.nodes) < 200_000, "Conditional proof node budget exceeded"
        return super().node(variable, low, high)

    def negate(self, value: int) -> int:
        return self.xor(value, ONE)

    def either(self, left: int, right: int) -> int:
        return self.xor(self.xor(left, right), self.multiply(left, right))

    def choose(self, condition: int, yes: int, no: int) -> int:
        return self.xor(
            self.multiply(condition, yes), self.multiply(self.negate(condition), no)
        )

    def both(self, *values: int) -> int:
        result = ONE
        for value in values:
            result = self.multiply(result, value)
        return result

    def dependencies(self, node: int) -> set[int]:
        pending, seen, result = [node], set[int](), set[int]()
        while pending:
            current = pending.pop()
            if current <= ONE or current in seen:
                continue
            seen.add(current)
            variable, low, high = self.nodes[current]
            result.add(variable)
            pending.extend((low, high))
        return result

    def count_assignments(self, node: int, variables: int) -> int:
        @cache
        def count(current: int, level: int) -> int:
            if current == ZERO:
                return 0
            if current == ONE:
                return 1 << (variables - level)
            variable, low, high = self.nodes[current]
            assert level <= variable < variables
            return (1 << (variable - level)) * (
                count(low, variable + 1) + count(high, variable + 1)
            )

        return count(node, 0)


def certify_conditional_erasure_map(
    gates: tuple[Instruction, ...],
    *,
    n_wires: int,
    logical_wires: tuple[int, ...],
    mapping: Mapping[tuple[int, ...], tuple[int, ...]],
) -> dict[str, int]:
    """Prove every allowed raw branch is the given isometry times its scalar.

    Mapping keys/values use logical_wires order. Every other quantum input is
    zero and must be zero at return. All listed columns, their arbitrary complex
    superpositions and external entanglement are included. The accepted atoms
    are Boolean reversible gates, Z/CZ, the literal owned CCZ expansion,
    ClassicalXor and guarded H/measure/X-clear blocks. Expand reset outcomes
    before calling. Unsupported atoms or failed obligations raise AssertionError.

    Inactive result slots must already be zero, avoiding any ambiguity between
    retained and default-zero guarded outputs. Effect activity depends only on
    earlier outcomes. Local Kraus completeness then proves the sum of branch
    probabilities is one; no constant scalar across different histories is
    assumed. Return counts describe actual effect sites and allowed histories.
    """
    assert n_wires > 0 and mapping
    assert len(set(logical_wires)) == len(logical_wires)
    assert all(0 <= wire < n_wires for wire in logical_wires)
    width = len(logical_wires)
    assert len(set(mapping.values())) == len(mapping), "Expected map must be injective"
    assert all(
        len(before) == len(after) == width and set((*before, *after)) <= {0, 1}
        for before, after in mapping.items()
    )
    arities = {
        "x": 1,
        "cx": 2,
        "ccx": 3,
        "z": 1,
        "cz": 2,
        "swap": 2,
        "h": 1,
        "measure": 1,
    }
    for gate in gates:
        assert gate.matrix is None, "Conditional proof refuses matrix overrides"
        if gate.name == "classical_xor":
            parse_classical_xor(gate)
            continue
        assert (
            gate.name in arities and not gate.params
        ), "Unsupported conditional proof atom"
        assert len(gate.wires) == arities[gate.name]
        assert len(set(gate.wires)) == len(gate.wires)
        assert all(0 <= wire < n_wires for wire in gate.wires)

    support = _ConditionalSupport()
    qubits = [ZERO] * n_wires
    for index, wire in enumerate(logical_wires):
        qubits[wire] = support.variable(index)
    initial = tuple(qubits)
    classical: dict[int, int] = {}
    phase, allowed = ZERO, ONE
    events = conditional_events = 0

    def guard(gate: Instruction) -> int:
        clauses = instruction_condition_clauses(gate)
        if not clauses:
            return ONE
        result = ZERO
        for clause in clauses:
            conjunction = ONE
            for bit, value in clause:
                assert bit in classical, "Guard reads an undefined classical slot"
                old = classical[bit]
                conjunction = support.multiply(
                    conjunction, old if value else support.negate(old)
                )
            result = support.either(result, conjunction)
        return result

    def assign(gate: Instruction) -> None:
        model = parse_classical_xor(gate)
        active = guard(gate)
        value = ONE if model.constant else ZERO
        for bit in model.sources:
            assert bit in classical, "XOR reads an undefined classical slot"
            value = support.xor(value, classical[bit])
        if model.target not in classical:
            assert active == ONE, "First classical write must be unconditional"
        classical[model.target] = support.choose(
            active, value, classical.get(model.target, ZERO)
        )

    position = 0
    while position < len(gates):
        gate = gates[position]
        active, name, wires = guard(gate), gate.name, gate.wires
        if name == "classical_xor":
            assign(gate)
            position += 1
            continue
        if name == "h":
            if (
                position + 2 < len(gates)
                and gates[position + 1].name == "ccx"
                and gates[position + 2].name == "h"
            ):
                left, right, target = gates[position + 1].wires
                clauses = instruction_condition_clauses(gates[position + 1])
                assert len(clauses) <= 1
                assert gates[position : position + 3] == ccz_ir(
                    (left, right, target), conditions=clauses[0] if clauses else ()
                ), "Unrecognized CCZ basis change"
                phase = support.xor(
                    phase,
                    support.both(
                        guard(gates[position + 1]),
                        qubits[left],
                        qubits[right],
                        qubits[target],
                    ),
                )
                position += 3
                continue
            position += 1
            while position < len(gates) and gates[position].name == "classical_xor":
                assign(gates[position])
                position += 1
            assert position + 1 < len(gates), "Incomplete X-erasure block"
            measurement, clear = gates[position : position + 2]
            assert measurement.name == "measure" and measurement.wires == wires
            assert guard(measurement) == active, "X-erasure guards changed after H"
            assert support.dependencies(active) <= set(
                range(width, width + events)
            ), "Effect activity depends on logical input or a future outcome"
            slot = int(measurement.metadata["classical_bit"])
            if active != ONE:
                assert slot in classical, "Inactive result has no initialized slot"
            old_result = classical.get(slot, ZERO)
            assert (
                support.both(support.negate(active), old_result) == ZERO
            ), "Inactive result slot is not zero"
            outcome = support.variable(width + events)
            allowed = support.multiply(
                allowed, support.negate(support.both(support.negate(active), outcome))
            )
            phase = support.xor(phase, support.both(active, outcome, qubits[wires[0]]))
            classical[slot] = support.choose(active, outcome, old_result)
            assert clear.name == "x" and clear.wires == wires
            assert guard(clear) == support.multiply(
                active, outcome
            ), "Incorrect measurement clearing feedback"
            qubits[wires[0]] = support.multiply(
                qubits[wires[0]], support.negate(active)
            )
            events += 1
            conditional_events += active != ONE
            position += 2
            continue
        if name in {"x", "cx", "ccx"}:
            qubits[wires[-1]] = support.xor(
                qubits[wires[-1]],
                support.both(active, *(qubits[wire] for wire in wires[:-1])),
            )
        elif name in {"z", "cz"}:
            phase = support.xor(
                phase, support.both(active, *(qubits[wire] for wire in wires))
            )
        elif name == "swap":
            left, right = wires
            old_left, old_right = qubits[left], qubits[right]
            qubits[left] = support.choose(active, old_right, old_left)
            qubits[right] = support.choose(active, old_left, old_right)
        else:
            raise AssertionError(f"Unpaired conditional proof atom: {name}")
        position += 1

    valid = ZERO
    expected = [ZERO] * n_wires
    for before, after in mapping.items():
        column = support.both(
            *(
                initial[wire] if bit else support.negate(initial[wire])
                for wire, bit in zip(logical_wires, before, strict=True)
            )
        )
        valid = support.either(valid, column)
        for wire, bit in zip(logical_wires, after, strict=True):
            if bit:
                expected[wire] = support.either(expected[wire], column)
    domain = support.multiply(valid, allowed)
    assert domain != ZERO, "Empty conditional proof domain"
    assert support.multiply(domain, phase) == ZERO, "Nonzero complex branch phase"
    for wire in range(n_wires):
        assert (
            support.both(domain, support.xor(qubits[wire], expected[wire])) == ZERO
        ), f"Output or clean-work mismatch on wire {wire}"
    assert not support.dependencies(allowed).intersection(range(width))
    histories = support.count_assignments(allowed, width + events) >> width
    return {
        "effects": events,
        "conditional_effects": conditional_events,
        "allowed_histories": histories,
        "logical_columns": len(mapping),
        "boolean_nodes": len(support.nodes),
    }
