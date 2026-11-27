"""Prove zero work at specified points of a phase-free permutation circuit."""

from ..core.ir import CircuitIR

# Bounds apply to exact symbolic analysis, never to accepted quantum values.
_TERM_BUDGET = 100_000
_INPUT_BUDGET = 4096
_PRODUCT_BUDGET = 5_000_000


def require_zero_releases(
    circuit: CircuitIR,
    inputs: tuple[int, ...],
    releases: tuple[tuple[int, tuple[int, ...]], ...],
) -> None:
    """Check clean releases for every input via Boolean algebraic normal forms.

    Inputs are independent arbitrary bits; all other wires start zero. Offsets
    identify the instant before the corresponding instruction. X/CX/CCX contain
    no input-dependent phase, so this is also a complete complex-amplitude and
    external-entanglement guarantee for each released zero factor.

    Nonzero releases raise ValueError. Unsupported instructions and exhausted
    symbolic budgets raise NotImplementedError, without admitting execution.
    """

    def valid_wire(wire: int) -> bool:
        return type(wire) is int and 0 <= wire < circuit.n_wires

    if len(set(inputs)) != len(inputs) or not all(map(valid_wire, inputs)):
        raise ValueError("Clean-release inputs require unique circuit wire IDs")
    points: dict[int, set[int]] = {}
    for offset, wires in releases:
        if (
            type(offset) is not int
            or not 0 <= offset <= len(circuit.instructions)
            or not wires
            or len(set(wires)) != len(wires)
            or not all(map(valid_wire, wires))
        ):
            raise ValueError("Clean-release points require valid offsets and wires")
        points.setdefault(offset, set()).update(wires)
    for gate in circuit.instructions:
        if (
            gate.name not in {"x", "cx", "ccx"}
            or gate.params
            or gate.metadata
            or gate.matrix is not None
        ):
            raise NotImplementedError(
                "Clean-release proof requires unconditional X/CX/CCX primitives"
            )

    # Slice from the observed release points, retaining their original times.
    # A retained result may be highly nonlinear yet irrelevant to zero padding.
    needed: set[int] = set()
    selected: set[int] = set()
    for offset in reversed(range(len(circuit.instructions) + 1)):
        needed.update(points.get(offset, ()))
        if offset:
            gate = circuit.instructions[offset - 1]
            if gate.wires[-1] in needed:
                selected.add(offset - 1)
                needed.update(gate.wires[:-1])

    variables = tuple(wire for wire in inputs if wire in needed)
    if len(variables) > _INPUT_BUDGET:
        raise NotImplementedError("Clean-release symbolic input budget exhausted")
    values = {wire: frozenset((1 << i,)) for i, wire in enumerate(variables)}
    zero: frozenset[int] = frozenset()
    term_count = len(values)
    products = 0
    for offset in range(len(circuit.instructions) + 1):
        for wire in points.get(offset, ()):
            if values.get(wire, zero):
                raise ValueError(
                    f"Nonzero clean release at instruction {offset}, wire {wire}; "
                    "exact Boolean polynomial is not zero"
                )
        if offset not in selected:
            continue
        gate = circuit.instructions[offset]
        control = frozenset((0,))  # Constant one is the empty monomial.
        for wire in gate.wires[:-1]:
            right = values.get(wire, zero)
            if control == right:
                continue  # Every Boolean polynomial obeys p*p = p.
            products += len(control) * len(right)
            if products > _PRODUCT_BUDGET:
                raise NotImplementedError(
                    "Clean-release symbolic product budget exhausted"
                )
            combined: set[int] = set()
            for left_monomial in control:
                for right_monomial in right:
                    monomial = left_monomial | right_monomial
                    if monomial in combined:
                        combined.remove(monomial)
                    else:
                        combined.add(monomial)
                    if len(combined) > _TERM_BUDGET:
                        raise NotImplementedError(
                            "Clean-release symbolic term budget exhausted"
                        )
            control = frozenset(combined)
        target = gate.wires[-1]
        previous = values.get(target, zero)
        updated = previous ^ control
        term_count += len(updated) - len(previous)
        if term_count > _TERM_BUDGET:
            raise NotImplementedError("Clean-release symbolic term budget exhausted")
        values[target] = updated
