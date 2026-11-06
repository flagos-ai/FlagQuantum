"""Read original inverse graphs with explicit physical work lifetimes."""

from typing import Any

from flagquantum.compiler.registers.inverse_iteration import inverse_iteration_outputs
from flagquantum.core.ir import Instruction
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_inverse_iteration import layout as iteration_layout
from tests.unit.test_inverse_iteration import original_iteration
from tests.unit.test_register_constant_addition import reference_arithmetic_gates


def walk_inverse_source(
    source: Any,
    inputs: SourcePorts,
    desired: SourcePorts,
    n: int,
    p: int,
    adjoint: bool,
    pool: tuple[int, ...],
    bits: tuple[int, ...],
    *,
    bindings: dict[Any, tuple[int, ...]] | None = None,
    allowed_frees: tuple[tuple[int, ...], ...] = (),
) -> tuple[
    tuple[Instruction, ...], SourcePorts, tuple[int, ...], tuple[tuple[int, ...], ...]
]:
    """Execute original graph readers; allocate/free add no physical instructions."""
    freed: list[tuple[int, ...]] = []

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        nonlocal pool
        base = getattr(bloq, "subbloq", bloq)
        name = type(base).__name__
        if name == "Allocate":
            assert not inputs and set(desired) == {"reg"}
            assert len(desired["reg"]) == bloq.signature.get_right("reg").total_bits()
            return (), desired
        if name == "Free":
            assert inputs["reg"] in allowed_frees and not desired
            freed.append(inputs["reg"])
            return (), {}
        if name == "XGate":
            return (Instruction("x", inputs["q"]),), inputs
        if name == "_KaliskiIteration":
            assert base.bitsize == n and base.mod == p
            reverse = type(bloq).__name__ == "Adjoint"
            assert reverse == adjoint
            cu, cv, cr, cs, cm, cf, ct, cw, cbits = iteration_layout(n)
            if reverse:
                cr, cw = inverse_iteration_outputs(cr, cw)
            canonical = dict(
                u=cu, v=cv, r=cr, s=cs, m=(cm,), f=(cf,), is_terminal=(ct,)
            )
            qmap = dict(zip(cw, pool, strict=True))
            for key, wires in canonical.items():
                qmap.update(zip(wires, inputs[key], strict=True))
            cmap = dict(zip(cbits, bits, strict=True))
            gates = []
            for gate in original_iteration(n, p, reverse):
                metadata = dict(gate.metadata)
                if "classical_bit" in metadata:
                    metadata["classical_bit"] = cmap[metadata["classical_bit"]]
                if "conditions" in metadata:
                    metadata["conditions"] = tuple(
                        (cmap[b], val) for b, val in metadata["conditions"]
                    )
                gates.append(
                    Instruction(
                        gate.name,
                        tuple(qmap[w] for w in gate.wires),
                        gate.params,
                        gate.matrix,
                        metadata,
                    )
                )
            result, pool = inverse_iteration_outputs(inputs["r"], pool, adjoint=reverse)
            outputs = dict(inputs, r=result)
            assert all(outputs[k] == val for k, val in desired.items())
            return tuple(gates), outputs
        if name in ("XorK", "BitwiseNot", "AddK"):
            allocations = (*pool[:n], *pool[n : 2 * n - 1]) if name == "AddK" else ()
            slots = dict(zip(pool[n : 2 * n - 1], bits[: n - 1], strict=True))
            primitives = reference_arithmetic_gates(
                bloq, inputs, allocations, slots, expected_outputs=inputs
            )
            return primitives, inputs
        return None

    gates, outputs = walk_source_graph(source, inputs, desired, leaf, bindings=bindings)
    return gates, outputs, pool, tuple(freed)
