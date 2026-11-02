"""AO-02 paired pre-target arithmetic costs, with complete-domain XOR checks.

This supplements the immutable ninety-case corpus; it does not replace it.
No preparation, target optimization, numerical execution or speed claim is made.
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import Schedule, schedule_arithmetic

from .arithmetic_baseline import Costs, costs, revision
from .expressions import EDGES, EXPRESSIONS, Case, Expression
from .oracle import evaluate
from .verification import classical_apply, encode


def five_times(x: int) -> int:
    return 5 * x


def cases() -> tuple[Case, ...]:
    by_name = {expression.name: expression for expression in (*EXPRESSIONS, *EDGES)}
    return (
        Case(by_name["square"], (2,)),
        Case(by_name["square"], (3,)),
        Case(by_name["constant_product"], (4,)),
        Case(by_name["constant_product"], (5,)),
        Case(Expression("five_times", "5*x", ("x",), five_times), (4,)),
        Case(by_name["add"], (2, 2)),
        Case(by_name["add"], (1, 3)),
        Case(by_name["multiply_add"], (1, 1, 2)),
    )


@dataclass(frozen=True)
class Pair:
    case: str
    expression: str
    strategy: str
    conservative: Costs
    selected: Costs
    rules: tuple[str, ...]
    exhaustive_input_target_checks: int


def measure(case: Case, strategy: Schedule) -> Pair:
    graph = capture(
        case.expression.function,
        dict(zip(case.expression.arguments, case.widths, strict=True)),
    )
    block = graph.body.blocks[0]
    position = 0
    registers = []
    for width in case.widths:
        registers.append(tuple(range(position, position + width)))
        position += width
    output_width = block.operations[-1].operands[0].type.parameters[0]
    registers.append(tuple(range(position, position + output_width)))
    position += output_width
    candidate = schedule_arithmetic(
        (graph,),
        {arg.id: registers[i] for i, arg in enumerate(block.arguments)},
        registers[-1],
        entry=graph.name,
        n_qubits=256,
        workspace=tuple(range(position, 256)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    if candidate.conservative is None:
        raise AssertionError("paired evidence requires a real AO-01 candidate")
    before = candidate.conservative
    old = costs(
        replace(before.circuit, n_wires=position + before.peak_workspace, shape=()),
        before.peak_workspace,
    )
    new = costs(
        replace(
            candidate.circuit, n_wires=position + candidate.peak_workspace, shape=()
        ),
        candidate.peak_workspace,
    )
    a = (new.workspace_qubits, new.operations, new.depth)
    b = (old.workspace_qubits, old.operations, old.depth)
    if not all(x <= y for x, y in zip(a, b, strict=True)) or a == b:
        raise AssertionError("paired case has no strict non-worse resource gain")
    checks = 0
    for values in product(*(range(1 << width) for width in case.widths)):
        answer = (
            sum((values[0],) * 5)
            if case.expression.name == "five_times"
            else evaluate(case.expression.name, values)
        )
        for target in range(1 << output_width):
            initial = encode((*values, target), tuple(registers))
            expected = encode((*values, target ^ answer), tuple(registers))
            if (
                classical_apply(candidate.circuit, initial) != expected
                or classical_apply(before.circuit, initial) != expected
            ):
                raise AssertionError("paired XOR/cleanup does not match mathematics")
            checks += 1
    return Pair(
        case.identifier,
        case.expression.text,
        strategy,
        old,
        new,
        tuple(sorted({rule for _, rule in candidate.implementation_rules})),
        checks,
    )


@dataclass(frozen=True)
class Receipt:
    source_revision: str
    rows: tuple[Pair, ...]
    schema: str = "flagquantum.synthesis.arithmetic_implementations.v1"
    scope: str = (
        "exact pre-target arithmetic including cleanup; no preparation or state execution"
    )
    baseline: str = (
        "actual retained AO-01 candidate, not a regenerated historical artifact"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in cases():
        for strategy in ("retain", "recompute"):
            row = measure(case, strategy)
            rows.append(row)
            print(
                row.case,
                row.strategy,
                row.selected.workspace_qubits,
                row.selected.operations,
                flush=True,
            )
    args.output.write_text(
        json.dumps(asdict(Receipt(revision(), tuple(rows))), indent=2, sort_keys=True)
        + "\n"
    )


if __name__ == "__main__":
    main()
