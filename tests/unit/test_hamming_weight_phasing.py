"""Original compute/phase/erase graph and complete coherent branch checks."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.hamming_weight_phasing import (
    hamming_weight_phasing_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_hamming_weight import layout
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def source_count_leaf(
    bloq: Any, inputs: SourcePorts, parts: SourcePorts, bits: tuple[int, ...]
) -> SourceResult | None:
    """Read a pinned Hamming count or its measured adjoint."""
    from qualtran import Adjoint
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    x, junk = parts["x"], parts["junk"]
    n = len(x)
    if isinstance(bloq, HammingWeightCompute):
        assert bloq.bitsize == n and inputs == {"x": x}
        return (
            reference_arithmetic_gates(bloq, parts, (), {}, cirq_style=True),
            parts,
        )
    if isinstance(bloq, Adjoint) and isinstance(bloq.subbloq, HammingWeightCompute):
        assert inputs == parts and bloq.subbloq.bitsize == n
        output = {"x": x}
        if n == 1:
            with pytest.raises(KeyError):
                bloq.decompose_bloq()
            # Empty source junk cannot be bound by its adjoint adapter.
            # The one-bit forward source is a single self-inverse CX.
            return (
                reference_arithmetic_gates(
                    bloq.subbloq,
                    parts,
                    (),
                    {},
                    cirq_style=True,
                ),
                output,
            )
        return (
            reference_arithmetic_gates(
                bloq.decompose_bloq(),
                inputs,
                (),
                dict(zip(junk, bits, strict=True)),
                cirq_style=True,
                expected_outputs=output,
                reuse_allocations=False,
            ),
            output,
        )
    return None


@pytest.mark.parametrize(
    "n,exponent",
    [(1, 0.25), (2, 1.0), (3, -0.375), (4, 0.125), (7, -0.375), (8, 0.625)],
)
def test_hamming_phasing_source_graph_and_complete_complex_state(
    n: int, exponent: float
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.rotations.hamming_weight_phasing import HammingWeightPhasing
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    x, junk, count, bits = layout(n)
    width = n + len(junk) + len(count)
    src = HammingWeightPhasing(n, exponent, eps=3e-8)
    parts = {"x": x, "junk": junk, "out": count}
    for adjoint in (False, True):
        source = src.adjoint() if adjoint else src

        def leaf(
            bloq: Any, inputs: SourcePorts, desired: SourcePorts
        ) -> SourceResult | None:
            count_leaf = source_count_leaf(bloq, inputs, parts, bits)
            if count_leaf is not None:
                return count_leaf
            if isinstance(bloq, ZPowGate):
                assert bloq.eps == pytest.approx(src.eps / len(count))
                return (
                    Instruction(
                        "phase", inputs["q"], params={"theta": pi * bloq.exponent}
                    ),
                ), inputs
            return None

        gates = hamming_weight_phasing_ir(
            x, junk, count, bits, exponent=exponent, adjoint=adjoint
        )
        expected, outputs = walk_source_graph(source, {"x": x}, {"x": x}, leaf)
        assert outputs == {"x": x}
        assert_source(gates, expected)
        assert tuple(g for g in gates if g.name == "phase") == tuple(
            g for g in expected if g.name == "phase"
        )
        counts = Counter(g.name for g in gates)
        model = get_cost_value(src, QECGatesCost())
        assert counts["phase"] == len(count)
        assert counts["measure"] == counts["reset"] == model.measurement == len(junk)
        assert counts["t"] + counts["tdg"] == 4 * model.and_bloq == 4 * len(junk)
        assert counts["cx"] == 16 * len(junk) + 2 * n.bit_count()
        if n == 1:
            with pytest.raises(RuntimeError, match="QAny"):
                get_cost_value(src, QubitCount())
        else:
            assert get_cost_value(src, QubitCount()) == width

        rows = tuple(
            tuple((a << (width - n)) >> i & 1 for i in reversed(range(width)))
            for a in range(1 << n)
        )
        assert certified_and_channel_map(
            tuple(g for g in gates if g.name != "phase"), rows
        ) == (rows, len(junk))
        indices = [2 * (a << (width - n)) + s for a in range(1 << n) for s in (0, 1)]
        weights = torch.arange(1, len(indices) + 1, dtype=torch.float64)
        amps = torch.complex(weights, weights * (weights % 5 - 2))
        amps /= amps.norm()
        sign = -1 if adjoint else 1
        phases = torch.tensor(
            [
                exp(1j * pi * sign * exponent * a.bit_count())
                for a in range(1 << n)
                for _ in (0, 1)
            ],
            dtype=torch.complex128,
        )
        if n <= 4:
            initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
            result = torch.zeros_like(initial)
            initial[indices] = amps
            result[indices] = amps * phases
            for history in product((0, 1), repeat=len(junk)):
                torch.testing.assert_close(
                    forced_branch(initial, gates, history),
                    result * 2 ** (-len(junk) / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            before = dict(zip(indices, amps.tolist(), strict=True))
            after = dict(zip(indices, (amps * phases).tolist(), strict=True))
            for seed in (3, 41):
                actual = _execute_sparse(CircuitIR(width + 1, gates), before, seed=seed)
                assert all(
                    abs(actual.get(i, 0) - after.get(i, 0)) < 2e-12
                    for i in actual.keys() | after.keys()
                )


def test_hamming_phasing_validation() -> None:
    with pytest.raises(ValueError, match="adjoint"):
        hamming_weight_phasing_ir((0,), (), (1,), (), exponent=0.25, adjoint=1)
    with pytest.raises(ValueError, match="finite"):
        hamming_weight_phasing_ir((0,), (), (1,), (), exponent=float("nan"))
    with pytest.raises(ValueError, match="output wires"):
        hamming_weight_phasing_ir((0,), (), (), (), exponent=0.25)


def test_append_hamming_phasing_example() -> None:
    from examples.algorithms.register_hamming_weight_phasing import main

    main()
