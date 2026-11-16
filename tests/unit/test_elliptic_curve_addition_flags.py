"""Verify the pinned ECAdd flag stage and its coherent clean-work binding."""

from collections import Counter
from importlib.metadata import version
from itertools import product
from typing import Any

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_addition_flags import (
    EllipticCurveAdditionFlags,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_modular_negation import reference_binding
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("n,p", [(2, 1), (3, 5), (5, 17)])
def test_locked_stage_wiring_and_resources(n: int, p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic import Equals
    from qualtran.bloqs.basic_gates import ZeroState
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepOne
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.bloqs.mod_arithmetic import ModNeg
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert version("qualtran") == "0.7.0"
    source = _ECAddStepOne(n, p)
    native = EllipticCurveAdditionFlags(n, p)
    assert [(reg.name, reg.bitsize, reg.side.name) for reg in source.signature] == [
        (reg.name, reg.dtype.num_qubits, reg.side.name)
        for reg in native.source_signature
    ]
    q, c = native.workspace_requirements()
    words = dict(
        zip(
            ("a", "b", "x", "y"),
            (tuple(range(i * n, (i + 1) * n)) for i in range(4)),
            strict=True,
        )
    )
    flags = tuple(range(4 * n, 4 * n + 5))
    work = tuple(range(4 * n + 5, 4 * n + 5 + q))
    slots = tuple(range(71, 71 + c))
    outputs = {
        **words,
        **{
            name: (wire,)
            for name, wire in zip(("f1", "f2", "f3", "f4", "ctrl"), flags, strict=True)
        },
    }

    composite = source.decompose_bloq()
    steps = sorted(composite.iter_bloqsoqs(), key=lambda step: step[0].i)
    initial_flags: dict[Any, tuple[int, ...]] = {
        right[0]: (flags[node.i],)
        for node, _, right in steps
        if isinstance(node.bloq, ZeroState)
    }
    blocks: dict[int, tuple[Instruction, ...]] = {}
    modneg_steps = iter((6, 8))

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if isinstance(bloq, ZeroState):
            assert not inputs and len(desired["q"]) == 1
            return (), desired
        if isinstance(bloq, ModNeg):
            step = next(modneg_steps)
            alloc, measured = reference_binding(n, work, slots)
        elif isinstance(bloq, (Equals, MultiControlX)):
            step = dict(zip(flags, (5, 7, 11, 16, 19), strict=True))[
                inputs["target"][0]
            ]
            length = n if isinstance(bloq, Equals) else len(inputs["controls"])
            alloc = work[: length - 1]
            measured = dict(zip(alloc, slots[: length - 1], strict=True))
        else:
            return None
        assert steps[step][0].bloq == bloq
        blocks[step] = reference_arithmetic_gates(
            bloq,
            inputs,
            alloc,
            measured,
            reuse_allocations=not isinstance(bloq, ModNeg),
            preserve_components=isinstance(bloq, ModNeg),
        )
        return (), inputs

    _, actual_outputs = walk_source_graph(
        composite, words, outputs, leaf, bindings=initial_flags
    )
    expected = tuple(gate for step in sorted(blocks) for gate in blocks[step])
    assert actual_outputs == outputs
    actual = native.make_on(
        **words, flags=flags, workspace=work, classical_bits=slots
    ).instructions
    assert_source(actual, expected)
    gates = Counter(gate.name for gate in actual)
    model = get_cost_value(source, QECGatesCost())
    assert gates["measure"] == model.measurement == 12 * n - 8
    assert gates["t"] + gates["tdg"] == model.total_t_count() == 4 * (12 * n - 8)
    assert get_cost_value(source, QubitCount()) == 6 * n + 3
    assert len((*flags, *work)) + 4 * n == 6 * n + 5


def test_coherent_append_all_residues_and_clean_work() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add import _ECAddStepOne

    n, p = 3, 3
    native = EllipticCurveAdditionFlags(n, p)
    assert all(
        reg.side == RegisterSide.RIGHT for reg in tuple(native.source_signature)[:5]
    )
    source = _ECAddStepOne(n, p)
    q, c = native.workspace_requirements()
    width = 4 * n + 5 + q
    wires = tuple(reversed(range(width)))
    words = {
        name: wires[i * n : (i + 1) * n] for i, name in enumerate(("a", "b", "x", "y"))
    }
    flags, work = wires[4 * n : 4 * n + 5], wires[4 * n + 5 :]
    slots = tuple(5 * i + 71 for i in range(c))
    op = native.make_on(**words, flags=flags, workspace=work, classical_bits=slots)
    ir = DynamicCircuit(width).append(op).to_ir()

    def basis(values: dict[str, Any]) -> int:
        return sum(
            ((int(values[name]) >> (len(bits) - i - 1)) & 1) << (width - w - 1)
            for name, bits in {
                **words,
                **{
                    name: (wire,)
                    for name, wire in zip(
                        ("f1", "f2", "f3", "f4", "ctrl"), flags, strict=True
                    )
                },
            }.items()
            for i, w in enumerate(bits)
        )

    before, after = {}, {}
    for index, coordinates in enumerate(product(range(p), repeat=4), 1):
        values = dict(zip(words, coordinates, strict=True))
        before[basis({**values, "f1": 0, "f2": 0, "f3": 0, "f4": 0, "ctrl": 0})] = (
            complex(index, 11 - index)
        )
        after[basis(source.on_classical_vals(**values))] = complex(index, 11 - index)
    norm = sum(abs(v) ** 2 for v in before.values()) ** 0.5
    before = {k: v / norm for k, v in before.items()}
    after = {k: v / norm for k, v in after.items()}
    bit_words = tuple(
        tuple((k >> (width - i - 1)) & 1 for i in range(width)) for k in before
    )
    mapped, measurements = certified_and_channel_map(ir.instructions, bit_words)
    assert measurements == 12 * n - 8
    assert tuple(
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in mapped
    ) == tuple(after)
    for seed in (0, 1, 19):
        assert execute_sparse_register(ir, before, seed=seed) == pytest.approx(
            after, abs=1e-12
        )
