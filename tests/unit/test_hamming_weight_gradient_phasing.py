"""Original Hamming-count QVR composition and coherent gradient restoration."""

from cmath import exp
from collections import Counter
from itertools import product
from math import pi, sqrt, tau
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.hamming_weight_phasing import (
    hamming_weight_gradient_phasing_ir,
)
from flagquantum.compiler.registers.phase_addition import phase_add_workspace_size
from flagquantum.compiler.registers.scaled_phase_addition import (
    scaled_phase_add_ir,
    scaled_phase_add_terms,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_hamming_weight import layout
from tests.unit.test_hamming_weight_phasing import source_count_leaf
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "n,exponent,epsilon",
    [
        (1, 0.25, 0.25),
        (2, 0.125, 0.5),
        (3, -0.375, 0.25),
        (4, 0.125, 0.25),
        (8, 0.625, 0.25),
    ],
)
def test_gradient_hamming_source_binding_phase_and_cleanup(
    n: int, exponent: float, epsilon: float
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import Adjoint, BloqError, QFxp
    from qualtran.bloqs.rotations.hamming_weight_phasing import (
        HammingWeightPhasingViaPhaseGradient,
    )
    from qualtran.bloqs.rotations.phase_gradient import AddScaledValIntoPhaseReg
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrPhaseGradient
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    x, junk, count, count_bits = layout(n)
    source = HammingWeightPhasingViaPhaseGradient(n, exponent, epsilon)
    qvr = source.phase_oracle
    m = int(qvr.b_grad)
    assert qvr.cost_dtype == QFxp(len(count), 0) and qvr.gamma == exponent / 2
    graph = qvr.decompose_bloq()
    ((stage, _, _),) = tuple(graph.iter_bloqsoqs())
    scaled = stage.bloq
    assert isinstance(scaled, AddScaledValIntoPhaseReg)
    assert scaled.gamma_dtype == source.gamma_dtype and scaled.phase_bitsize == m
    assert scaled.x_dtype == qvr.cost_dtype and scaled.gamma == qvr.gamma
    terms = scaled_phase_add_terms(
        len(count),
        0,
        m,
        qvr.gamma,
        scaled.gamma_dtype.bitsize,
        scaled.gamma_dtype.num_frac,
    )
    requirements = tuple(
        phase_add_workspace_size(len(count[part]), m, right_shift=shift)
        for part, shift in terms
    )
    required = max(requirements, default=0)
    gradient = tuple(range(n + len(junk) + len(count), n + len(junk) + len(count) + m))
    work = tuple(range(gradient[-1] + 1, gradient[-1] + 1 + required))
    addition_bits = tuple(range(40, 40 + required))
    width = n + len(junk) + len(count) + m + required
    ports = {"x": x, "phase_grad": gradient}
    parts = {"x": x, "junk": junk, "out": count}
    model = get_cost_value(source, QECGatesCost())
    assert model.and_bloq == model.measurement == len(junk)
    assert model.toffoli == get_cost_value(qvr, QECGatesCost()).toffoli
    if n == 2:
        assert get_cost_value(source, QubitCount()) == n + len(junk) + len(count) + m
    else:
        with pytest.raises(RuntimeError, match="QAny|register dtypes"):
            get_cost_value(source, QubitCount())

    increments = tuple(
        int(graph.on_classical_vals(out=w, phase_grad=0)["phase_grad"])
        for w in range(n + 1)
    )
    assert (
        max(
            abs(exp(1j * pi * exponent * w) - exp(1j * tau * delta / 2**m))
            for w, delta in enumerate(increments)
        )
        <= epsilon
    )
    events = sum(requirements) + len(junk)
    tail = len(count) + len(junk) + m + required
    for adjoint in (False, True):
        original = source.adjoint() if adjoint else source
        inner_graph = original.decompose_bloq()
        raw = scaled_phase_add_ir(
            count,
            gradient,
            work,
            addition_bits,
            input_fraction_bits=0,
            coefficient=scaled.gamma,
            coefficient_width=scaled.gamma_dtype.bitsize,
            coefficient_fraction_bits=scaled.gamma_dtype.num_frac,
            adjoint=adjoint,
        )

        def leaf(
            bloq: Any,
            inputs: SourcePorts,
            desired: SourcePorts,
            *,
            inverse: bool = adjoint,
            inner=raw,
        ) -> SourceResult | None:
            count_leaf = source_count_leaf(bloq, inputs, parts, count_bits)
            if count_leaf is not None:
                return count_leaf
            core = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
            if not isinstance(core, QvrPhaseGradient):
                return None
            expected = qvr.adjoint() if inverse else qvr
            assert bloq == expected
            assert inputs == {"out": count, "phase_grad": gradient}
            return inner, inputs

        expected, outputs = walk_source_graph(original, ports, ports, leaf)
        gates = hamming_weight_gradient_phasing_ir(
            x,
            junk,
            count,
            gradient,
            work,
            count_bits,
            addition_bits,
            exponent=exponent,
            epsilon=epsilon,
            adjoint=adjoint,
        )
        assert outputs == ports
        assert_source(gates, expected)
        counts = Counter(g.name for g in gates)
        assert counts["measure"] == counts["reset"] == events
        assert counts["t"] + counts["tdg"] == 4 * events

        if n == 1:
            with pytest.raises(KeyError):
                inner_graph.on_classical_vals(x=1, phase_grad=0)
        elif adjoint and n > 2:
            with pytest.raises(BloqError, match="register dtypes"):
                inner_graph.on_classical_vals(x=(1 << n) - 1, phase_grad=0)
        else:
            sign = -1 if adjoint else 1
            for value in range(1 << n):
                assert inner_graph.on_classical_vals(x=value, phase_grad=0) == {
                    "x": value,
                    "phase_grad": (sign * increments[value.bit_count()]) % (1 << m),
                }

        sign = -1 if adjoint else 1
        states, mapped = [], []
        start, end = {}, {}
        norm = sqrt(2 ** (n + m))
        for value, y in product(range(1 << n), range(1 << m)):
            delta = sign * increments[value.bit_count()]
            i = (value << tail) | (y << required)
            j = (value << tail) | (((y + delta) % (1 << m)) << required)
            states.append(tuple(i >> k & 1 for k in reversed(range(width))))
            mapped.append(tuple(j >> k & 1 for k in reversed(range(width))))
            start[i] = exp(1j * (value - tau * y / 2**m)) / norm
            end[i] = start[i] * exp(1j * tau * delta / 2**m)
        assert certified_and_channel_map(gates, tuple(states)) == (
            tuple(mapped),
            events,
        )
        if events <= 4:
            initial = torch.zeros(1 << width, dtype=torch.complex128)
            result = torch.zeros_like(initial)
            initial[list(start)] = torch.tensor(
                list(start.values()), dtype=torch.complex128
            )
            result[list(end)] = torch.tensor(list(end.values()), dtype=torch.complex128)
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(initial, gates, history),
                    result * 2 ** (-events / 2),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 41):
                actual = execute_sparse_register(
                    CircuitIR(width, gates), start, seed=seed
                )
                assert all(
                    abs(actual.get(k, 0) - end.get(k, 0)) < 2e-12
                    for k in actual.keys() | end.keys()
                )


def test_hamming_gradient_rejects_shared_quantum_wires() -> None:
    with pytest.raises(ValueError, match="distinct"):
        hamming_weight_gradient_phasing_ir(
            (0,),
            (),
            (1,),
            (0, 2, 3),
            (),
            (),
            (),
            exponent=0.25,
            epsilon=0.25,
        )


def test_append_hamming_gradient_phasing_example() -> None:
    from examples.algorithms.register_hamming_weight_gradient_phasing import main

    main()
