"""Verify the native bounded increment against the locked ModAddK permutation."""

from dataclasses import asdict
from math import sqrt

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.bounded_increment import (
    bounded_increment_ir,
    bounded_increment_requirements,
    source_bounded_increment_cost,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "n,modulus,values",
    [
        (1, 1, (1,)),
        (1, 2, ()),
        (2, 4, (0,)),
        (2, 4, (0, 0)),
        (2, 3, ()),
        (2, 3, (0, 0)),
        (3, 5, (1,)),
        (3, 7, (1, 0, 1)),
        (4, 9, (0, 1)),
    ],
)
def test_bounded_increment_full_domain_and_every_erasure_branch(n, modulus, values):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic import ModAddK

    nc = len(values)
    source = ModAddK(n, modulus, add_val=1, cvs=values)
    work, slots = bounded_increment_requirements(n, modulus, nc)
    controls = tuple(range(nc))
    target = tuple(range(nc, nc + n))
    workspace = tuple(range(nc + n, nc + n + work))
    words = range(1 << (nc + n))
    inputs = tuple(
        tuple(int(bit) for bit in f"{word:0{nc + n}b}") + (0,) * work for word in words
    )
    for direction, reference in ((False, source), (True, source**-1)):
        gates = bounded_increment_ir(
            target,
            modulus,
            workspace,
            tuple(range(slots)),
            controls=controls,
            control_values=values,
            adjoint=direction,
        )
        result, events = certified_and_channel_map(gates, inputs)
        output_words = []
        for word in words:
            ctrl, x = word >> n, word & ((1 << n) - 1)
            out = reference.on_classical_vals(x=x, ctrl=ctrl if nc else None)
            output_words.append((ctrl << n) | out["x"])
        expected = tuple(
            tuple(int(bit) for bit in f"{word:0{nc + n}b}") + (0,) * work
            for word in output_words
        )
        assert result == expected
        assert events == sum(gate.name == "measure" for gate in gates)
        if (n, modulus, values) == (2, 3, (0, 0)):
            amplitudes = tuple(1j**i / sqrt(len(words)) for i in words)
            initial = {
                word << work: amp for word, amp in zip(words, amplitudes, strict=True)
            }
            final = {
                word << work: amp
                for word, amp in zip(output_words, amplitudes, strict=True)
            }
            for pattern in ((0,), (1,), (0, 1), (1, 0)):
                assert _branch(
                    gates,
                    n + nc + work,
                    (pattern * events)[:events],
                    initial,
                    normalize_measurements=True,
                ) == pytest.approx(final, abs=3e-12)


def test_bounded_increment_locked_cost_and_opaque_source():
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.mod_arithmetic import ModAddK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ModAddK(3, 7, add_val=1, cvs=(0, 0))
    model = estimate_component_resources(source_bounded_increment_cost(3, 2))
    assert {k: v for k, v in asdict(model.gates).items() if v} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
