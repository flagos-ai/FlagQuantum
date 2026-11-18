"""Verify the explicit square value-contract realization at the SDK boundary."""

from hashlib import sha1
from math import sqrt
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_cost_phasing_file import source_module as source_module

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 3])
def test_square_right_value_and_left_adjoint_against_original_tensors(
    source_module: Any, width: int
) -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic import multiplication

    raw = Path(multiplication.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "efed2d6652c7dcc943c82e0c5d35ae5b3b627c09"
    )
    source = multiplication.Square(width)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    tensor = torch.as_tensor(source.tensor_contract().copy())
    emitter = import_register_emitter(source, local_ports=("result",))
    oracle = BlackBoxPrepare(
        (("a", width),), (), emitter, local_junk_layout=(("result", 2 * width),)
    )
    work, slots = emitter.requirements()
    total = 3 * width + work
    circuits = []
    for inverse in (False, True):
        ir = (
            Circuit(total)
            .append(
                oracle.make_on(
                    selection=tuple(range(width)),
                    junk=(),
                    workspace=tuple(range(width, total)),
                    classical_bits=tuple(range(slots)),
                    adjoint=inverse,
                )
            )
            .to_ir()
        )
        circuits.append(ir)
    forward, inverse = circuits
    count = certify_erasure_histories(
        forward.instructions,
        n_wires=total,
        zero_wires=tuple(range(width, total)),
        check_clean_output=False,
    )
    back_count = certify_erasure_histories(
        inverse.instructions, n_wires=total, zero_wires=tuple(range(3 * width, total))
    )
    for a in range(1 << width):
        initial = torch.zeros(1 << total, dtype=torch.complex128)
        initial[a << (2 * width + work)] = 1
        actual = forced_branch(initial, forward.instructions, (0,) * count) * sqrt(
            2**count
        )
        expected = torch.zeros_like(actual)
        expected[:: 1 << work] = tensor[:, a]
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        restored = forced_branch(
            actual, inverse.instructions, (0,) * back_count
        ) * sqrt(2**back_count)
        torch.testing.assert_close(restored, initial, atol=1e-11, rtol=0)
