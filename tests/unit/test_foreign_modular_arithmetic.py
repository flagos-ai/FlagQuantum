"""Actual source ModAddK permutations enter the existing native implementation."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source_modular_arithmetic() -> None:
    import kickmix
    from qualtran.bloqs.mod_arithmetic import mod_addition

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    raw = Path(mod_addition.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "0a167e84fe00d5df50488c4b28925ea6114728f7"
    )


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize(
    "width,modulus,value,controls",
    (
        (2, 3, 1, (0, 0)),
        (2, 3, -1, (1,)),
        (2, 3, 2, (1, 1)),
        (2, 4, 1, ()),
        (1, 2, 1, (0,)),
        (2, 1, 3, ()),
    ),
)
def test_actual_mod_add_source_all_binary_inputs_and_history_proof(
    width: int, modulus: int, value: int, controls: tuple[int, ...], adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_addition import ModAddK

    base = ModAddK(width, modulus, value, controls)
    source = base.adjoint() if adjoint else base
    reference = base**-1 if adjoint else base
    ports = tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
    ir, swaps = _snapshot(source, ports)
    logical = width + len(controls)
    work = tuple(range(logical, ir.n_wires))
    assert swaps == 0
    events = certify_erasure_histories(
        ir.instructions, n_wires=ir.n_wires, zero_wires=work
    )
    assert events == sum(g.name == "measure" for g in ir.instructions)
    repeated, repeated_swaps = _snapshot(source, ports)
    assert repeated_swaps == swaps and repeated.to_json() == ir.to_json()
    restored = CircuitIR.from_json(ir.to_json())
    assert restored.instructions == ir.instructions
    grid = torch.arange(1 << logical, dtype=torch.float64)
    coherent = torch.complex(grid + 1, grid % 3 - 1)
    coherent /= coherent.norm()
    for vector in (*torch.eye(1 << logical, dtype=torch.complex128), coherent):
        expected = torch.zeros_like(vector)
        for index, amplitude in enumerate(vector):
            data = dict(x=index % (1 << width))
            if controls:
                data["ctrl"] = index >> width
            output = reference.on_classical_vals(**data)
            destination = (int(output.get("ctrl", 0)) << width) + int(output["x"])
            expected[destination] += amplitude
        clean = torch.zeros(1 << len(work), dtype=torch.complex128)
        clean[0] = 1
        for history in ((0,) * events, tuple(i % 2 for i in range(events))):
            torch.testing.assert_close(
                forced_branch(
                    torch.kron(vector, clean), restored.instructions, history
                ),
                torch.kron(expected, clean) * 2 ** (-events / 2),
                atol=3e-12,
                rtol=0,
            )


def test_private_modular_feedback_keys_do_not_shadow_source_records() -> None:
    import cirq

    from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
    from flagquantum.ecosystem.cirq._modular_arithmetic import _private_modular_keys

    a, b, c = cirq.LineQubit.range(3)
    marker = object()
    source = cirq.Circuit(
        cirq.measure(a, key="fq_private_modular_0"),
        cirq.measure(b, key="internal").with_tags(marker),
        cirq.X(c).with_classical_controls("fq_private_modular_0"),
        cirq.Z(c).with_classical_controls("internal").with_tags(marker),
        strategy=cirq.InsertStrategy.NEW,
    )
    renamed = _private_modular_keys(cirq, source, marker)
    assert cirq.measurement_key_names(renamed) == {
        "fq_private_modular_0",
        "fq_private_modular_1",
    }
    ir = import_cirq_feedback(renamed, qubit_order=(a, b, c)).ir
    assert ir.instructions[-2].metadata["conditions"] == ((0, 1),)
    assert ir.instructions[-1].metadata["conditions"] == ((1, 1),)
    assert not any(op.tags for op in renamed.all_operations())
