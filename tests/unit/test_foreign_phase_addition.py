"""Import actual source phase arithmetic through its existing native realization."""

from collections import Counter
from hashlib import sha1
from importlib.metadata import version
from itertools import product
from math import sqrt
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.phase.phase_addition import phase_add_workspace_size
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from tests._reference_dependencies import unavailable_reference
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    import kickmix
    from qualtran.bloqs.rotations import phase_gradient

    assert version("qualtran") == "0.7.0"
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    raw = Path(phase_gradient.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "97885bc9d4c054df375854ea6d9d4d3afecb63e4"
    )


@pytest.mark.parametrize(
    "m,n,shift", [(1, 1, 0), (2, 2, 0), (3, 2, 0), (2, 3, 0), (2, 3, 1), (2, 3, 3)]
)
@pytest.mark.parametrize("cv,sign", tuple(product((None, 0, 1), (-1, 1))))
def test_source_permutation_all_inputs_and_complete_erased_channel(
    m: int, n: int, shift: int, cv: int | None, sign: int
) -> None:
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    source = AddIntoPhaseGrad(m, n, right_shift=shift, sign=sign, controlled_by=cv)
    ports = tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
    ir, swaps = _snapshot(source, ports)
    logical = m + n + int(cv is not None)
    work = phase_add_workspace_size(m, n, right_shift=shift, controlled=cv is not None)
    assert ir.n_wires == logical + work and swaps == 0
    restored = CircuitIR.from_json(ir.to_json())
    assert restored.to_json() == ir.to_json()
    inputs = tuple(
        tuple((i >> (logical - 1 - j)) & 1 for j in range(logical)) + (0,) * work
        for i in range(1 << logical)
    )
    actual, events = certified_and_channel_map(restored.instructions, inputs)
    outputs = []
    for i in range(1 << logical):
        c = i >> (m + n) if cv is not None else None
        x, y = (i >> n) % (1 << m), i % (1 << n)
        kwargs = {"x": x, "phase_grad": y}
        if c is not None:
            kwargs["ctrl"] = c
        value = source.on_classical_vals(**kwargs)
        out = ((c or 0) << (m + n)) | (int(value["x"]) << n) | int(value["phase_grad"])
        outputs.append(out)
    expected = tuple(
        tuple((i >> (logical - 1 - j)) & 1 for j in range(logical)) + (0,) * work
        for i in outputs
    )
    assert actual == expected and events == work
    counts = Counter(g.name for g in ir.instructions)
    assert counts["t"] + counts["tdg"] == 4 * events
    initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
    desired = torch.zeros_like(initial)
    amplitudes = torch.complex(
        torch.arange(1, (1 << logical) + 1, dtype=torch.float64),
        torch.arange(1 << logical, dtype=torch.float64) - 2,
    )
    amplitudes /= amplitudes.norm()
    stride = 1 << work
    initial[::stride] = amplitudes
    for i, out in enumerate(outputs):
        desired[out << work] = amplitudes[i]
    histories = (
        tuple(product((0, 1), repeat=events)) if events <= 3 else ((0,) * events,)
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(initial, restored.instructions, history),
            desired / sqrt(2**events),
            atol=4e-12,
            rtol=0,
        )


def test_missing_original_physical_recipe_is_not_an_emission_model() -> None:
    from qualtran._infra.bloq import DecomposeNotImplementedError
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    source = AddIntoPhaseGrad(2, 2)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    assert get_cost_value(source, QECGatesCost()).toffoli == 0
    ir, _ = _snapshot(
        source, tuple((r.name, int(r.total_bits())) for r in source.signature)
    )
    counts = Counter(g.name for g in ir.instructions)
    assert counts["t"] + counts["tdg"] == 4 and counts["measure"] == 1


def test_private_phase_keys_do_not_alias_source_measurements() -> None:
    try:
        import cirq
    except ModuleNotFoundError as _reference_import_error:
        unavailable_reference(_reference_import_error)
    from attrs import frozen
    from qualtran._infra.gate_with_registers import GateWithRegisters
    from qualtran._infra.registers import Signature
    from qualtran.bloqs.rotations.phase_gradient import AddIntoPhaseGrad

    @frozen
    class MeasuredPhase(GateWithRegisters):
        @property
        def signature(self) -> Any:
            return Signature.build(x=2, phase_grad=2, flag=1)

        def decompose_from_registers(self, *, context: Any, **quregs: Any) -> Any:
            yield cirq.measure(*quregs["flag"], key="fq_private_modular_0")
            for sign in (1, -1):
                yield AddIntoPhaseGrad(2, 2, sign=sign).on_registers(
                    x=quregs["x"], phase_grad=quregs["phase_grad"]
                )

    source = MeasuredPhase()
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    ir, _ = _snapshot(source, ports)
    assert set(ir.metadata["interop"]["measurement_records"]) == {
        "fq_private_modular_0",
        "fq_private_modular_1",
        "fq_private_modular_2",
    }
    repeated, _ = _snapshot(source, ports)
    assert repeated.to_json() == ir.to_json()
