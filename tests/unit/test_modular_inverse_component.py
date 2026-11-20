"""Public inverse graph retains directional histories and exact source channels."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.modular.modular_inverse_component import (
    modular_inverse_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.inverse_component_checks import inverse_source_stream
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_inverse_algorithm import assert_complex_maps, cases, pack
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def bound_public(
    n: int,
    p: int,
    *,
    adjoint: bool = False,
    inputs: tuple[tuple[int, ...], ...] | None = None,
) -> BoundRegisterProgram:
    if inputs is None:
        inputs = (
            (tuple(range(n)), tuple(range(n, 5 * n))) if adjoint else (tuple(range(n)),)
        )
    occupied = {wire for word in inputs for wire in word}
    return bind_register_program(
        modular_inverse_program(n, p, adjoint=adjoint),
        inputs,
        clean=tuple(w for w in range(10 * n + 12) if w not in occupied),
        input_ownership=tuple((w, False) for w in sorted(occupied)),
        allow_workspace_outputs=True,
    )


@pytest.mark.parametrize("n,p", [(2, 1), (3, 5), (4, 9), (8, 251)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_inlined_source_stream_directional_history_and_resources(
    n: int, p: int, adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import KaliskiModInverse
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    bound = bound_public(n, p, adjoint=adjoint)
    inputs = {"x": tuple(range(n))}
    if adjoint:
        inputs["junk"] = tuple(range(n, 5 * n))
    source = KaliskiModInverse(n, p, uncompute=adjoint)
    assert_source(
        bound.circuit.instructions, inverse_source_stream(source, bound, inputs)
    )
    count = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert count["t"] + count["tdg"] == model.total_t_count(ts_per_cswap=7)
    assert count["measure"] == model.measurement
    live = {w for word in inputs.values() for w in word}
    peak = len(live)
    for event in bound.events:
        if event.kind == "allocate":
            live.update(event.wires)
        else:
            live.difference_update(event.wires)
        peak = max(peak, len(live))
    assert peak == get_cost_value(source, QubitCount())
    assert live == {w for word in bound.outputs for w in word}


@pytest.mark.parametrize("n,p", [(3, 5), (4, 9), (5, 13)])
def test_every_unit_zero_complex_spectator_history_and_work_restore(
    n: int, p: int
) -> None:
    forward = bound_public(n, p)
    inverse = bound_public(n, p, adjoint=True, inputs=forward.outputs)
    width = 10 * n + 13
    spectator = (width - 1,)
    data = cases(n, p)
    before: list[tuple[int, ...]] = []
    after: list[tuple[int, ...]] = []
    state: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    restored: dict[int, complex] = {}
    norm = (
        sum(
            abs(complex(2 * i + s + 1, 3 * i - s - 4)) ** 2
            for i in range(len(data))
            for s in (0, 1)
        )
        ** 0.5
    )
    for i, (x, y, history, terminal) in enumerate(data):
        junk = (history << (2 * n)) | terminal
        for values, ports, rows in (
            ((x,), (tuple(range(n)),), before),
            ((y, junk), forward.outputs, after),
        ):
            raw = pack(values, ports, width)
            rows.append(tuple((raw >> (width - 1 - w)) & 1 for w in range(width)))
        for s in (0, 1):
            amplitude = complex(2 * i + s + 1, 3 * i - s - 4) / norm
            state[pack((x, s), (tuple(range(n)), spectator), width)] = amplitude
            expected[pack((y, junk, s), (*forward.outputs, spectator), width)] = (
                amplitude
            )
            restored[pack((x, s), (*inverse.outputs, spectator), width)] = amplitude
    actual, events = certified_and_channel_map(
        forward.circuit.instructions, tuple(before)
    )
    assert actual == tuple(after)
    back, back_events = certified_and_channel_map(inverse.circuit.instructions, actual)
    expected_back = tuple(
        tuple(
            (pack((x,), inverse.outputs, width) >> (width - 1 - w)) & 1
            for w in range(width)
        )
        for x, _, _, _ in data
    )
    assert back == expected_back and back_events == events
    for seed in (2, 11, 23):
        seen = execute_sparse_register(
            CircuitIR(width, forward.circuit.instructions), state.copy(), seed=seed
        )
        assert_complex_maps(seen, expected)
        assert_complex_maps(
            execute_sparse_register(
                CircuitIR(width, inverse.circuit.instructions), seen, seed=seed + 2
            ),
            restored,
        )


@pytest.mark.parametrize("adjoint", [False, True])
def test_unchanged_all_ones_loader_failure(adjoint: bool) -> None:
    with pytest.raises(ValueError):
        modular_inverse_program(3, 7, adjoint=adjoint)


def test_normal_append_uses_stored_graph_after_physical_emitter_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum as fq
    import flagquantum.compiler.registers.modular.modular_inverse as physical
    from examples.algorithms.register_modular_inverse import ModularInverse

    def replaced(*args: object, **kwargs: object) -> None:
        raise AssertionError("The append consumer must bind its stored component graph")

    monkeypatch.setattr(physical, "modular_inverse_ir", replaced)
    circuit = fq.Circuit(38)
    op = ModularInverse()
    circuit.append(op)
    assert circuit.to_ir().instructions == op.bind().circuit.instructions
    inverse = ModularInverse(adjoint=True)
    circuit.append(inverse)
    assert circuit.to_ir().instructions == (
        *op.bind().circuit.instructions,
        *inverse.bind().circuit.instructions,
    )


@pytest.mark.parametrize("n,p", [(3, 5), (5, 7)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_append_keeps_existing_physical_ports_and_source_stream(
    n: int, p: int, adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import KaliskiModInverse

    from examples.algorithms.register_modular_inverse import ModularInverse
    from flagquantum.compiler.registers.modular.modular_inverse import (
        modular_inverse_outputs,
    )

    original = tuple(range(n))
    junk = tuple(range(n, 5 * n))
    output, _ = modular_inverse_outputs(original, tuple(range(5 * n, 10 * n + 8)))
    bound = ModularInverse(n, p, adjoint=adjoint).bind()
    assert bound.outputs == ((original,) if adjoint else (output, junk))
    inputs = {"x": output, "junk": junk} if adjoint else {"x": original}
    assert_source(
        bound.circuit.instructions,
        inverse_source_stream(
            KaliskiModInverse(n, p, uncompute=adjoint), bound, inputs
        ),
    )
