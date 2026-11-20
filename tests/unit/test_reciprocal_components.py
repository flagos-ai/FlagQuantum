"""Stored reciprocal lifetimes, mixed formats and exact measured channels."""

from collections import Counter
from fractions import Fraction
from itertools import product
from math import floor, sqrt

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.fixed_point.reciprocal_component import (
    reciprocal_program,
)
from flagquantum.compiler.registers.fixed_point.reciprocal_newton import (
    reciprocal_newton_requirements,
    reciprocal_newton_step_ir,
)
from flagquantum.compiler.registers.fixed_point.reciprocal_newton_component import (
    reciprocal_newton_program,
)
from flagquantum.compiler.registers.fixed_point.reciprocal_seed import (
    reciprocal_seed_ir,
)
from flagquantum.compiler.registers.fixed_point.reciprocal_seed_component import (
    reciprocal_seed_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _width(bound: BoundRegisterProgram, logical: int) -> int:
    return (
        max(
            (q for g in bound.circuit.instructions for q in g.wires),
            default=logical - 1,
        )
        + 2
    )


def _complex_channel(
    bound: BoundRegisterProgram,
    logical: int,
    images: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    width = _width(bound, logical)
    tail = width - logical
    before: dict[int, complex] = {}
    after: dict[int, complex] = {}
    for (initial, image), spectator in product(images, (0, 1)):
        value = complex(initial + 1, image - initial + spectator + 1)
        before[(initial << tail) | spectator] = value
        after[(image << tail) | spectator] = value
    norm = sqrt(sum(abs(v) ** 2 for v in before.values()))
    before, after = (
        {i: v / norm for i, v in state.items()} for state in (before, after)
    )
    basis = tuple(tuple(i >> q & 1 for q in reversed(range(width))) for i in before)
    expected = tuple(tuple(i >> q & 1 for q in reversed(range(width))) for i in after)
    actual, events = certified_and_channel_map(bound.circuit.instructions, basis)
    assert actual == expected
    assert events == sum(g.name == "measure" for g in bound.circuit.instructions)
    # The certificate validates every conditional erase for every input, hence
    # every history. Actual primitives additionally check coherent trajectories.
    for seed in (2, 31):
        state = execute_sparse_register(
            CircuitIR(width, bound.circuit.instructions), before, seed=seed
        )
        assert all(
            abs(state.get(i, 0) - after.get(i, 0)) < 8e-12
            for i in state.keys() | after.keys()
        )
    if width > 14:
        return torch.empty(0), torch.empty(0)
    initial_vector = torch.zeros(1 << width, dtype=torch.complex128)
    expected_vector = torch.zeros_like(initial_vector)
    for i, v in before.items():
        initial_vector[i] = torch.tensor(v, dtype=torch.complex128)
    for i, v in after.items():
        expected_vector[i] = torch.tensor(v, dtype=torch.complex128)
    return initial_vector, expected_vector


@pytest.mark.parametrize("n,f,p", ((1, 0, 1), (2, 0, 2), (3, 1, 3), (5, 1, 6)))
@pytest.mark.parametrize("expanded", (False, True))
def test_owned_seed_source_cleanup_and_inverse(
    n: int, f: int, p: int, expanded: bool
) -> None:
    a, output = tuple(range(n)), tuple(range(n, n + p))
    work = tuple(range(n + p, n + p + n - f))
    for adjoint in (False, True):
        record = reciprocal_seed_program(n, f, p, adjoint=adjoint)
        assert verify_program_record(record) == ()
        if expanded:
            record, _ = expand_component_calls(record)
        bound = bind_register_program(
            record,
            (a, output) if adjoint else (a,),
            clean=work if adjoint else (*output, *work),
            allow_workspace_outputs=not adjoint,
            input_ownership=tuple((q, False) for q in output) if adjoint else (),
        )
        assert bound.outputs == ((a,) if adjoint else (a, output))
        gates = bound.circuit.instructions
        flag = next(
            g.wires[1]
            for g in gates
            if g.name == "cx" and g.wires[0] == a[0] and g.wires[1] not in output
        )
        scratch = tuple(
            g.wires[0]
            for g, following in zip(gates[:-1], gates[1:], strict=True)
            if g.name == "h" and following.name == "t"
        )
        assert len(scratch) == n - f - 1
        original = reciprocal_seed_ir(
            a,
            output,
            flag,
            scratch,
            tuple(range(len(scratch))),
            source_fraction_bits=f,
            adjoint=adjoint,
        )
        assert_source(
            normalized_feedback(bound.circuit.instructions),
            normalized_feedback(original),
        )
        images = []
        for value in range(1 << n):
            exponent = (value >> f).bit_length()
            seed = 1 << (p - exponent) if exponent else 0
            zero, seeded = value << p, value << p | seed
            images.append((seeded, zero) if adjoint else (zero, seeded))
        _complex_channel(bound, n + p, tuple(images))


@pytest.mark.parametrize(
    "n,f,p", ((1, 0, 1), (2, 0, 1), (2, 0, 2), (2, 1, 2), (3, 1, 3))
)
@pytest.mark.parametrize("expanded", (False, True))
def test_newton_complete_raw_domain_and_owned_lifetimes(
    n: int, f: int, p: int, expanded: bool
) -> None:
    widths = (n, p, p)
    a, estimate, output = (
        tuple(range(n)),
        tuple(range(n, n + p)),
        tuple(range(n + p, n + 2 * p)),
    )
    logical = sum(widths)
    record = reciprocal_newton_program(n, f, p)
    assert verify_program_record(record) == ()
    if expanded:
        record, _ = expand_component_calls(record)
    bound = bind_register_program(
        record, (a, estimate, output), clean=tuple(range(logical, logical + 200))
    )
    assert bound.outputs == (a, estimate, output)
    work, slots = reciprocal_newton_requirements(n, p)
    flat = reciprocal_newton_step_ir(
        a,
        estimate,
        output,
        tuple(range(logical, logical + work)),
        tuple(range(slots)),
        source_fraction_bits=f,
    )
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in flat
    )
    allocated = sum(len(e.wires) for e in bound.events if e.kind == "allocate")
    freed = sum(len(e.wires) for e in bound.events if e.kind == "free")
    assert allocated == freed
    images = []
    for word, value, occupied in product(range(1 << n), range(1 << p), range(1 << p)):
        x = Fraction(value, 1 << p)
        update = floor((2 * x - Fraction(word, 1 << f) * x * x) * (1 << p)) % (1 << p)
        prefix = ((word << p) | value) << p
        images.append((prefix | occupied, prefix | (occupied ^ update)))
    initial, expected = _complex_channel(bound, logical, tuple(images))
    if n == p == 1:
        events = sum(g.name == "measure" for g in bound.circuit.instructions)
        for outcomes in product((0, 1), repeat=events):
            torch.testing.assert_close(
                forced_branch(initial, bound.circuit.instructions, outcomes),
                expected / sqrt(2**events),
                atol=5e-12,
                rtol=0,
            )


def _reciprocal_image(word: int, n: int, f: int) -> int:
    if f <= 0 or word < 1 << f:
        return 0
    a = Fraction(word, 1 << f)
    estimate = Fraction(1, 1 << floor(a).bit_length())
    for _ in range((n - 1).bit_length()):
        estimate = Fraction(
            floor((2 * estimate - a * estimate * estimate) * (1 << n)), 1 << n
        )
    return floor(estimate * (1 << f))


@pytest.mark.parametrize(
    "n,f", ((1, -7), (1, 0), (3, -1), (3, 0), (2, 1), (3, 1), (3, 2))
)
@pytest.mark.parametrize("expanded", (False, True))
def test_original_mixed_thru_signature_and_coherent_occupied_result(
    n: int, f: int, expanded: bool
) -> None:
    from qualtran.bloqs.arithmetic.multiplication import InvertRealNumber

    original = InvertRealNumber(n, f)
    record = reciprocal_program(n, f)
    assert verify_program_record(record) == ()
    ports = record.functions[0].registers
    assert ports is not None
    assert [
        (r.name, r.side.name, r.dtype.quantum_bits, dict(r.dtype.parameters))
        for r in ports
    ] == [
        (
            r.name,
            r.side.name,
            r.bitsize,
            {"fractional_width": r.dtype.num_frac, "signed": False},
        )
        for r in original.signature
    ]
    if expanded:
        record, _ = expand_component_calls(record)
    a, output = tuple(range(n)), tuple(range(n, 2 * n))
    bound = bind_register_program(
        record, (a, output), clean=tuple(range(2 * n, 2 * n + 250))
    )
    assert bound.outputs == (a, output)
    if f <= 0:
        assert not bound.circuit.instructions and not bound.events
    for event in bound.events:
        assert not set(event.wires) & set((*a, *output))
    images = tuple(
        (
            (word << n) | occupied,
            (word << n) | (occupied ^ _reciprocal_image(word, n, f)),
        )
        for word, occupied in product(range(1 << n), repeat=2)
    )
    _complex_channel(bound, 2 * n, images)
    # The fixed-width lower approximation includes zero at a=1 for one fraction
    # bit. It keeps the occupied integer prefix and has a self-inverse XOR map.
    for before, after in images:
        assert (before >> f if f > 0 else before) == (after >> f if f > 0 else after)


def test_append_stored_paths_replace_flat_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers.fixed_point import (
        reciprocal,
        reciprocal_newton,
        reciprocal_seed,
    )

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("Flat reciprocal roots must be replaceable")

    for module, name in (
        (reciprocal, "reciprocal_ir"),
        (reciprocal_newton, "reciprocal_newton_step_ir"),
        (reciprocal_seed, "reciprocal_seed_ir"),
    ):
        monkeypatch.setattr(module, name, refuse)
    bound = bind_register_program(
        reciprocal_program(2, 1), ((0, 1), (2, 3)), clean=tuple(range(4, 50))
    )
    width = _width(bound, 4)
    root_circuit = DynamicCircuit(width)
    root_op = FieldOperation(width, bound.circuit.instructions)
    root_circuit.append(root_op)
    root_circuit.append(root_op)
    root_state = {(0b1011 << (width - 4)) | 1: (1 + 2j) / sqrt(5)}
    root_actual = execute_sparse_register(root_circuit.to_ir(), root_state, seed=11)
    assert all(
        abs(root_actual.get(i, 0) - root_state.get(i, 0)) < 8e-12
        for i in root_actual.keys() | root_state.keys()
    )
    # Sparse primitive execution covers the complete root. The smallest stored
    # Newton step additionally exercises the ordinary dynamic append entry.
    newton = bind_register_program(
        reciprocal_newton_program(1, 0, 1),
        ((0,), (1,), (2,)),
        clean=tuple(range(3, 15)),
    )
    assert width > 4 and len(bound.circuit.instructions) > 0
    width = _width(newton, 3)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[(0b111 << (width - 3)) | 1] = torch.tensor(
        (1 + 2j) / sqrt(5), dtype=torch.complex128
    )
    circuit = DynamicCircuit(width, inputs=state, dtype=torch.complex128)
    op = FieldOperation(width, newton.circuit.instructions)
    circuit.append(op)
    circuit.append(op)
    actual = run_dynamic(circuit, shots=4, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states.reshape(4, -1), state.expand(4, -1), atol=5e-12, rtol=0
    )


@pytest.mark.parametrize("n,f", ((0, 0), (True, 0), (2, 2), (2, True)))
def test_invalid_root_parameters(n: int, f: int) -> None:
    with pytest.raises(ValueError):
        reciprocal_program(n, f)
