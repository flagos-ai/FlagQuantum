"""Full stored Itoh--Tsujii roots, aliases, retained junk and clean scratch."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.register_field_inverse import FieldInverse
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.field_inverse import (
    gf2_inverse_decomposition,
    gf2_inverse_workspace_size,
)
from flagquantum.compiler.registers.finite_field.frobenius_component import (
    field_frobenius_program,
)
from flagquantum.compiler.registers.finite_field.inverse_component import (
    field_inverse_program,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_multiplication_program,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_addition import binary_field_encoding
from flagquantum.simulation.finite_field_inverse import field_inverse_classical_values
from tests.unit.test_field_frobenius_component import dtype
from tests.unit.test_field_linear import _source_type

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
FIELDS = ((1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (6, 67), (8, 283), (8, 285))


@pytest.mark.parametrize("width,modulus", FIELDS)
def test_exact_original_root_with_owned_child_adapters(
    width: int, modulus: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import GF2MulViaKaratsuba
    from qualtran.bloqs.gf_arithmetic.gf2_square import GF2Square

    field_type = dtype(width, modulus)
    native = field_inverse_program(field_type, modulus=modulus)
    root = next(f for f in native.functions if f.name == "main")
    assert root.registers is not None
    builder = ComponentProgramBuilder(Signature(root.registers))

    class OriginalCalls:
        def allocate(self, n: int | None = None, dtype: Any = None) -> Any:
            assert dtype == _source_type(width, modulus)
            assert n in (None, width)
            return builder.allocate(dtype=field_type)

        def free(self, value: Any) -> None:
            builder.free(value)

        def add(self, bloq: Any, **values: Any) -> Any:
            if isinstance(bloq, GF2Addition):
                child = field_addition_program(field_type)
            elif isinstance(bloq, GF2Square):
                child = field_frobenius_program(
                    field_type, int(bloq.k), modulus=modulus, adjoint=bloq.uncompute
                )
            else:
                assert isinstance(bloq, GF2MulViaKaratsuba)
                child = karatsuba_multiplication_program(field_type, modulus=modulus)
            outputs = builder.call_program(child, values)
            result = tuple(outputs[r.name] for r in bloq.signature.rights())
            return result[0] if len(result) == 1 else result

    source = GF2Inverse(_source_type(width, modulus))
    outputs = source.build_composite_bloq(OriginalCalls(), **builder.inputs)
    assert builder.finalize(outputs) == native


@pytest.mark.parametrize("width,modulus", FIELDS)
def test_stored_graph_full_native_stream_aliases_and_source_model(
    width: int, modulus: int
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import GF2Inverse
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = tuple(range(width))
    pool = tuple(range(width, 12 * width))
    actual = FieldInverse(dtype(width, modulus), source, pool, modulus).bind()
    expected = gf2_inverse_decomposition(source, pool, modulus=modulus)
    assert actual.circuit.instructions == expected.instructions
    assert actual.outputs == expected.outputs
    assert actual.classical_outputs == ()
    counts = Counter(g.name for g in actual.circuit.instructions)
    model = get_cost_value(GF2Inverse(_source_type(width, modulus)), QECGatesCost())
    assert (counts["ccx"], counts["cx"]) == (model.toffoli, model.clifford)
    assert set(counts) <= {"ccx", "cx"}
    live = set(source)
    for event in actual.events:
        if event.kind == "allocate":
            assert live.isdisjoint(event.wires)
            live.update(event.wires)
        else:
            assert event.kind == "free" and set(event.wires) <= live
            live.difference_update(event.wires)
    assert live == {wire for word in actual.outputs for wire in word}


@pytest.mark.parametrize("width,modulus", FIELDS[:4])
def test_append_all_complex_result_junk_spectator_and_exact_restore(
    width: int, modulus: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flagquantum.compiler.registers.finite_field.field_inverse as physical

    def replaced(*args: object, **kwargs: object) -> None:
        raise AssertionError("Append must bind the stored inverse")

    needed = gf2_inverse_workspace_size(width, modulus=modulus)
    monkeypatch.setattr(physical, "gf2_inverse_decomposition", replaced)
    count = width + needed + 2
    operation = FieldInverse(
        dtype(width, modulus),
        tuple(range(width)),
        tuple(range(width, count - 2)),
        modulus,
    )
    bound = operation.bind()
    encoding = binary_field_encoding(
        _source_type(width, modulus).gf_type.irreducible_poly
    )
    state = torch.zeros(1 << count, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for value in range(1 << width):
        source, inverse, junk = field_inverse_classical_values(
            encoding, encoding.gf_type(value)
        )
        values = [int(source), int(inverse), *(int(v) for v in junk)]
        output = sum(
            ((v >> (width - 1 - i)) & 1) << (count - 1 - w)
            for word, v in zip(bound.outputs, values, strict=True)
            for i, w in enumerate(word)
        )
        for spectator in (0, 1):
            index = value << (count - width) | spectator << 1
            state[index] = torch.tensor(
                complex(value + 1, 2 * spectator - 1), dtype=torch.complex128
            )
            expected[output | spectator << 1] = state[index]
    norm = state.norm()
    state /= norm
    expected /= norm
    circuit = Circuit(count, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=1e-12, rtol=0)
    restored = (
        circuit.compose(CircuitIR(count, tuple(reversed(bound.circuit.instructions))))
        .state()
        .flatten()
    )
    torch.testing.assert_close(restored, state, atol=1e-12, rtol=0)
    assert torch.count_nonzero(restored[1::2]) == 0


@pytest.mark.parametrize("width,modulus", FIELDS[4:])
def test_sparse_full_complex_words_and_clean_padding(width: int, modulus: int) -> None:
    needed = gf2_inverse_workspace_size(width, modulus=modulus)
    count = width + needed + 2
    assert count < 63
    bound = FieldInverse(
        dtype(width, modulus),
        tuple(range(width)),
        tuple(range(width, count - 2)),
        modulus,
    ).bind()
    values = np.repeat(np.arange(1 << width, dtype=np.int64), 2)
    spectators = np.tile(np.array([0, 1], dtype=np.int64), 1 << width)
    initial = (values << (count - width)) | (spectators << 1)
    indices = initial.copy()
    amplitudes = (values + 1 + 1j * (2 * spectators - 1)).astype(np.complex128)
    amplitudes /= np.linalg.norm(amplitudes)
    for gate in bound.circuit.instructions:
        assert gate.name in {"cx", "ccx"}
        enabled = np.ones(len(indices), dtype=np.int64)
        for wire in gate.wires[:-1]:
            enabled &= (indices >> (count - 1 - wire)) & 1
        indices ^= enabled << (count - 1 - gate.wires[-1])
    encoding = binary_field_encoding(
        _source_type(width, modulus).gf_type.irreducible_poly
    )
    expected = []
    for value, spectator in zip(values, spectators, strict=True):
        source, inverse, junk = field_inverse_classical_values(
            encoding, encoding.gf_type(int(value))
        )
        words = [int(source), int(inverse), *(int(v) for v in junk)]
        expected.append(
            sum(
                ((v >> (width - 1 - i)) & 1) << (count - 1 - w)
                for word, v in zip(bound.outputs, words, strict=True)
                for i, w in enumerate(word)
            )
            | int(spectator) << 1
        )
    assert indices.tolist() == expected
    # These permutation primitives preserve the full complex amplitudes exactly.
    assert dict(zip(indices.tolist(), amplitudes.tolist(), strict=True)) == dict(
        zip(expected, amplitudes.tolist(), strict=True)
    )
    output_wires = {w for word in bound.outputs for w in word} | {count - 2}
    clean_mask = sum(
        1 << (count - 1 - w) for w in range(count) if w not in output_wires
    )
    assert not np.any(indices & clean_mask)
