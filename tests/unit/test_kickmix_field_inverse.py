"""Check the pinned explicit inverse chain and every measured cleanup phase."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.kickmix_field_inverse import (
    kickmix_field_inverse_chain_erase_ir,
    kickmix_field_inverse_chain_ir,
    kickmix_field_inverse_chain_size,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_field_linear import _source_type
from tests.unit.test_kickmix_field_linear import _block

pytestmark = pytest.mark.unit


def _words(width: int, *, reversed_wires: bool = False) -> tuple[tuple[int, ...], ...]:
    total = 2 * width + kickmix_field_inverse_chain_size(width)
    words = [tuple(range(start, start + width)) for start in range(0, total, width)]
    if reversed_wires:
        words = [word[::-1] for word in words]
    return words[0], words[1], sum(words[2:], ())


def _bits(width: int) -> tuple[int, ...]:
    return tuple(
        reversed(range(width if kickmix_field_inverse_chain_size(width) else 1))
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 8, 9, 17])
def test_explicit_chain_matches_pinned_gate_stream(width: int) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    builder = km.CircuitBuilder()
    source, target = (
        builder.create_quantum_register(width, name=name)
        for name in ("input", "target")
    )
    field = km.GF2Field(width, modulus=0x11B if width == 8 else None)
    target, scaffold = builder.init_gf2_inverse_with_scaffold(
        source, target=target, field=field
    )
    assert len(scaffold) == kickmix_field_inverse_chain_size(width)
    builder.del_gf2_inverse_with_scaffold(
        source, target=target, scaffold=scaffold, field=field
    )
    circuit = str(builder.finish_circuit())
    source_wires, target_wires, chain = _words(width, reversed_wires=True)
    forward = kickmix_field_inverse_chain_ir(
        source_wires, target_wires, chain, modulus=field.modulus
    )
    erase = kickmix_field_inverse_chain_erase_ir(
        source_wires,
        target_wires,
        chain,
        _bits(width),
        modulus=field.modulus,
        parity_bit=width,
    )
    for name, ir in (("gf_inverse_chain", forward), ("gf_uninverse_chain", erase)):
        program = lower_reversible_phase(CircuitIR(2 * width + len(chain), ir))
        reference = [
            line
            for line in _block(circuit, name).splitlines()
            if not line.startswith("INFO_TAG")
        ]
        resets = [line for line in reference if line.startswith("R ")]
        if name == "gf_uninverse_chain" and (width - 1).bit_count() > 1:
            assert resets == [f"R q{wire}" for wire in target_wires[::-1]]
        else:
            assert not resets
        reference = [line for line in reference if not line.startswith("R ")]
        first_toggle = next(
            (i for i, line in enumerate(reference) if line.startswith("BIT_INVERT ")),
            None,
        )
        if first_toggle is not None:
            reference.insert(first_toggle, f"BIT_STORE0 b{width}")
        assert encode_kmx(program).splitlines()[1:] == reference
        if name == "gf_uninverse_chain":
            assert not Counter(op.opcode for op in program.operations)["ccx"]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,modulus", [(1, 3), (2, 7), (3, 11), (4, 19), (5, 37), (8, 283)]
)
def test_every_branch_preserves_all_field_inputs(width: int, modulus: int) -> None:
    np = pytest.importorskip("numpy")
    field = _source_type(width, modulus).gf_type
    values = np.arange(1 << width)
    inverses = values.copy()
    inverses[1:] = field(values[1:]) ** -1
    source, target, chain = _words(width)
    forward = kickmix_field_inverse_chain_ir(source, target, chain, modulus=modulus)
    erase = kickmix_field_inverse_chain_erase_ir(
        source, target, chain, _bits(width), modulus=modulus, parity_bit=width
    )
    wires = [(values >> shift) & 1 for shift in reversed(range(width))]
    original = np.array(wires)
    wires.extend(np.zeros_like(values) for _ in (*target, *chain))
    slots = {}
    phase = np.zeros_like(values)
    outcome = 1
    for block in (forward, erase):
        for op in lower_reversible_phase(CircuitIR(len(wires), block)).operations:
            match op.opcode:
                case "cx":
                    x, y = op.qubits
                    wires[y] ^= wires[x]
                case "ccx":
                    x, y, z = op.qubits
                    wires[z] ^= wires[x] & wires[y]
                case "swap":
                    x, y = op.qubits
                    wires[x], wires[y] = wires[y], wires[x]
                case "hmr":
                    (wire,) = op.qubits
                    slots[op.result_bit] = outcome
                    phase ^= wires[wire] * outcome
                    wires[wire].fill(0)
                    outcome <<= 1
                case "cz":
                    x, y = op.qubits
                    phase ^= (wires[x] & wires[y]) * slots[op.condition_bit]
                case "bit_store0":
                    slots[op.result_bit] = 0
                case "bit_invert":
                    slots[op.result_bit] ^= slots[op.condition_bit]
                case _:
                    pytest.fail(f"Unexpected opcode {op.opcode}")
        if block is forward:
            result = sum(
                wires[wire] << shift for shift, wire in enumerate(reversed(target))
            )
            np.testing.assert_array_equal(result, inverses)
    np.testing.assert_array_equal(wires[:width], original)
    np.testing.assert_array_equal(wires[width:], 0)
    np.testing.assert_array_equal(phase, 0)
    assert slots.get(width, 0) == 0


@pytest.mark.parametrize("width,modulus", [(2, 7), (3, 11)])
def test_append_roundtrip_restores_complex_amplitudes(width: int, modulus: int) -> None:
    source, target, chain = _words(width)
    total = 2 * width + len(chain)
    forward = kickmix_field_inverse_chain_ir(source, target, chain, modulus=modulus)
    erase = kickmix_field_inverse_chain_erase_ir(
        source, target, chain, _bits(width), modulus=modulus, parity_bit=width
    )
    state = torch.zeros(1 << total, dtype=torch.complex128)
    basis = torch.arange(1 << width, dtype=torch.float64)
    amps = torch.complex(basis + 1, 3 * basis - 5)
    state[:: 1 << (total - width)] = amps / amps.norm()
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, (*forward, *erase)))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
