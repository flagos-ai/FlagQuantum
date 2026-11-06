"""Independent Vlasov-Ampere block-encoding Circuit, using dense local matrices."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

DTYPE = torch.complex128


def _ry(angle: float) -> torch.Tensor:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[c, -s], [s, c]], dtype=DTYPE)


def _permutation(width: int, map_index: Callable[[int], int]) -> torch.Tensor:
    matrix = torch.zeros((1 << width, 1 << width), dtype=DTYPE)
    for basis in range(1 << width):
        matrix[map_index(basis), basis] = 1
    return matrix


def _embed(matrix: torch.Tensor, wires: tuple[int, ...], width: int) -> torch.Tensor:
    masks = tuple(1 << (width - wire - 1) for wire in wires)
    remove = sum(masks)
    result = torch.zeros((1 << width, 1 << width), dtype=DTYPE)
    for basis in range(1 << width):
        input_index = 0
        for mask in masks:
            input_index = (input_index << 1) | bool(basis & mask)
        for subspace in range(1 << len(wires)):
            output = basis & ~remove
            for bit, mask in enumerate(reversed(masks)):
                if subspace & (1 << bit):
                    output |= mask
            result[output, basis] = matrix[subspace, input_index]
    return result


def _real_preparation(amplitudes: list[float]) -> torch.Tensor:
    """Prefix-conditioned real rotations, derived directly from the amplitudes."""
    width = len(amplitudes).bit_length() - 1
    result = torch.eye(len(amplitudes), dtype=DTYPE)
    for level in range(width):
        tail = 1 << (width - level - 1)
        matrices = []
        for offset in range(0, len(amplitudes), 2 * tail):
            low = amplitudes[offset : offset + tail]
            high = amplitudes[offset + tail : offset + 2 * tail]
            lower, upper = (
                (low[0], high[0])
                if tail == 1
                else (math.hypot(*low), math.hypot(*high))
            )
            rotation = _ry(2 * math.atan2(upper, lower))
            matrices.append(torch.kron(rotation, torch.eye(tail, dtype=DTYPE)))
        result = torch.block_diag(*matrices) @ result
    return result


def _source_array(source: str, function: str, operation: str) -> list[float]:
    text = source.split(f"qfunc {function}(", 1)[1].split("\nq", 1)[0]
    match = re.search(rf"{operation}\([^\[]*?(\[.*?\])", text, re.S)
    assert match
    return ast.literal_eval(match[1])


def _phase_table(source: str, suffix: str) -> torch.Tensor:
    text = source.split(f"qperm apply_phase_table_expanded___{suffix}(", 1)[1].split(
        "\nq", 1
    )[0]
    rotations = [float(angle) for angle in re.findall(r"RZ\(([-\d.]+),", text)]
    [global_angle] = [
        float(angle) for angle in re.findall(r"phase \(([-\d.]+)\)", text)
    ]
    if suffix == "0":
        angles = [global_angle + rotations[0] * (bit - 0.5) for bit in range(2)]
    else:
        angles = [
            global_angle
            + rotations[0] * (low - 0.5)
            + rotations[1] * ((low ^ high) - 0.5)
            + rotations[2] * (high - 0.5)
            for high in range(2)
            for low in range(2)
        ]
    return torch.diag(torch.exp(1j * torch.tensor(angles, dtype=torch.float64)))


def _derivative(source: str) -> torch.Tensor:
    # Derivative local wires: x[2:0] in MSB order, flag, lcu1, lcu2.
    a = _source_array(source, "prepare_select_expanded___0", "inplace_prepare_state")
    prepare_lcu2 = _real_preparation([math.sqrt(value) for value in a])
    plus_minus = torch.block_diag(
        _permutation(4, lambda word: (word - 1) % 16),
        _permutation(4, lambda word: (word + 1) % 16),
    )
    prepare = torch.kron(prepare_lcu2, torch.eye(16, dtype=DTYPE))
    diagonal = torch.kron(_phase_table(source, "0"), torch.eye(16, dtype=DTYPE))
    dirichlet = prepare.mH @ diagonal @ plus_minus @ prepare

    reflection = _permutation(4, lambda word: word ^ (6 if word & 8 else 0))
    z = torch.diag(torch.tensor([1] * 8 + [-1] * 8, dtype=DTYPE))
    a = _source_array(
        source, "prepare_bounday_x_expanded___0", "inplace_prepare_amplitudes"
    )
    inverse = _embed(_real_preparation(a).mH, (1, 2), 4)
    mark = _permutation(4, lambda word: word ^ bool((word >> 1) & 3))
    boundary = reflection @ mark @ inverse @ z @ reflection

    d0 = _embed(dirichlet, (4, 3, 0, 1, 2), 5)
    d1 = _embed(boundary, (0, 1, 2, 3), 5)
    select = _embed(torch.block_diag(d0, d1), (4, 0, 1, 2, 3, 5), 6)
    a = _source_array(source, "prepare_select_expanded___1", "inplace_prepare_state")
    prepare_lcu1 = _embed(_real_preparation([math.sqrt(value) for value in a]), (4,), 6)
    return prepare_lcu1.mH @ select @ prepare_lcu1


def vlasov_reference(source: str) -> fq.Circuit:
    circuit = fq.Circuit(22, dtype=DTYPE)
    for wire in range(7):
        circuit.h(wire).cx(wire, 7 + wire)

    outer = (14, 15)
    electric, x, v = 0, (1, 2, 3), (4, 5, 6)
    lcu2, lcu1, indicator, block_flag, zeta_flag, flag = (16, 17, 18, 19, 20, 21)

    def controlled(
        selector: tuple[int, ...],
        pattern: int,
        wires: tuple[int, ...],
        matrix: torch.Tensor,
    ) -> None:
        identity = torch.eye(matrix.shape[0], dtype=DTYPE)
        diagonal = torch.block_diag(
            *(
                matrix if basis == pattern else identity
                for basis in range(1 << len(selector))
            )
        )
        circuit.any(*selector, *wires, matrix=diagonal)

    def rotate_then_flip(table: list[float]) -> torch.Tensor:
        x_gate = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
        return torch.block_diag(*(x_gate @ _ry(angle) for angle in table))

    probabilities = _source_array(
        source, "prepare_select_expanded___2", "inplace_prepare_state"
    )
    preparation = _real_preparation([math.sqrt(value) for value in probabilities])
    circuit.any(*outer, matrix=preparation)
    controlled(outer, 0, (*x, flag, lcu1, lcu2), _derivative(source))
    table = _source_array(
        source, "assign_amplitude_table_expanded___0", "select_rotation"
    )
    controlled(outer, 0, (*v, indicator), rotate_then_flip(table))

    def mark_zeta(word: int) -> int:
        x_value = word >> 4
        v_raw = (word >> 1) & 7
        signed = v_raw if v_raw < 4 else v_raw - 8
        return word ^ ((x_value == 0 and signed > 0) or (x_value == 7 and signed <= 0))

    controlled(outer, 0, (*x, *v, zeta_flag), _permutation(7, mark_zeta))
    controlled(
        outer,
        0,
        (electric, block_flag),
        _permutation(2, lambda word: word ^ (word >> 1)),
    )

    controlled(outer, 1, (electric,), _permutation(1, lambda word: word ^ 1))
    mark_velocity = _permutation(4, lambda word: word ^ bool(word >> 1))
    a = _source_array(
        source, "load_v_H_vector_expanded___0", "inplace_prepare_amplitudes"
    )
    force = _embed(_real_preparation(a), (0, 1, 2), 4) @ mark_velocity
    a = _source_array(
        source, "load_v_vector_expanded___0", "inplace_prepare_amplitudes"
    )
    current = mark_velocity @ _embed(_real_preparation(a).mH, (0, 1, 2), 4)
    controlled((*outer, electric), 2, (*v, flag), force)
    controlled((*outer, electric), 3, (*v, flag), current)
    table = _source_array(
        source, "assign_amplitude_table_expanded___1", "select_rotation"
    )
    controlled(outer, 1, (electric, indicator), rotate_then_flip(table))

    circuit.any(*outer, matrix=_phase_table(source, "1"))
    circuit.any(*outer, matrix=preparation.mH.contiguous())
    return circuit


pytestmark = pytest.mark.integration


def original() -> Path:
    fixtures = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"
    manifest = json.loads((fixtures / "vlasov_sources.json").read_text())
    [row] = manifest["fixtures"]
    source = fixtures / row["file"]
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((fixtures / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    return source


def test_original_vlasov_preserves_full_state_and_cleans_workspace() -> None:
    path = original()
    program = synth.synthesize(
        path, max_qubits=24, max_workspace=2, memory_limit_bytes=256 * 1024**2
    )
    assert program.report["workspace_qubits"] == 2
    assert dict(program.registers) == {
        "data": tuple(range(7)),
        "data_ref": tuple(range(7, 14)),
        "block": tuple(range(14, 22)),
    }
    expected = fq.run(vlasov_reference(path.read_text())).statevector().reshape(-1)
    result = fq.run(program).statevector().reshape(1 << 22, 4)
    # Compare in slices so the verification does not require extra full buffers.
    for start in range(0, len(expected), 65536):
        actual = result[start : start + 65536]
        torch.testing.assert_close(
            actual[:, 0], expected[start : start + len(actual)], atol=1e-11, rtol=0
        )
        assert actual[:, 1:].abs().max().item() < 1e-11
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(path, max_operations=64)


SIGNED = """
qstruct Outer { inner:Inner; flag:qbit; }
qstruct Inner { value:qnum<3,SIGNED,0>; }
qfunc mark(value:qnum<3,True,0>, flag:qbit) {flag ^= (value < 0);}
qfunc main(output r:Outer) {
  allocate(r); hadamard_transform(r); S(r.flag);
  phase(r.inner.value,0.19);
  mark(r.inner.value,r.flag);
}
"""


def test_signed_nested_readout_phase_and_xor() -> None:
    program = synth.synthesize(SIGNED, source_format="qmod", max_workspace=0)
    reference = fq.Circuit(4, dtype=DTYPE)
    for wire in range(4):
        reference.h(wire)
    reference.s(0)
    values = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1], dtype=torch.float64)
    reference.any(1, 2, 3, matrix=torch.diag(torch.exp(0.19j * values))).cx(1, 0)
    assert program.report["qmod_signed_registers"] == ("r.inner.value",)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_signed_formal_rejects_unsigned_field() -> None:
    with pytest.raises(InteropConversionError, match="signed integer actual"):
        synth.synthesize(
            SIGNED.replace("3,SIGNED,0", "3,UNSIGNED,0"), source_format="qmod"
        )


SELECT = """
qstruct Pair { index:qnum<2>; target:qbit; }
qfunc choose(const index:qnum<2>, target:qbit) {
  repeat (i: 2) {
    control (index == i) {
      if (i == 0) { H(target); } else { RY(0.33,target); }
    }
  }
}
qfunc main(output pair:Pair) {
  allocate(pair); hadamard_transform(pair); S(pair.target);
  choose(pair.index,pair.target);
}
"""


def test_masked_borrowed_selector_stays_coherent() -> None:
    program = synth.synthesize(SELECT, source_format="qmod", max_workspace=0)
    identity = torch.eye(2, dtype=DTYPE)
    h = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    matrix = torch.block_diag(h, _ry(0.33), identity, identity)
    reference = (
        fq.Circuit(3, dtype=DTYPE).h(0).h(1).h(2).s(0).any(1, 2, 0, matrix=matrix)
    )
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


def test_masked_selection_rejects_unproved_selector_mutation() -> None:
    source = """
    qstruct Pair { flag:qbit; }
    qfunc main(output pair:Pair) {
      allocate(pair);
      control (pair.flag == 0) { H(pair.flag); }
    }
    """
    with pytest.raises(
        InteropConversionError, match="selected gate cannot write its control"
    ):
        synth.synthesize(source, source_format="qmod")


def test_source_definition_budget_is_still_bounded() -> None:
    helpers = "".join(f"qfunc aux{i}(a:qbit) {{ H(a); }}" for i in range(64))
    with pytest.raises(InteropConversionError, match="source exceeds 64 definitions"):
        synth.synthesize(
            helpers + "qfunc main(output q:qbit) {allocate(q);}", source_format="qmod"
        )
