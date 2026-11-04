"""Check the full 8-bit circuit without a dense 33-qubit statevector."""

from cmath import exp
from collections import Counter, defaultdict
from dataclasses import replace
from math import sqrt, tau

import numpy as np
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.core.ir import CircuitIR


def _execute_sparse(ir: CircuitIR, state: dict[int, complex]) -> dict[int, complex]:
    """Execute the emitted gate set directly on occupied computational states."""
    for op in ir.instructions:
        masks = tuple(1 << (ir.n_wires - q - 1) for q in op.wires)
        name = op.name
        if name == "h":
            (mask,) = masks
            updated: defaultdict[int, complex] = defaultdict(complex)
            for basis, value in state.items():
                updated[basis & ~mask] += value / sqrt(2)
                updated[basis | mask] += (-value if basis & mask else value) / sqrt(2)
            state = dict(updated)
        elif name in ("phase", "s", "sdg", "cz"):
            mask = sum(masks)
            phases = {"s": 1j, "sdg": -1j, "cz": -1}
            phase = phases[name] if name in phases else exp(1j * op.params["theta"])
            state = {
                basis: phase * value if basis & mask == mask else value
                for basis, value in state.items()
            }
        elif name in ("x", "cx", "ccx"):
            control, target = sum(masks[:-1]), masks[-1]
            state = {
                basis ^ target if basis & control == control else basis: value
                for basis, value in state.items()
            }
        elif name == "cswap":
            control, left, right = masks
            state = {
                (
                    basis ^ (left | right)
                    if basis & control and bool(basis & left) != bool(basis & right)
                    else basis
                ): value
                for basis, value in state.items()
            }
        else:
            raise AssertionError(f"Unexpected gate {name}")
    return state


def test_select_thc_qrom_phase_gradient() -> None:
    op = SelectTHC.make_on(
        mu=(0, 1),
        nu=(2, 3),
        succ=4,
        nu_eq_mp1=5,
        plus_mn=6,
        plus_a=7,
        plus_b=8,
        sys_a=(9, 10),
        sys_b=(11, 12),
        networks=(((0, 1, 0.2),), ((0, 1, 0.5),)),
        one_body_networks=((), ((0, 1, -0.4),)),
        workspace=tuple(range(13, 33)),
    )
    # Success, failure, one-body, both spins, and a padded address in superposition.
    inputs = np.zeros(8192, dtype=np.complex128)
    for index, word in enumerate(
        (
            "0001100000101",
            "0100100100110",
            "0100110010110",
            "1100100000110",
            "0100000100101",
            "0001101011001",
        )
    ):
        inputs[int(word, 2)] = (1j) ** index / sqrt(6)

    def quantized(table):
        step = tau / 256
        return tuple(
            tuple(
                (left, right, round(angle / step) * step) for left, right, angle in net
            )
            for net in table
        )

    exact = replace(op, workspace=())
    rounded = replace(
        exact,
        networks=quantized(op.networks),
        one_body_networks=quantized(op.one_body_networks),
    )
    ir = fq.Circuit(33, dtype=torch.complex128).append(op).to_ir()
    result = _execute_sparse(
        ir, {int(i) << 20: complex(inputs[i]) for i in np.flatnonzero(inputs)}
    )
    logical = np.zeros_like(inputs)
    leakage = 0.0
    for basis, value in result.items():
        if basis & ((1 << 20) - 1):
            leakage += abs(value) ** 2
        else:
            logical[basis >> 20] = value
    expected, unrounded = (
        fq.Circuit(13, dtype=torch.complex128, inputs=torch.from_numpy(inputs))
        .append(reference)
        .state()
        .detach()
        .numpy()
        .ravel()
        for reference in (rounded, exact)
    )
    np.testing.assert_allclose(logical, expected, rtol=0, atol=1e-12)
    assert leakage < 1e-24
    # At most two reflections, each containing a rotation and its adjoint.
    assert np.linalg.norm(logical - unrounded) <= 4 * np.pi / 256
    print("8-bit amplitude error", np.max(np.abs(logical - expected)))
    print("Unquantized amplitude error", np.max(np.abs(logical - unrounded)))
    print("Workspace leakage probability", leakage)
    print("Emitted gates", dict(Counter(op.name for op in ir.instructions)))
