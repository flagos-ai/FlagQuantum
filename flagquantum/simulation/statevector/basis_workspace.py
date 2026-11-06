"""Check native gates using a deterministic computational-basis workspace map."""

import cmath
import math
from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ...core.ir import CircuitIR, Instruction

_S = 2**-0.5


class _UnsupportedWorkspaceError(ValueError):
    pass


def _propagate(
    instructions: Sequence[Instruction], data: int
) -> NDArray[np.complex128]:
    """Apply only gates whose exact computational workspace labels stay defined."""
    size = 1 << data
    indices = np.arange(size, dtype=np.int64)
    work = np.zeros(size, dtype=np.int64)
    state = np.zeros(size, dtype=np.complex128)
    state[0] = 1
    mixed = {"h", "ry", "rx", "u3"}
    for instruction in instructions:
        op, wires = instruction.name, instruction.wires
        target = wires[-1]
        if instruction.matrix is not None or instruction.metadata.get("is_channel"):
            raise _UnsupportedWorkspaceError(
                "compact workspace requires ordinary native gates"
            )
        phi = instruction.params.get("theta", 0)
        if any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in instruction.params.values()
        ):
            raise _UnsupportedWorkspaceError(
                "compact workspace requires finite static gate parameters"
            )

        def bit(wire: int, work_words: NDArray[np.int64] = work) -> NDArray[np.int64]:
            return (
                indices >> (data - wire - 1) & 1
                if wire < data
                else work_words >> (wire - data) & 1
            )

        if op == "swap":
            if max(wires) >= data:
                raise _UnsupportedWorkspaceError("scratch swap outside profile")
            difference = bit(wires[0]) ^ bit(wires[1])
            offsets = difference * (
                (1 << (data - 1 - wires[0])) | (1 << (data - 1 - wires[1]))
            )
            perm = indices ^ offsets
            work = work[perm]
            state = state[perm]
        elif op in {"x", "cx", "ccx"}:
            controls = wires[:-1]
            selected = bit(controls[0]) if controls else 1
            for wire in controls[1:]:
                selected = selected & bit(wire)
            if target >= data:
                work ^= selected << (target - data)
            else:
                perm = indices ^ (selected << (data - target - 1))
                # Every basis word is a unique label in the same workspace
                # fiber. A target-dependent scratch predicate would violate this.
                if not np.array_equal(perm[perm], indices):
                    raise _UnsupportedWorkspaceError(
                        f"work map cannot preserve reversible target {instruction}"
                    )
                work = work[perm]
                state = state[perm]
        elif op in mixed:
            if target >= data:
                raise _UnsupportedWorkspaceError(
                    "workspace cannot be in a superposition"
                )
            permutation = indices ^ (1 << (data - target - 1))
            if not np.array_equal(work[permutation], work):
                raise _UnsupportedWorkspaceError(
                    f"quantum gate leaves the classical workspace domain {instruction}"
                )
            flipped = state[permutation]
            if op == "h":
                state = _S * (state * (1 - 2 * bit(target)) + flipped)
            elif op == "ry":
                state = (
                    math.cos(phi / 2) * state
                    + math.sin(phi / 2) * (2 * bit(target) - 1) * flipped
                )
            elif op == "rx":
                state = math.cos(phi / 2) * state - 1j * math.sin(phi / 2) * flipped
            else:
                theta, phase, lam = (
                    instruction.params[name] for name in ("theta", "phi", "lbd")
                )
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                phi_unit, lam_unit = cmath.exp(1j * phase), cmath.exp(1j * lam)
                target_one = bit(target)
                state = np.where(
                    target_one,
                    c * phi_unit * lam_unit * state + s * phi_unit * flipped,
                    c * state - s * lam_unit * flipped,
                )
        elif op in {"phase", "cphase", "cz", "rz", "z", "s", "sdg"}:
            mask = bit(target)
            if op in {"cphase", "cz"}:
                mask &= bit(wires[0])
            if op in {"cz", "z"}:
                phase = -1
            elif op == "s":
                phase = 1j
            elif op == "sdg":
                phase = -1j
            else:
                phase = cmath.exp(1j * phi)
            state *= np.where(mask, phase, 1)
            if op == "rz":
                state *= cmath.exp(-0.5j * phi)
        else:
            raise _UnsupportedWorkspaceError(f"unsupported {op}")
    if np.any(work):
        raise _UnsupportedWorkspaceError("workspace did not return to zero")
    return state


def compact_workspace_state(program: CircuitIR) -> torch.Tensor | None:
    prefix = program.metadata.get("construction_basis_workspace_prefix")
    if (
        type(prefix) is not int
        or not 1 <= prefix <= 16
        or not 1 <= program.n_wires - prefix <= 12
        or len(program.instructions) > 100_000
        or program.observables
        or program.measurements
    ):
        return None
    try:
        state = _propagate(program.instructions, prefix)
    except _UnsupportedWorkspaceError:
        return None
    # Materialize the exact standard return shape. Only the zero workspace
    # slice is occupied, as certified by the gate-by-gate workspace proof.
    result = torch.zeros(
        (1 << prefix, 1 << (program.n_wires - prefix)), dtype=torch.complex128
    )
    result[:, 0] = torch.from_numpy(state)
    return result.reshape(1, -1)
