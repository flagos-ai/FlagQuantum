"""Explicit import of already populated optional MolecularData objects."""

from __future__ import annotations

from importlib import import_module
from numbers import Integral, Real

import torch

from ...algorithms.chemistry_domain.problem import MolecularProblem
from ..openfermion import _upstream


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool or a tensor")
    return int(value)


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar")
    return float(value)


def from_openfermion_molecular_data(
    source: object, *, max_orbitals: int = 32
) -> MolecularProblem:
    """Snapshot populated real closed-shell spatial integrals into native data.

    OpenFermion is required only on explicit invocation. Both integral arrays
    must already be cached in the object; this import never loads a file or
    runs a calculation. It converts upstream (ps|qr) to native (pq|rs), without
    a half-factor. Nuclear repulsion enters once; optional HF energy is retained.
    The reference fills the first n_electrons/2 spatial orbitals.

    The positive orbital budget is checked before copying dense O(n**4) data.
    Wrong types raise TypeError; unsupported metadata, shapes, nonfinite values
    or integral symmetries raise ValueError. No source arrays are retained.
    See the chemistry README for the runnable conversion and measurement path.
    """
    limit = _integer(max_orbitals, "max_orbitals")
    if limit < 1:
        raise ValueError("max_orbitals must be positive")
    upstream = _upstream()
    if not isinstance(source, upstream.MolecularData):
        raise TypeError("source must be an OpenFermion MolecularData")
    n = _integer(getattr(source, "n_orbitals", None), "n_orbitals")
    if not 1 <= n <= limit:
        raise ValueError("n_orbitals exceeds the positive spatial-orbital budget")
    electrons = _integer(getattr(source, "n_electrons", None), "n_electrons")
    if _integer(getattr(source, "multiplicity", None), "multiplicity") != 1:
        raise ValueError("only closed-shell multiplicity=1 is supported")
    width = getattr(source, "n_qubits", None)
    if width is not None and _integer(width, "n_qubits") != 2 * n:
        raise ValueError("n_qubits must equal twice n_orbitals")
    nuclear = _real(getattr(source, "nuclear_repulsion", None), "nuclear_repulsion")
    reference = getattr(source, "hf_energy", None)
    if reference is not None:
        reference = _real(reference, "hf_energy")
    numpy = import_module("numpy")
    integrals = []
    for name, shape in (
        ("one_body_integrals", (n, n)),
        ("two_body_integrals", (n, n, n, n)),
    ):
        # In upstream 1.7.1 the public properties can load files on a cache miss.
        value = getattr(source, "_" + name, None)
        if not isinstance(value, numpy.ndarray):
            raise TypeError(f"{name} must already be a cached dense NumPy array")
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(
                f"{name} must be a real integral tensor with shape {shape}"
            )
        integrals.append(torch.from_numpy(value.astype("float64", copy=True)))
    return MolecularProblem(
        integrals[0],
        integrals[1].permute(0, 3, 1, 2),
        electrons,
        nuclear_repulsion=nuclear,
        reference_energy=reference,
    )
