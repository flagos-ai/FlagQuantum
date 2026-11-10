"""Read real closed-shell MolecularData HDF5 integrals at the format boundary."""

from __future__ import annotations

import math
import operator
from os import PathLike
from typing import Any

import torch

from ...algorithms.chemistry_domain.problem import MolecularProblem


def _dataset(source: Any, name: str) -> Any:
    """Validate the external h5py object before reading its payload."""
    import h5py

    if not isinstance(source.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"{name} must be an embedded HDF5 dataset")
    value = source[name]
    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
        raise ValueError(f"{name} must be an embedded nonvirtual HDF5 dataset")
    return value


def _integer(source: Any, name: str) -> int:
    value = _dataset(source, name)
    if value.shape != () or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar")
    return int(value[()])


def _real(source: Any, name: str) -> float:
    value = _dataset(source, name)
    if value.shape != () or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real scalar")
    result = float(value[()])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _missing(source: Any, name: str) -> bool:
    if name not in source:
        return True
    value = _dataset(source, name)
    return value.shape == () and value.dtype.kind == "b" and not bool(value[()])


def _integrals(source: Any, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = _dataset(source, name)
    if value.shape != shape or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real integral tensor with shape {shape}")
    # HDF5 converts byte order and integer storage before NumPy crosses into Torch.
    result = torch.from_numpy(value.astype("float64")[()])
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain finite integrals")
    return result


def read_molecular_hdf5(
    path: str | PathLike[str], *, max_orbitals: int = 32
) -> MolecularProblem:
    """Read a restricted real OpenFermion MolecularData file into native tensors.

    Args:
        path: Exact local HDF5 filename, including its suffix if any.
        max_orbitals: Positive spatial-orbital budget, checked before tensor reads.

    Returns:
        Owned chemist-notation integrals in hartree. Nuclear repulsion enters
        once; core_energy is zero. The reference fills the first NELEC/2 spatial
        orbitals. Optional hf_energy is preserved without recomputation.

    Raises:
        ImportError: Install flagquantum[chemistry-hdf5] to enable file input.
        OSError: The file cannot be opened as HDF5.
        ValueError: Required metadata or integrals violate the supported format.
        TypeError: max_orbitals is not an integer.
    """
    if isinstance(max_orbitals, bool) or operator.index(max_orbitals) < 1:
        raise ValueError("max_orbitals must be a positive integer")
    limit = operator.index(max_orbitals)
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "install flagquantum[chemistry-hdf5] for MolecularData HDF5 input"
        ) from exc
    with h5py.File(path, "r") as source:
        n = _integer(source, "n_orbitals")
        if not 1 <= n <= limit:
            raise ValueError("n_orbitals exceeds the positive spatial-orbital budget")
        electrons = _integer(source, "n_electrons")
        if not 0 <= electrons <= 2 * n or electrons % 2:
            raise ValueError("closed-shell n_electrons must be even and fit the space")
        if _integer(source, "multiplicity") != 1:
            raise ValueError("only closed-shell multiplicity=1 is supported")
        if not _missing(source, "n_qubits") and _integer(source, "n_qubits") != 2 * n:
            raise ValueError("n_qubits must equal twice n_orbitals")
        nuclear = _real(source, "nuclear_repulsion")
        reference = (
            None if _missing(source, "hf_energy") else _real(source, "hf_energy")
        )
        one = _integrals(source, "one_body_integrals", (n, n))
        # MolecularData[p,q,r,s]=(ps|qr); native[p,q,r,s]=(pq|rs).
        two = _integrals(source, "two_body_integrals", (n, n, n, n)).permute(0, 3, 1, 2)
    return MolecularProblem(
        one, two, electrons, nuclear_repulsion=nuclear, reference_energy=reference
    )
