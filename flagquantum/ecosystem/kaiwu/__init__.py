"""Pure-data interoperability helpers for the QBoson Kaiwu SDK.

This package deliberately excludes authentication, task submission, polling,
and result retrieval.  Those responsibilities belong to ``flagquantum.remote``.
"""

from .matrix import (
    HamiltonianIsingEncoding,
    IntegerPrecisionReport,
    KaiwuInteropError,
    KaiwuMatrixValidationError,
    KaiwuPrecisionError,
    QuboIsingEncoding,
    canonicalize_ising_matrix,
    decode_hamiltonian_spins,
    decode_qubo_spins,
    encode_hamiltonian_as_ising,
    encode_qubo_as_ising,
    ising_energy,
    prepare_integer_precision,
)
from .qdiffusion import bind_qdiffusion_builder, bound_qdiffusion_workflow
from .sampler import KaiwuPrecisionEvidence, KaiwuSampler, KaiwuTransferRecord

__all__ = (
    "HamiltonianIsingEncoding",
    "IntegerPrecisionReport",
    "KaiwuInteropError",
    "KaiwuMatrixValidationError",
    "KaiwuPrecisionError",
    "KaiwuPrecisionEvidence",
    "KaiwuSampler",
    "KaiwuTransferRecord",
    "QuboIsingEncoding",
    "canonicalize_ising_matrix",
    "decode_hamiltonian_spins",
    "bind_qdiffusion_builder",
    "bound_qdiffusion_workflow",
    "decode_qubo_spins",
    "encode_hamiltonian_as_ising",
    "encode_qubo_as_ising",
    "ising_energy",
    "prepare_integer_precision",
)
