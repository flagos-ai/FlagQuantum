"""Pure-data interoperability helpers for the QBoson Kaiwu SDK.

This package deliberately excludes authentication, task submission, polling,
and result retrieval.  Those responsibilities belong to ``flagquantum.remote``.
"""

from .matrix import (
    IntegerPrecisionReport,
    KaiwuInteropError,
    KaiwuMatrixValidationError,
    KaiwuPrecisionError,
    QuboIsingEncoding,
    canonicalize_ising_matrix,
    decode_qubo_spins,
    encode_qubo_as_ising,
    ising_energy,
    prepare_integer_precision,
)
from .qdiffusion import bind_qdiffusion_builder, bound_qdiffusion_workflow
from .sampler import KaiwuSampler, KaiwuTransferRecord

__all__ = (
    "IntegerPrecisionReport",
    "KaiwuInteropError",
    "KaiwuMatrixValidationError",
    "KaiwuPrecisionError",
    "KaiwuSampler",
    "KaiwuTransferRecord",
    "QuboIsingEncoding",
    "canonicalize_ising_matrix",
    "bind_qdiffusion_builder",
    "bound_qdiffusion_workflow",
    "decode_qubo_spins",
    "encode_qubo_as_ising",
    "ising_energy",
    "prepare_integer_precision",
)
