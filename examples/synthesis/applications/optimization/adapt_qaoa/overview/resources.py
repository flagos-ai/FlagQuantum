"""Measure the source gate basis through public post-synthesis conversion."""

from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis

SOURCE_BASIS = ("cz", "rz", "rx", "ry")


@dataclass(frozen=True)
class BasisMetrics:
    gates: tuple[str, ...]
    resources: ResourceEstimate
    conversion_identity: str
    maximum_state_error: float


def source_basis_metrics(program: synth.Program) -> BasisMetrics:
    conversion = convert_basis(
        program.circuit, gates=SOURCE_BASIS, max_added_operations=16384
    )
    estimate = estimate_resources(conversion.program)
    if set(estimate.operation_counts) - set(SOURCE_BASIS):
        raise ValueError("the reported resource count escaped the source basis")
    native = fq.run(program).statevector()
    converted = fq.run(conversion.program).statevector()
    torch.testing.assert_close(native, converted, rtol=0, atol=2e-12)
    return BasisMetrics(
        SOURCE_BASIS,
        estimate,
        conversion.conversion_identity,
        float((native - converted).abs().max()),
    )
