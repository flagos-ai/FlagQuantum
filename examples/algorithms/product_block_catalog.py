"""The original concrete and symbolic Product tutorial cases."""

from collections.abc import Callable, Mapping
from typing import Literal, cast

from sympy import Symbol

from flagquantum.algorithms.primitives.product_block_encoding import (
    ProductBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.circuit import _CircuitOperation
from flagquantum.compiler._component_trace.symbolic_math import (
    SymbolicInt,
    SymbolicReal,
    smax,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature

Binder = Callable[
    [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    _CircuitOperation,
]


def product_child(
    label: Literal["t", "h", "x"],
    *,
    alpha: float = 1.0,
    ancilla: int = 0,
    resource: int = 0,
    epsilon: float = 0.0,
) -> UnitaryBlockEncoding:
    """Construct a tutorial child with the original gate and metadata."""

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("tdg" if label == "t" and inverse else label, ports["q"]),)

    return UnitaryBlockEncoding(
        (("q", 1),),
        emit,
        alpha=alpha,
        ancilla_bitsize=ancilla,
        resource_bitsize=resource,
        epsilon=epsilon,
    )


def product_child_binder(child: UnitaryBlockEncoding) -> Binder:
    """Keep system, ancilla and resource operands in the supplied source order."""

    def bind(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> _CircuitOperation:
        return child.make_on(
            system=ports["system"],
            ancilla=ports["ancilla"],
            resource=ports["resource"],
            workspace=work,
            classical_bits=slots,
            adjoint=inverse,
        )

    return bind


def product_example_signature(block: ProductBlockEncoding) -> Signature:
    """Omit zero-size ports and retain opaque word types."""
    return Signature(
        tuple(
            Register(name, opaque_register_type(width))
            for name, width in (
                ("system", block.system_bitsize),
                ("ancilla", block.ancilla_bitsize),
                ("resource", block.resource_bitsize),
            )
            if width
        )
    )


def product_symbolic_example() -> tuple[Signature, SymbolicReal, SymbolicReal]:
    """Describe the frozen symbolic fixture without attempting physical binding."""
    alpha1, alpha2 = Symbol("alpha1"), Symbol("alpha2")
    a1, a2 = Symbol("a1"), Symbol("a2")
    eps1, eps2 = Symbol("eps1"), Symbol("eps2")
    ancilla = cast(SymbolicInt, smax(a1, a2) + 1)
    signature = Signature(
        (
            Register("system", opaque_register_type(1)),
            Register("ancilla", opaque_register_type(ancilla)),
        )
    )
    return signature, alpha1 * alpha2, alpha1 * eps1 + alpha2 * eps2
