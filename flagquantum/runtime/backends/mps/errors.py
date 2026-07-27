"""Stable exception contracts for the matrix-product-state backend."""


class NonlocalMPSCompilationError(ValueError):
    """A gate cannot execute through an adjacent rank boundary."""


class MPSFullMaterializationError(RuntimeError):
    """Production rank-local MPS results cannot reconstruct the full state."""


class MPSForwardLifetimeError(RuntimeError):
    """A layer-local forward tensor survived beyond its declared last use."""


class MPSReverseContractError(RuntimeError):
    """The sharded reverse tape, transport, or memory contract is invalid."""


__all__ = (
    "MPSForwardLifetimeError",
    "MPSFullMaterializationError",
    "MPSReverseContractError",
    "NonlocalMPSCompilationError",
)
