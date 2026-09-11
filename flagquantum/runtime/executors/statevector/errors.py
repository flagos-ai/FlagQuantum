"""Stable errors for distributed statevector execution."""


class FullStateMaterializationError(RuntimeError):
    """A rank-local production result cannot reconstruct global state."""


__all__ = ("FullStateMaterializationError",)
