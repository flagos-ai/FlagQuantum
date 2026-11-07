"""Private structural metadata shared by native block compositions."""

from typing import Protocol

from ...compiler._component_trace.symbolic_math import SymbolicReal
from .black_box_prepare import BlackBoxPrepare


class _BlockMetadata(Protocol):
    @property
    def system_bitsize(self) -> int: ...

    @property
    def ancilla_bitsize(self) -> int: ...

    @property
    def resource_bitsize(self) -> int: ...

    @property
    def alpha(self) -> SymbolicReal: ...

    @property
    def epsilon(self) -> SymbolicReal: ...

    @property
    def signal_state(self) -> BlackBoxPrepare: ...
