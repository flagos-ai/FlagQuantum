"""Bind dirty Montgomery products used between elliptic-curve arithmetic stages."""

from dataclasses import dataclass

from ...compiler.registers.modular.single_window_mod_mul import (
    single_window_qroam_table,
)
from ...compiler.registers.modular.windowed_mod_mul import windowed_mod_mul_ir
from ...core.ir import Instruction


@dataclass(frozen=True)
class CurveWindowProduct:
    """Pair a source product with the adjoint that consumes its retained roles."""

    width: int
    modulus: int
    window: int

    def __post_init__(self) -> None:
        n, p, w = self.width, self.modulus, self.window
        if type(n) is not int or n < 2:
            raise ValueError("Curve Montgomery product requires width >= 2")
        if type(p) is not int or p <= 1 or p % 2 == 0 or 2 * p - 1 >= 1 << n:
            raise ValueError(
                "Curve Montgomery product requires odd p > 1 and 2p-1 to fit"
            )
        if type(w) is not int or not 1 <= w <= n or n % w:
            raise ValueError(
                "Curve Montgomery product requires a window dividing the width"
            )

    @property
    def _bank_words(self) -> int:
        return single_window_qroam_table(
            self.width, self.window, self.modulus
        ).block_shape[0]

    @property
    def work_size(self) -> int:
        n, w = self.width, self.window
        return 2 * n + 1 + w + self._bank_words * (n + w) + (n + w - 1)

    def emit(
        self,
        x: tuple[int, ...],
        y: tuple[int, ...],
        workspace: tuple[int, ...],
        bits: tuple[int, ...],
    ) -> tuple[tuple[Instruction, ...], tuple[int, ...], tuple[Instruction, ...]]:
        """Return compute, rotated product wires, and cleanup in matching layouts."""
        n, w = self.width, self.window
        if len(x) != n or len(y) != n or len(workspace) != self.work_size:
            raise ValueError(
                "Curve Montgomery product requires the declared operand and work sizes"
            )
        bank_start = 2 * n + 1 + w
        bank_end = bank_start + self._bank_words * (n + w)
        bank = tuple(
            workspace[i : i + n + w] for i in range(bank_start, bank_end, n + w)
        )
        args = (
            x,
            y,
            workspace[:n],
            workspace[n : 2 * n],
            workspace[2 * n],
            self.modulus,
            workspace[2 * n + 1 : bank_start],
            bank,
            workspace[bank_end:],
            bits,
        )
        forward, result = windowed_mod_mul_ir(*args)
        reverse, _ = windowed_mod_mul_ir(*args, adjoint=True)
        return forward, result, reverse
