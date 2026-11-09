"""Annotation vocabulary and captured operations for bounded Python programs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from types import GenericAlias
from typing import Annotated, Generic, TypeAlias, TypeVar, overload

_Num = TypeVar("_Num", bound="_QuantumRegister")
_Element = TypeVar("_Element", bound="QNum")


class _QuantumRegister:
    """Shared annotation base for owned or borrowed quantum storage."""


def _width_spec(spec: object) -> int:
    args = spec if isinstance(spec, tuple) else (spec,)
    if len(args) not in {1, 3}:
        raise TypeError("QNum expects width or (width, False, 0)")
    width = args[0]
    if type(width) is not int or not 1 <= width <= 256:
        raise ValueError("QNum width must be a plain integer in 1..256")
    if len(args) == 3 and (
        args[1] is not False or type(args[2]) is not int or args[2] != 0
    ):
        raise ValueError("QNum supports unsigned integers only: False, 0")
    return width


class QNum(_QuantumRegister):
    """Quantum unsigned-number annotation, not a host numeric value.

    Use Output[QNum] with allocate(size, out), or QNum[width, False, 0]
    in a captured annotation. Program arithmetic is compiled from retained AST.
    """

    def __new__(cls) -> QNum:
        raise TypeError("quantum values are declared as qfunc ports, not host objects")

    def __class_getitem__(cls, spec: object) -> GenericAlias:
        if cls is not QNum:
            raise TypeError("QBit has a fixed one-bit type")
        _width_spec(spec)
        return GenericAlias(cls, spec)

    def __add__(self, other: QNum | int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __radd__ = __add__

    def __mul__(self, other: QNum | int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    def __mod__(self, divisor: int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    def __and__(self, other: QNum | int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __rand__ = __and__

    def __or__(self, other: QNum | int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __ror__ = __or__

    def __xor__(self, other: QNum | int) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __rxor__ = __xor__

    def __invert__(self) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    def __eq__(self, other: object) -> bool:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __ne__(self, other: object) -> bool:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __lt__(self, other: QNum | int) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __le__(self, other: QNum | int) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __gt__(self, other: QNum | int) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __ge__(self, other: QNum | int) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __ior__(self: _Num, expression: QNum | int) -> _Num:
        raise TypeError("fresh quantum assignment is captured inside qfunc")


class QBit(QNum):
    """One-bit quantum annotation, with the same captured-port ownership rules."""


class QArray(_QuantumRegister, Generic[_Element]):
    """Captured quantum array; the current profile admits QArray[QBit]."""

    def __new__(cls) -> QArray[_Element]:
        raise TypeError("quantum arrays are declared as qfunc ports, not host objects")

    @overload
    def __getitem__(self, index: int) -> _Element: ...

    @overload
    def __getitem__(self, index: slice) -> QArray[_Element]: ...

    def __getitem__(self, index: int | slice) -> _Element | QArray[_Element]:
        raise TypeError("quantum array views are captured inside qfunc")

    def __len__(self) -> int:
        raise TypeError("quantum array length is captured inside qfunc")


Output: TypeAlias = Annotated[_Num, "flagquantum.synth.output"]
Const: TypeAlias = Annotated[_Num, "flagquantum.synth.const"]


def allocate(
    size_or_out: int | QNum | QArray[QBit],
    out: QNum | QArray[QBit] | None = None,
    /,
) -> None:
    """Declare fresh zero storage inside a captured qfunc.

    Pass a sized output, or a literal width and an unsigned output. This
    operation cannot run in the host interpreter; use synthesize and fq.run.
    """
    raise TypeError("allocate is captured inside qfunc; synthesize the program")


def hadamard_transform(out: QNum | QArray[QBit], /) -> None:
    """Prepare uniform amplitudes on an initialized captured quantum port.

    The compiler applies Hadamard to each logical bit. This operation cannot
    run in the host interpreter; use synthesize and fq.run.
    """
    raise TypeError("hadamard_transform is captured inside qfunc")


def unitary(
    matrix: Sequence[Sequence[complex | float | int]],
    target: QNum | QArray[QBit] | Sequence[QBit],
    /,
) -> None:
    """Apply a bounded static matrix to writable quantum storage inside qfunc.

    Arrays and explicit scalar lists use logical register order: position zero
    is the least significant matrix bit. Matrices must be unitary.
    """
    raise TypeError("unitary is captured inside qfunc")


def qft(target: QNum | QArray[QBit], /) -> None:
    """Apply the positive-sign Fourier transform to a complete writable register.

    Element zero is the least significant bit; the transform includes final bit
    reversal. Capture this call inside qfunc and execute the synthesized program.
    """
    raise TypeError("qft is captured inside qfunc")


def modular_exp(n: int, a: int, x: QNum, power: QNum, /) -> None:
    """Apply x <- a**power * x mod n on the valid residue subspace.

    Values of x at least n stay unchanged. The static base must be coprime to
    n; initialized target and exponent registers must be distinct. Capture the
    operation inside qfunc, then synthesize and execute the program.
    """
    raise TypeError("modular_exp is captured inside qfunc")


def phase(expression: QNum | float, coefficient: float, /) -> None:
    """Apply exp(i * coefficient * expression) inside a captured qfunc."""
    raise TypeError("phase is captured inside qfunc")


def cx(control: QBit, target: QBit, /) -> None:
    """Apply controlled X to initialized scalar operands inside qfunc."""
    raise TypeError("CX is captured inside qfunc")


def swap(first: QBit, second: QBit, /) -> None:
    """Swap two initialized scalar operands inside qfunc."""
    raise TypeError("SWAP is captured inside qfunc")


def crx(angle: float, control: QBit, target: QBit, /) -> None:
    """Apply controlled X rotation with a static angle in radians inside qfunc."""
    raise TypeError("CRX is captured inside qfunc")


def r(theta: float, phi: float, target: QBit, /) -> None:
    """Apply an equatorial rotation with static angles in radians."""
    raise TypeError("R is captured inside qfunc")


def u(theta: float, phi: float, lam: float, gamma: float, target: QBit, /) -> None:
    """Apply a single-qubit Euler rotation with an explicit global phase."""
    raise TypeError("U is captured inside qfunc")


def x(target: QBit, /) -> None:
    """Flip an initialized scalar qubit inside a captured qfunc."""
    raise TypeError("X is captured inside qfunc")


def h(target: QBit, /) -> None:
    """Apply Hadamard to an initialized scalar qubit or array element."""
    raise TypeError("H is captured inside qfunc")


def s(target: QBit, /) -> None:
    """Apply diag(1, i) to an initialized scalar qubit or array element."""
    raise TypeError("S is captured inside qfunc")


def rx(angle: float, target: QBit, /) -> None:
    """Apply exp(-i*angle*X/2) in radians inside qfunc."""
    raise TypeError("RX is captured inside qfunc")


def rz(angle: float, target: QBit, /) -> None:
    """Apply diag(exp(-i*angle/2), exp(i*angle/2)) in radians."""
    raise TypeError("RZ is captured inside qfunc")


def phase_gate(angle: float, target: QBit, /) -> None:
    """Apply diag(1, exp(i*angle)) in radians inside qfunc."""
    raise TypeError("PHASE is captured inside qfunc")


def rzz(angle: float, target: QNum | QArray[QBit], /) -> None:
    """Apply exp(-i*angle*Z*Z/2) to an initialized two-bit register."""
    raise TypeError("RZZ is captured inside qfunc")


def prepare_bell_state(state: int, target: QNum | QArray[QBit], /) -> None:
    """Initialize a fresh two-bit register as Bell state 1 (Phi-) or 2 (Psi+)."""
    raise TypeError("Bell preparation is captured inside qfunc")


def prepare_ghz_state(size: int, target: QNum | QArray[QBit], /) -> None:
    """Initialize a fresh register as (|0...0> + |1...1>)/sqrt(2)."""
    raise TypeError("GHZ preparation is captured inside qfunc")


def prepare_amplitudes(
    amplitudes: Sequence[float], bound: float, target: QNum | QArray[QBit], /
) -> None:
    """Initialize fresh storage from a normalized real literal amplitude vector."""
    raise TypeError("amplitude preparation is captured inside qfunc")


def prepare_state(
    probabilities: Sequence[float], bound: float, target: QNum | QArray[QBit], /
) -> None:
    """Initialize fresh storage with nonnegative square-root amplitudes."""
    raise TypeError("probability preparation is captured inside qfunc")


def prepare_uniform_interval_state(
    start: int, stop: int, target: QNum | QArray[QBit], /
) -> None:
    """Prepare a uniform superposition over a half-open integer interval."""
    raise TypeError("uniform preparation is captured inside qfunc")


def control(condition: QBit, body: Callable[[], None], /) -> None:
    """Apply a supported scalar operation coherently when condition is one."""
    raise TypeError("control is captured inside qfunc")


def within() -> AbstractContextManager[None]:
    """Capture local arithmetic preparation for the following apply block."""
    raise TypeError("within is captured inside qfunc")


def apply() -> AbstractContextManager[None]:
    """Capture an operation between preparation and its checked inverse."""
    raise TypeError("apply is captured inside qfunc")


def adjoint() -> AbstractContextManager[None]:
    """Capture the checked inverse of static Const-only qfunc calls."""
    raise TypeError("adjoint is captured inside qfunc")
