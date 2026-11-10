"""Annotation vocabulary and captured operations for bounded Python programs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from types import GenericAlias
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar, overload

from ...observables import Observable

_Num = TypeVar("_Num", bound="_QuantumRegister")
_Element = TypeVar("_Element", bound="QNum")


class _QuantumRegister:
    """Shared annotation base for owned or borrowed quantum storage."""


def _numeric_spec(spec: object) -> tuple[int, bool, int]:
    args = spec if isinstance(spec, tuple) else (spec,)
    if len(args) not in {1, 3}:
        raise TypeError("QNum expects width or (width, signed, fractional_bits)")
    width = args[0]
    if type(width) is not int or not isinstance(width, int) or not 1 <= width <= 256:
        raise ValueError("QNum width must be a plain integer in 1..256")
    signed, fractional = (False, 0) if len(args) == 1 else args[1:]
    if type(signed) is not bool or not isinstance(signed, bool):
        raise ValueError("QNum signedness must be a literal Boolean")
    if (
        type(fractional) is not int
        or not isinstance(fractional, int)
        or not 0 <= fractional <= width
    ):
        raise ValueError("QNum fractional bits must be an integer in 0..width")
    return width, signed, fractional


def _width_spec(spec: object) -> int:
    return _numeric_spec(spec)[0]


class QNum(_QuantumRegister):
    """Quantum binary-scaled number annotation, not a host numeric value.

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

    def __add__(self, other: QNum | float) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __radd__ = __add__

    def __mul__(self, other: QNum | float) -> QNum:
        raise TypeError("quantum expressions are captured inside qfunc, not executed")

    __rmul__ = __mul__

    def __sub__(self, other: QNum | float) -> QNum:
        raise TypeError("numeric subtraction is captured inside qfunc")

    def __rsub__(self, other: float) -> QNum:
        raise TypeError("numeric subtraction is captured inside qfunc")

    def __neg__(self) -> QNum:
        raise TypeError("numeric negation is captured inside qfunc")

    def __truediv__(self, divisor: float) -> QNum:
        raise TypeError("real division is captured inside phase expressions only")

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

    def __lt__(self, other: QNum | float) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __le__(self, other: QNum | float) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __gt__(self, other: QNum | float) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __ge__(self, other: QNum | float) -> QBit:
        raise TypeError("quantum comparisons are captured inside qfunc, not executed")

    def __ixor__(self: _Num, mask: QNum | float) -> _Num:
        raise TypeError("raw word XOR is captured inside qfunc")

    def __ior__(self: _Num, expression: QNum | float) -> _Num:
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


def drop(target: QNum | QArray[QBit], /) -> None:
    """End access to owned hidden storage while retaining its entangled environment.

    This does not reset, uncompute or reuse the physical qubits. Hidden storage
    must be dropped exactly once at the unconditional qfunc root before return.
    """
    raise TypeError("drop is captured inside qfunc; synthesize the program")


def qpe(callback: Callable[[], None], phase_result: QNum, /) -> None:
    """Estimate a pure quantum callback's eigenphase into a fresh counter.

    Use one positional captured operation or qfunc call in a zero-argument
    lambda, at the root or in root pure unitary regions. The allocated unused
    counter has one to eight signed or unsigned all-fractional bits. Callback
    storage must be initialized, writable and disjoint from the counter;
    allocation, publication and unsupported nested controls are rejected.
    """
    raise TypeError("qpe is captured inside qfunc; synthesize the program")


def select_rotation(
    axis: Literal["X", "Y", "Z"],
    angles: Sequence[float],
    selector: QNum | QArray[QBit],
    target: QBit,
    /,
) -> None:
    """Apply the raw selector word's static Pauli rotation to one disjoint qubit.

    Tables contain 2**N angles for one to eight selector bits. Numeric sign and
    fractional metadata do not reinterpret the selector's stored bit pattern.
    """
    raise TypeError("select_rotation is captured inside qfunc")


def hadamard_transform(out: QNum | QArray[QBit], /) -> None:
    """Prepare uniform amplitudes on an initialized captured quantum port.

    The compiler applies Hadamard to each logical bit. This operation cannot
    run in the host interpreter; use synthesize and fq.run.
    """
    raise TypeError("hadamard_transform is captured inside qfunc")


def unitary(
    matrix: Sequence[Sequence[complex | float | int]],
    target: QNum | QArray[QBit] | Sequence[QNum | QArray[QBit]],
    /,
) -> None:
    """Apply a bounded static matrix to writable quantum storage inside qfunc.

    Whole ports and flat explicit storage sequences use logical order: position zero
    is the least significant matrix bit. Matrices must be unitary.
    """
    raise TypeError("unitary is captured inside qfunc")


def qft(target: QNum | QArray[QBit], /) -> None:
    """Apply the positive-sign Fourier transform to a complete writable register.

    Element zero is the least significant bit; the transform includes final bit
    reversal. Capture this call inside qfunc and execute the synthesized program.
    """
    raise TypeError("qft is captured inside qfunc")


def qct_type2(target: QNum | QArray[QBit], /) -> None:
    """Apply the normalized cosine Type-II transform to a complete register."""
    raise TypeError("qct_type2 is captured inside qfunc")


def qst_type2(target: QNum | QArray[QBit], /) -> None:
    """Apply the normalized sine Type-II transform to a complete register."""
    raise TypeError("qst_type2 is captured inside qfunc")


def qct_qst_type2(target: QNum | QArray[QBit], selector: QBit, /) -> None:
    """Apply cosine Type II for selector zero and negative sine Type II for one."""
    raise TypeError("qct_qst_type2 is captured inside qfunc")


def qct_qst_type1(target: QNum | QArray[QBit], /) -> None:
    """Apply the Type-I transform, including its positive-i sine-sector phase."""
    raise TypeError("qct_qst_type1 is captured inside qfunc")


def suzuki_trotter(
    operator: Observable,
    time: float,
    order: int,
    repetitions: int,
    target: QNum | QArray[QBit] | Sequence[QBit],
    /,
) -> None:
    """Apply an ordered first/second-order Pauli product formula inside qfunc.

    Observable indices address target elements: index zero is the least
    significant bit. Bind a static Observable; time may be a classical real
    parameter. The formula preserves identity phase and declared term order.
    """
    raise TypeError("suzuki_trotter is captured inside qfunc")


def qdrift(
    operator: Observable,
    time: float,
    samples: int,
    target: QNum | QArray[QBit] | Sequence[QBit],
    seed: int = 0,
    /,
) -> None:
    """Capture a reproducible sampled Pauli product, including identity phase.

    Use 1..2048 samples and an optional classical seed in 0..2**32-1.
    Observable indices address logical target elements, with index zero least
    significant. The sampled circuit approximates Hamiltonian evolution.

    Examples:
        >>> import flagquantum as fq
        >>> from flagquantum import synth
        >>> operator = 0.7 * fq.X(0) + 0.3 * fq.Z(0)
        >>> @synth.qfunc
        ... def main(q: synth.Output[synth.QBit]) -> None:
        ...     synth.allocate(q)
        ...     synth.qdrift(operator, 0.37, 32, q)
        >>> program = synth.synthesize(main)
        >>> result = fq.run(program)
    """
    raise TypeError("qdrift is captured inside qfunc")


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


def inplace_prepare_amplitudes(
    amplitudes: Sequence[float],
    bound: float,
    target: QNum | QArray[QBit] | Sequence[QNum | QArray[QBit]],
    /,
) -> None:
    """Apply the reversible real preparation unitary to initialized storage."""
    raise TypeError("inplace amplitude preparation is captured inside qfunc")


def inplace_prepare_state(
    probabilities: Sequence[float],
    bound: float,
    target: QNum | QArray[QBit] | Sequence[QNum | QArray[QBit]],
    /,
) -> None:
    """Apply square-root probability preparation without allocating or reset."""
    raise TypeError("inplace probability preparation is captured inside qfunc")


def prepare_uniform_interval_state(
    start: int, stop: int, target: QNum | QArray[QBit], /
) -> None:
    """Prepare a uniform superposition over a half-open integer interval."""
    raise TypeError("uniform preparation is captured inside qfunc")


def control(condition: QNum | QArray[QBit] | bool, body: Callable[[], None], /) -> None:
    """Apply a supported operation coherently; array-controlled X requires all ones."""
    raise TypeError("control is captured inside qfunc")


def within() -> AbstractContextManager[None]:
    """Capture arithmetic locals or unitary preparation before the apply block."""
    raise TypeError("within is captured inside qfunc")


def apply() -> AbstractContextManager[None]:
    """Capture an operation between preparation and its checked inverse."""
    raise TypeError("apply is captured inside qfunc")


def adjoint() -> AbstractContextManager[None]:
    """Capture the checked inverse of static gates and reversible qfunc calls."""
    raise TypeError("adjoint is captured inside qfunc")


def minimum(left: QNum | float, right: QNum | float, /) -> QNum:
    """Capture an exact two-operand minimum inside a native qfunc."""
    raise TypeError("numeric minimum is captured inside qfunc")


def maximum(left: QNum | float, right: QNum | float, /) -> QNum:
    """Capture an exact two-operand maximum inside a native qfunc."""
    raise TypeError("numeric maximum is captured inside qfunc")
