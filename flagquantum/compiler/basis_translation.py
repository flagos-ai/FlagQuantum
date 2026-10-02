"""Rewrite named opcodes into a target's native gate set through exact identities.

A named gate has no matrix on this layer: the matrix of `h` or `rzz` belongs to
`flagquantum.simulation`, which the Compiler layer must not import. So the only
way a named gate can leave a program for a basis that does not carry it is a
closed table of identities that a reader can check by hand and a test can pin
against the runtime. This module owns that table and the deterministic search
that composes it, which is the Compiler-layer counterpart of the equivalence
library and basis search Qiskit ships.

Every rule is an exact closed form. `cx` is `cz` conjugated by one `h`, `rzz` is
`cx` around a `rz`, and the rest follow from those. Seventeen of the eighteen
entries reproduce their source exactly; `cphase` is the exception, because
turning a controlled rotation into a controlled phase needs a third `rz` on the
control wire and that rotation leaves a global phase behind. FlagQuantum IR has
no field for a global phase, so a `cphase` translation is equal to its source
only up to that one phase -- the same contract the one-qubit and two-qubit
syntheses publish, and the reason `cphase` is the one entry a test pins as
inexact rather than allowing either answer.

The table spans both arities the IR declares more than one opcode at. Its two
three-wire entries cover `ccx` and `cswap`, the only arity-3 unitary opcodes
there are. That is the whole of the multi-controlled family here: Qiskit splits
it into a Gray-code, a recursive and a V-chain construction, but the first of
those returns `CCXGate` itself at two controls and the other two exist to trade
ancillas for fewer entanglers at five controls and up, which this IR cannot
express. `benchmarks/compiler_basis_translation.py` measures the Gray-code form
against the one taken here and records why it was not taken.

No rule introduces an angle the caller did not write. That is a narrower contract
than an equivalence library usually carries, and it has a measured cost: every
entangling rule in the table routes down to `cx` or `cz`, so a target publishing
only a parametrized entangler such as `rzz` is reached by the rules that already
name that rotation and not by the rest. Qiskit's standard library bottoms out in
a rotation like that by introducing `pi/2` and friends; this table may not, and
`benchmarks/compiler_basis_translation.py` records the difference per basis
rather than smoothing it over. Both facts are deliberate.

The number of entries is not the reach. Most of them do not name their own leaf
set: `cry` becomes a `crx`, which becomes a `crz`, which becomes two `cx` around
two `rz`, and `cswap` becomes a `ccx` around two `cx`. The search below expands
each rule's leaves through the table again, so a target that publishes `cx` and a
z-rotation still reaches `cry` through three levels, and the table stores the
shortest statement of each identity rather than its closure.

The search branches only on opcode names, never on a parameter value, so a
trainable angle cannot change the chosen path, and every rule forwards the
source instruction's own parameter objects and metadata instead of rebuilding
them from constants, so a trainable rotation stays in the autograd graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..core.ir import Instruction
from ..core.operator_schema import get_operator_schema
from .one_qubit_synthesis import synthesize_one_qubit

_HALF = 0.5


@dataclass(frozen=True)
class EquivalenceRule:
    """One exact rewrite of a named opcode into a fixed instruction sequence.

    `build` receives the concrete source instruction, so the replacement carries
    that instruction's own parameters and metadata rather than copies of them.
    """

    opcode: str
    build: Callable[[Instruction], tuple[Instruction, ...]]


def _gate(
    opcode: str,
    wires: tuple[int, ...],
    metadata: Mapping[str, Any],
    angle: Any = None,
) -> Instruction:
    params = {} if angle is None else {"theta": angle}
    return Instruction(opcode, wires, params=params, metadata=metadata)


def _cx_to_cz(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CX = (I x H) CZ (I x H)`, conjugating the target wire."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("h", (right,), metadata),
        _gate("cz", (left, right), metadata),
        _gate("h", (right,), metadata),
    )


def _cz_to_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CZ = (I x H) CX (I x H)`, conjugating the target wire."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("h", (right,), metadata),
        _gate("cx", (left, right), metadata),
        _gate("h", (right,), metadata),
    )


def _cy_to_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CY = Sdg(target) CX S(target)`."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("sdg", (right,), metadata),
        _gate("cx", (left, right), metadata),
        _gate("s", (right,), metadata),
    )


def _swap_to_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """The middle `cx` is reversed, so the three compose to a wire swap."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("cx", (left, right), metadata),
        _gate("cx", (right, left), metadata),
        _gate("cx", (left, right), metadata),
    )


def _rzz_to_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`RZZ(theta) = CX RZ(theta) CX`, rotating the target wire."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("cx", (left, right), metadata),
        _gate("rz", (right,), metadata, instruction.params["theta"]),
        _gate("cx", (left, right), metadata),
    )


def _crz_to_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CRZ(theta) = RZ(theta/2) CX RZ(-theta/2) CX` on the target wire."""

    left, right = instruction.wires
    metadata = instruction.metadata
    half = instruction.params["theta"] * _HALF
    return (
        _gate("rz", (right,), metadata, half),
        _gate("cx", (left, right), metadata),
        _gate("rz", (right,), metadata, -half),
        _gate("cx", (left, right), metadata),
    )


def _cphase_to_cx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CPHASE(theta)` is the `CRZ` form plus `RZ(theta/2)` on the control."""

    left, right = instruction.wires
    metadata = instruction.metadata
    half = instruction.params["theta"] * _HALF
    return (
        _gate("rz", (right,), metadata, half),
        _gate("cx", (left, right), metadata),
        _gate("rz", (right,), metadata, -half),
        _gate("cx", (left, right), metadata),
        _gate("rz", (left,), metadata, half),
    )


def _rxx_to_rzz(instruction: Instruction) -> tuple[Instruction, ...]:
    """`RXX(theta) = (H x H) RZZ(theta) (H x H)`."""

    left, right = instruction.wires
    metadata = instruction.metadata
    theta = instruction.params["theta"]
    return (
        _gate("h", (left,), metadata),
        _gate("h", (right,), metadata),
        _gate("rzz", (left, right), metadata, theta),
        _gate("h", (left,), metadata),
        _gate("h", (right,), metadata),
    )


def _ryy_to_rxx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`RYY(theta) = (S x S) RXX(theta) (Sdg x Sdg)`."""

    left, right = instruction.wires
    metadata = instruction.metadata
    theta = instruction.params["theta"]
    return (
        _gate("s", (left,), metadata),
        _gate("s", (right,), metadata),
        _gate("rxx", (left, right), metadata, theta),
        _gate("sdg", (left,), metadata),
        _gate("sdg", (right,), metadata),
    )


def _crx_to_crz(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CRX(theta) = (I x H) CRZ(theta) (I x H)`."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("h", (right,), metadata),
        _gate("crz", (left, right), metadata, instruction.params["theta"]),
        _gate("h", (right,), metadata),
    )


def _cry_to_crx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CRY(theta) = (I x Sdg) CRX(theta) (I x S)`."""

    left, right = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("sdg", (right,), metadata),
        _gate("crx", (left, right), metadata, instruction.params["theta"]),
        _gate("s", (right,), metadata),
    )


def _x_to_hzh(instruction: Instruction) -> tuple[Instruction, ...]:
    return (
        _gate("h", instruction.wires, instruction.metadata),
        _gate("z", instruction.wires, instruction.metadata),
        _gate("h", instruction.wires, instruction.metadata),
    )


def _rx_to_hrzh(instruction: Instruction) -> tuple[Instruction, ...]:
    """`RX(theta) = H RZ(theta) H`, exactly, not only up to a phase."""

    metadata = instruction.metadata
    return (
        _gate("h", instruction.wires, metadata),
        _gate("rz", instruction.wires, metadata, instruction.params["theta"]),
        _gate("h", instruction.wires, metadata),
    )


def _ry_to_shzhs(instruction: Instruction) -> tuple[Instruction, ...]:
    """`RY(theta) = Sdg H RZ(theta) H S`, exactly, not only up to a phase."""

    metadata = instruction.metadata
    return (
        _gate("sdg", instruction.wires, metadata),
        _gate("h", instruction.wires, metadata),
        _gate("rz", instruction.wires, metadata, instruction.params["theta"]),
        _gate("h", instruction.wires, metadata),
        _gate("s", instruction.wires, metadata),
    )


def _sdg_to_sss(instruction: Instruction) -> tuple[Instruction, ...]:
    """`Sdg = S S S`, the only way a `t`-only basis can express a `sdg`.

    Three rules above emit `sdg`, and a Clifford+T target that publishes `s` and
    `t` but not `sdg` is a real shape. Without this entry those rules stop one
    gate short of a basis that can express them, which is a table gap rather
    than a search limitation -- Qiskit's standard equivalence library carries the
    same three-`s` expansion of `SdgGate`.
    """

    metadata = instruction.metadata
    wires = instruction.wires
    return (
        _gate("s", wires, metadata),
        _gate("s", wires, metadata),
        _gate("s", wires, metadata),
    )


def _z_to_ss(instruction: Instruction) -> tuple[Instruction, ...]:
    """`Z = S S`, which is what carries `x` into a `t`-only basis.

    `_x_to_hzh` names a `z`, and a Clifford+T target publishes `s` and `t` but
    not `z`. Without this entry `x` is the one name such a target cannot carry
    that Qiskit's standard equivalence library can, because the library spells
    `ZGate` the same way.
    """

    metadata = instruction.metadata
    wires = instruction.wires
    return (
        _gate("s", wires, metadata),
        _gate("s", wires, metadata),
    )


def _ccx_to_t_form(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CCX` as the fifteen-gate `H`/`T`/`Tdg`/`CX` identity, exactly.

    This is the definition Qiskit's `MCXGrayCode` reaches at two controls rather
    than a Gray-code derivation of its own: `MCXGrayCode.__new__` returns
    `CCXGate` for one to four controls and only builds a Gray-code chain from five
    up, so at the one arity FlagQuantum IR declares -- `ccx` and `cswap` are its
    only arity-3 unitary opcodes -- the Gray code, the recursive and the V-chain
    construction are the same circuit. The measured alternative is in
    `benchmarks/compiler_basis_translation.py`: a Gray-code form over three
    `cphase` plus two `cx` stores seven leaves instead of fifteen and is equally
    exact, but costs eight two-qubit gates against six here, and its further
    expansion depends on the one rule in this table that is not exact.

    The identity is exact -- the T-count form of Barenco et al. -- so unlike
    `cphase` it leaves no global phase behind. The leaves it names carry no angle
    of their own; the fixed `pi/4` a `t` needs appears only when a target's
    one-qubit synthesis writes it, which is the target's accounting and not this
    rule's.
    """

    control0, control1, target = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("h", (target,), metadata),
        _gate("cx", (control1, target), metadata),
        _gate("tdg", (target,), metadata),
        _gate("cx", (control0, target), metadata),
        _gate("t", (target,), metadata),
        _gate("cx", (control1, target), metadata),
        _gate("tdg", (target,), metadata),
        _gate("cx", (control0, target), metadata),
        _gate("t", (control1,), metadata),
        _gate("t", (target,), metadata),
        _gate("h", (target,), metadata),
        _gate("cx", (control0, control1), metadata),
        _gate("t", (control0,), metadata),
        _gate("tdg", (control1,), metadata),
        _gate("cx", (control0, control1), metadata),
    )


def _cswap_to_cx_ccx(instruction: Instruction) -> tuple[Instruction, ...]:
    """`CSWAP` as `CX CCX CX`, all three conjugating the same target pair.

    The statement, not the closure: the `ccx` leaf is expanded by the rule above
    through the same search, so a target that can carry a Toffoli carries a
    Fredkin without this entry repeating fifteen gates. Qiskit's `CSwapGate`
    declares the same three-operand form on the same operand order.
    """

    control, target0, target1 = instruction.wires
    metadata = instruction.metadata
    return (
        _gate("cx", (target1, target0), metadata),
        _gate("ccx", (control, target0, target1), metadata),
        _gate("cx", (target1, target0), metadata),
    )


EQUIVALENCE_RULES: Mapping[str, tuple[EquivalenceRule, ...]] = MappingProxyType(
    {
        "cx": (EquivalenceRule("cx", _cx_to_cz),),
        "cz": (EquivalenceRule("cz", _cz_to_cx),),
        "cy": (EquivalenceRule("cy", _cy_to_cx),),
        "swap": (EquivalenceRule("swap", _swap_to_cx),),
        "rzz": (EquivalenceRule("rzz", _rzz_to_cx),),
        "crz": (EquivalenceRule("crz", _crz_to_cx),),
        "cphase": (EquivalenceRule("cphase", _cphase_to_cx),),
        "rxx": (EquivalenceRule("rxx", _rxx_to_rzz),),
        "ryy": (EquivalenceRule("ryy", _ryy_to_rxx),),
        "crx": (EquivalenceRule("crx", _crx_to_crz),),
        "cry": (EquivalenceRule("cry", _cry_to_crx),),
        "x": (EquivalenceRule("x", _x_to_hzh),),
        "z": (EquivalenceRule("z", _z_to_ss),),
        "rx": (EquivalenceRule("rx", _rx_to_hrzh),),
        "ry": (EquivalenceRule("ry", _ry_to_shzhs),),
        "sdg": (EquivalenceRule("sdg", _sdg_to_sss),),
        "ccx": (EquivalenceRule("ccx", _ccx_to_t_form),),
        "cswap": (EquivalenceRule("cswap", _cswap_to_cx_ccx),),
    }
)


@dataclass(frozen=True)
class _Translation:
    """A fully expanded rewrite, with the leaves it could not resolve."""

    instructions: tuple[Instruction, ...]
    unresolved: tuple[str, ...]
    rewritten: bool

    @property
    def rank(self) -> tuple[int, int]:
        """Resolved rewrites first, then the shortest of them.

        A rewrite that still names a gate the target does not publish sorts last
        whatever its length, because leaving the native set is a failure the
        caller has to see, not a shorter answer. `min` keeps the first of equal
        ranks, and the rules are tried in a fixed order, so a tie is broken by
        the table rather than by a mapping's iteration order.
        """

        return (1 if self.unresolved else 0, len(self.instructions))


def _expand(
    instruction: Instruction,
    *,
    can_run: Callable[[Instruction], bool],
    z_rotation: str | None,
    pulse_opcode: str | None,
    active: frozenset[str],
) -> _Translation | None:
    """Return the best rewrite of one instruction, or None when it has none.

    None means nothing can be said about the opcode at all -- it is unknown to
    the operator schema, it is not unitary, or the search re-entered it while
    expanding itself. A translation whose `rewritten` flag is False is the
    fallback for an opcode the table does not reach; the caller turns that into
    the same refusal it gave before the table existed.
    """

    if can_run(instruction):
        return _Translation((instruction,), (), False)
    opcode = instruction.name
    schema = get_operator_schema(opcode)
    if schema is None or not schema.unitary:
        return None
    if opcode in active:
        return None

    options: list[_Translation] = []
    if z_rotation is not None and pulse_opcode is not None:
        synthesized = synthesize_one_qubit(
            instruction, z_rotation=z_rotation, pulse_opcode=pulse_opcode
        )
        if synthesized is not None:
            options.append(
                _Translation(
                    synthesized,
                    tuple(item.name for item in synthesized if not can_run(item)),
                    True,
                )
            )
    nested_active = active | {opcode}
    for rule in EQUIVALENCE_RULES.get(opcode, ()):
        expanded: list[_Translation] = []
        for item in rule.build(instruction):
            child = _expand(
                item,
                can_run=can_run,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                active=nested_active,
            )
            if child is None:
                expanded = []
                break
            expanded.append(child)
        if not expanded:
            continue
        options.append(
            _Translation(
                instructions=tuple(
                    leaf for item in expanded for leaf in item.instructions
                ),
                unresolved=tuple(name for item in expanded for name in item.unresolved),
                rewritten=True,
            )
        )
    if not options:
        return _Translation((instruction,), (opcode,), False)
    return min(options, key=lambda item: item.rank)


def translate(
    instruction: Instruction,
    *,
    can_run: Callable[[Instruction], bool],
    z_rotation: str | None = None,
    pulse_opcode: str | None = None,
) -> tuple[Instruction, ...] | None:
    """Return `instruction` as the shortest rewrite whose leaves can all run.

    `can_run` decides whether one instruction is expressible by the target, so a
    caller that checks a native descriptor's declared parameters keeps that
    precision here. `z_rotation` and `pulse_opcode` are the basis the Euler
    synthesis needs; when either is None only the equivalence table is used.

    Returns None when the table and the synthesis both have nothing to say about
    the opcode, so the caller can report the opcode itself rather than a
    replacement that was never verified. When a rewrite exists but no fully
    runnable one does, the shortest rewrite is returned anyway -- the caller
    re-checks every leaf and names the gate the target is missing.
    """

    result = _expand(
        instruction,
        can_run=can_run,
        z_rotation=z_rotation,
        pulse_opcode=pulse_opcode,
        active=frozenset(),
    )
    if result is None or not result.rewritten:
        return None
    return result.instructions


__all__ = ("EQUIVALENCE_RULES", "EquivalenceRule", "translate")
