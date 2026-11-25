# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/lp_resource_state.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Compose Luis–Perina phase windows from shared rotations and reflection."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import acos, pi

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler._component_trace.gate_budget import legacy_gate_budget
from ...compiler._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ...compiler._component_trace.resource_generalization import (
    CliffordCall,
    PredicateControlledCall,
)
from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.bits.bitwise import on_each_ir
from ...compiler.registers.bits.on_each import on_each_word_program
from ...compiler.registers.component_adjoint import adjoint_component_hierarchy
from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_definition_match import (
    matches_component_definition,
)
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.component_graph import _namespace_helpers
from ...compiler.registers.component_rebindings import adjoint_word_rebinding
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_requirements,
    source_symbolic_zero_reflection_resource_formula,
)
from ...compiler.registers.primitives.controlled_rotation import controlled_rz_ir
from ...compiler.registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
)
from ...core._composition import inverted_instruction
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal, ceil, log2
from ...core.value_math import acos as symbolic_acos
from ...core.value_math import pi as symbolic_pi
from ._block_binding import _BoundBlockOperation, _validate_block_work
from .black_box_prepare import BlackBoxPrepare, PackedPrepareDeclaration
from .phase_estimation_window import PhaseEstimationWindow
from .prepared_state_reflection import PreparedStateReflection


def luis_perina_interim_program(width: int, *, adjoint: bool = False) -> ProgramRecord:
    """Retain the source's arbitrary-input m/anc graph and exact stored inverse.

    m is MSB-first caller storage. anc is a logical through output, with no
    private work or measurement. Shared Rz, CRz and word adjoints own their
    unchanged definitions and absolute phase.
    """
    if type(width) is not int or width < 1 or type(adjoint) is not bool:
        raise ValueError(
            "Interim Program requires a positive width and boolean adjoint"
        )
    bit = Register("q", quantum_bit_type())
    h = bit_gate_function("hadamard", (bit,), CircuitIR(1, (Instruction("h", (0,)),)))
    angle = -2 * pi / (2**width + 1)
    rz = bit_gate_function(
        "interim_rz",
        (bit,),
        CircuitIR(1, (Instruction("rz", (0,), params={"theta": angle}),)),
    )
    builder = ComponentProgramBuilder(LuisPerinaInterim(width).signature, (h, rz))
    m = builder.call_program(
        on_each_word_program(width, h), {"q": builder.inputs["m"]}
    )["q"]
    assert isinstance(m, RegisterBinding)
    bits = builder.split(m)
    anc = builder.call_dict(h.name, {"q": builder.inputs["anc"]})["q"]
    for i in range(width):
        values = builder.call_program(
            controlled_rz_program(-2 * pi * 2**i / (2**width + 1)),
            {"ctrl": bits[-i - 1], "q": anc},
        )
        bits[-i - 1], anc = values["ctrl"], values["q"]
    anc = builder.call_dict(rz.name, {"q": anc})["q"]
    anc = builder.call_dict(h.name, {"q": anc})["q"]
    record = builder.finalize({"m": builder.join(bits), "anc": anc})
    if not adjoint:
        return record
    inverse_rz = bit_gate_function(
        "interim_inverse_rz",
        (bit,),
        CircuitIR(1, (Instruction("rz", (0,), params={"theta": -angle}),)),
    )
    definitions = [*record.functions, inverse_rz]
    adjoints = {rz.name: inverse_rz.name}
    for position, function in enumerate(record.functions[1:]):
        ports = function.registers or ()
        if matches_component_definition(record, function.name, ProgramRecord((h,))):
            adjoints[function.name] = function.name
        elif tuple(port.name for port in ports) == ("reg", "reg"):
            inverse = adjoint_word_rebinding(
                function, f"interim_inverse_view_{position}"
            )
            definitions.append(inverse)
            adjoints[function.name] = inverse.name
        elif tuple(port.name for port in ports) == ("ctrl", "q"):
            for i in range(width):
                theta = -2 * pi * 2**i / (2**width + 1)
                if matches_component_definition(
                    record, function.name, controlled_rz_program(theta)
                ):
                    inverse_record = _namespace_helpers(
                        controlled_rz_program(theta, adjoint=True),
                        f"interim_inverse_crz_{position}",
                    )
                    definitions.extend(inverse_record.functions)
                    adjoints[function.name] = inverse_record.functions[0].name
                    break
    return adjoint_component_hierarchy(
        replace(record, functions=tuple(definitions)), adjoints
    )


@dataclass(frozen=True)
class LuisPerinaInterim:
    """Prepare the source's joint phase word and sine/cosine ancilla state.

    The word uses big-endian order. Both registers are THRU and accept arbitrary
    coherent inputs. No clean scratch, classical result slots or measurements
    are required. The caller owns the ancilla's logical output.
    """

    bitsize: SymbolicInt
    eps: float = 1e-11

    @property
    def signature(self) -> Signature:
        return Signature.build(m=self.bitsize, anc=1)

    def make_on(
        self, *, m: tuple[int, ...], anc: int, adjoint: bool = False
    ) -> _BoundBlockOperation:
        """Bind the complete original H/controlled-Rz/H stream for append."""
        if type(self.bitsize) is not int or self.bitsize < 1:
            raise ValueError("Luis–Perina interim requires a concrete positive width")
        if len(m) != self.bitsize or type(adjoint) is not bool:
            raise ValueError("Interim binding needs its declared width and direction")
        _validate_block_work((*m, anc), (), ())
        gates = [*on_each_ir(Instruction("h", (0,)), m), Instruction("h", (anc,))]
        for i, control in enumerate(reversed(m)):
            angle = -2 * pi * 2**i / (2**self.bitsize + 1)
            gates.extend(controlled_rz_ir(control, anc, angle))
        gates.extend(
            (
                Instruction(
                    "rz", (anc,), params={"theta": -2 * pi / (2**self.bitsize + 1)}
                ),
                Instruction("h", (anc,)),
            )
        )
        stream = tuple(gates)
        if adjoint:
            stream = tuple(inverted_instruction(gate) for gate in reversed(stream))
        return _BoundBlockOperation(stream, max((*m, anc)) + 1)

    def source_gate_cost(self) -> QECGateCost:
        """Query the source's independent QEC categories before decomposition."""
        return QECGateCost(clifford=3 * self.bitsize + 2, rotation=2 * self.bitsize + 1)

    def source_graph(self) -> SymbolicCallGraph[tuple[str, SymbolicReal]]:
        """Retain angle-specific declared calls independently of primitive counts."""
        n = self.bitsize
        angle = -2 * symbolic_pi(n) / (2**n + 1)
        calls: dict[tuple[str, SymbolicReal], SymbolicInt] = {
            ("Rz", angle): 1,
            ("OnEach(H)", n): 1,
            ("H", 0): 2,
        }
        if isinstance(n, int):
            calls.update({("CRz", angle * 2**i): 1 for i in range(n)})
        else:
            calls[("CRz", angle)] = n
        return symbolic_call_graph(
            ("LuisPerinaInterim", n),
            {
                ("LuisPerinaInterim", n): calls,
                ("OnEach(H)", n): {("H", 0): n},
            },
        )


@dataclass(frozen=True)
class LuisPerinaState(PhaseEstimationWindow):
    """Prepare the optimal sine window with one source amplitude-amplification step.

    Outputs are RIGHT-only fractional phase wires. The explicit workspace starts
    and ends clean in every measured reflection branch. Its first two wires are
    the source ancilla and flag. Remaining scratch belongs to the shared zero
    reflection and its result slots.
    """

    bitsize: SymbolicInt

    @property
    def m_bits(self) -> SymbolicInt:
        return self.bitsize

    @classmethod
    def from_standard_deviation_eps(cls, eps: SymbolicReal) -> "LuisPerinaState":
        return cls(ceil(log2(symbolic_pi(eps) / eps)))

    def requirements(self) -> tuple[int, int]:
        """Return clean physical workspace and reusable measurement result slots."""
        if type(self.bitsize) is not int or self.bitsize < 1:
            raise ValueError("Luis–Perina window requires a concrete positive width")
        work, slots = prepare_reflection_requirements(
            self.bitsize + 2, controlled=False
        )
        return work + 2, slots

    def bind(self) -> BlackBoxPrepare:
        """Bind named PREPARE for circuit.append(op) with explicit owned storage."""
        self.requirements()
        return PackedPrepareDeclaration(self).bind(self._emit)

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        if (len(workspace), len(classical_bits)) != self.requirements():
            raise ValueError(
                "Luis–Perina preparation needs its declared work and slots"
            )
        m = ports["qpe_reg"]
        _validate_block_work(m, workspace, classical_bits)
        anc, flag = workspace[:2]
        interim = LuisPerinaInterim(len(m))
        forward = interim.make_on(m=m, anc=anc).instructions
        inverse = interim.make_on(m=m, anc=anc, adjoint=True).instructions
        angle = acos(1 / (1 + 2 ** len(m)))
        rotate = (Instruction("ry", (flag,), params={"theta": angle}),)
        unrotate = (Instruction("ry", (flag,), params={"theta": -angle}),)
        reflection = (
            PreparedStateReflection.around_zero((1, 1, len(m)), global_phase=1j)
            .make_on(
                selection={"reg0_": (flag,), "reg1_": (anc,), "reg2_": m},
                reflection_workspace=workspace[2:],
                reflection_classical_bits=classical_bits,
                adjoint=adjoint,
            )
            .instructions
        )
        flip = (Instruction("x", (flag,)), Instruction("x", (anc,)))
        mark = (Instruction("cz", (flag, anc)),)
        if adjoint:
            return (
                *flip,
                *unrotate,
                *inverse,
                *reflection,
                *rotate,
                *forward,
                *mark,
                *inverse,
                *unrotate,
            )
        return (
            *rotate,
            *forward,
            *mark,
            *inverse,
            *unrotate,
            *reflection,
            *forward,
            *rotate,
            *flip,
        )

    def source_graph(self) -> SymbolicCallGraph[tuple[str, SymbolicReal]]:
        """Preserve all seven original direct callees and their multiplicities."""
        n = self.bitsize
        angle = symbolic_acos(1 / (1 + 2**n))
        return symbolic_call_graph(
            ("LuisPerinaState", n),
            {
                ("LuisPerinaState", n): {
                    ("LuisPerinaInterim", n): 2,
                    ("LuisPerinaInterim†", n): 1,
                    ("Ry", angle): 2,
                    ("Ry", -angle): 1,
                    ("ReflectionZero(i)", n + 2): 1,
                    ("X", 0): 2,
                    ("CZ", 0): 1,
                },
            },
        )

    def source_gate_cost(self) -> QECGateCost:
        """Reuse shared reflection categories, retaining symbolic source differences."""
        n = self.bitsize
        if isinstance(n, int):
            identity = ComponentCost("IdentityPrepare", n + 2)
            reflection = query_qec_gate_cost(
                PreparedStateReflection.around_zero(
                    (1, 1, n), global_phase=1j
                ).resource_model(identity, identity)
            )
        else:
            symbolic = source_symbolic_zero_reflection_resource_formula(n + 2)
            reflection = QECGateCost().plus(symbolic, 1)
        return (
            LuisPerinaInterim(n).source_gate_cost() * 3
            + reflection
            + QECGateCost(rotation=3, clifford=3)
        )

    def source_t_complexity(self) -> dict[str, SymbolicInt]:
        """Query the original historical T, rotation and Clifford model.

        Keep the reflection predicate's compute/erase calls until the shared
        historical query has priced them. Folding already aggregated modern
        QEC categories loses the source's different erasure Clifford charge.
        This declaration does not count emitted primitives or synthesize angles.
        Resolved widths must be positive; symbolic widths retain the source model.
        """
        n = self.bitsize
        if isinstance(n, int) and (type(n) is not int or n < 1):
            raise ValueError("Luis–Perina cost requires a positive width")
        width = n + 2
        controls = (0,) * width if isinstance(width, int) else width
        mark = query_qec_gate_cost(
            PredicateControlledCall(CliffordCall("cz", 2), controls),
            legacy_shims=True,
        )
        # Two reflection carrier X gates, then final CZ/X/X, plus three Ry.
        cost = (
            LuisPerinaInterim(n).source_gate_cost() * 3
            + mark
            + QECGateCost(rotation=3, clifford=5)
        )
        return legacy_gate_budget(cost)
