# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 first-quantized projectile cost declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/projectile

"""Resource models for first-quantized chemistry with a quantum projectile."""

from dataclasses import dataclass, replace
from functools import cached_property

from ..compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from ..compiler.registers.arithmetic.arithmetic_resource_model import source_add_cost
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from .first_quantization import (
    source_controlled_multiplexed_cswap_3d_cost,
    source_multiplexed_cswap_3d_cost,
)
from .first_quantization_kinetic import (
    projectile_prepare_power_two_signature,
    projectile_prepare_t_signature,
)
from .first_quantization_nu_resource_model import (
    projectile_prepare_nu_adjoint_model,
    source_projectile_prepare_nu_adjoint_gate_cost,
    source_projectile_prepare_nu_cost,
)
from .first_quantization_potential import prepare_uv_signature
from .first_quantization_projectile import (
    projectile_prepare_signature,
    projectile_prepare_tuv_signature,
    projectile_select_signature,
    projectile_select_t_signature,
    projectile_select_uv_signature,
)
from .first_quantization_resource_model import (
    _require_positive,
    _signed_conversion_cost,
    source_apply_nuclear_phase_cost,
    source_prepare_ij_cost,
    source_prepare_zeta_cost,
)
from .primitives.prepare_oracle import PrepareOracle


def source_projectile_power_two_cost(
    bitsize_n: int, bitsize_p: int, *, adjoint: bool = False
) -> ComponentCost:
    """Estimate the source unary preparation on an n-bit through register."""
    signature = projectile_prepare_power_two_signature(bitsize_n, bitsize_p)
    toffoli = bitsize_n - 2
    if not adjoint:
        toffoli += bitsize_n - bitsize_p
    return ComponentCost(
        "PreparePowerTwoStateWithProj",
        max(3, int(signature.n_qubits())),
        GateCost(toffoli=toffoli),
    )


def source_projectile_prepare_t_cost(
    num_bits_p: int, num_bits_n: int, *, adjoint: bool = False
) -> ComponentCost:
    """Price the two coordinate words and two unary words in the source graph."""
    unary = source_projectile_power_two_cost(num_bits_n, num_bits_p, adjoint=adjoint)
    # The coordinate inequalities have no adjoint Toffoli charge.
    coordinate_toffolis = 15 if adjoint else 31
    return ComponentCost(
        "PrepareTFirstQuantizationWithProj",
        int(projectile_prepare_t_signature(num_bits_p, num_bits_n).n_qubits()),
        calls=(
            CostCall(
                ComponentCost("Toffoli", 3, GateCost(toffoli=1)),
                repetitions=coordinate_toffolis,
            ),
            CostCall(unary, repetitions=2),
        ),
    )


def source_projectile_select_t_cost(num_bits_n: int) -> ComponentCost:
    """Count the source kinetic selector with two three-bit coordinate words."""
    signature = projectile_select_t_signature(num_bits_n)
    return ComponentCost(
        "SelectTFirstQuantizationWithProj",
        int(signature.n_qubits()),
        GateCost(toffoli=5 * (num_bits_n - 1) + 3),
    )


def source_projectile_select_uv_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, num_bits_nuc_pos: int
) -> ComponentCost:
    """Count the potential selector on three unequal-width p and q axes."""
    signature = projectile_select_uv_signature(
        num_bits_p, num_bits_n, num_atoms, num_bits_nuc_pos
    )
    if min(num_bits_p, num_bits_n) < 2:
        raise ValueError(
            "Source projectile SELECT UV requires at least two momentum bits"
        )
    return ComponentCost(
        "SelectUVFirstQuantizationWithProj",
        int(signature.n_qubits()),
        calls=_projectile_select_uv_calls(num_bits_p, num_bits_n, num_bits_nuc_pos),
    )


def source_projectile_select_uv_gate_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, num_bits_nuc_pos: int
) -> GateCost:
    """Return the declared gates even when a one-bit conversion has no source peak."""
    projectile_select_uv_signature(num_bits_p, num_bits_n, num_atoms, num_bits_nuc_pos)
    gates = GateCost()
    for call in _projectile_select_uv_calls(num_bits_p, num_bits_n, num_bits_nuc_pos):
        gates = gates.plus(
            estimate_component_resources(call.component).gates, call.repetitions
        )
    return gates


def _projectile_select_uv_calls(
    p: int, n: int, num_bits_nuc_pos: int
) -> tuple[CostCall, ...]:
    # Keep the independent electronic and projectile arithmetic families.
    # The call-graph query combines structurally equal children.
    return (
        CostCall(_signed_conversion_cost(p), repetitions=3),
        CostCall(_signed_conversion_cost(n), repetitions=3),
        CostCall(source_add_cost(p + 1, p + 1), repetitions=3),
        CostCall(source_add_cost(n + 1, n + 1), repetitions=3),
        CostCall(
            ComponentCost("Toffoli", 3, GateCost(toffoli=1)),
            repetitions=3 * (p + n + 2),
        ),
        CostCall(_signed_conversion_cost(p + 2), repetitions=3),
        CostCall(_signed_conversion_cost(n + 2), repetitions=3),
        CostCall(source_apply_nuclear_phase_cost(n, num_bits_nuc_pos)),
    )


def source_projectile_prepare_uv_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, m_param: int, lambda_zeta: int
) -> ComponentCost:
    """Model forward momentum transfer and nuclear-index preparation."""
    nu = source_projectile_prepare_nu_cost(num_bits_p, num_bits_n, m_param)
    zeta = source_prepare_zeta_cost(num_atoms, lambda_zeta)
    return _projectile_prepare_uv_model(num_bits_n, num_atoms, m_param, nu, zeta)


def projectile_prepare_uv_adjoint_model(
    num_bits_p: int,
    num_bits_n: int,
    num_atoms: int,
    m_param: int,
    lambda_zeta: int,
    *,
    decomposed: bool = False,
) -> ComponentCost:
    """Estimate the native adjoint with the corrected nested ν flag direction."""
    nu = projectile_prepare_nu_adjoint_model(
        num_bits_p, num_bits_n, m_param, decomposed=decomposed
    )
    zeta = source_prepare_zeta_cost(num_atoms, lambda_zeta, adjoint=True)
    return _projectile_prepare_uv_model(
        num_bits_n, num_atoms, m_param, nu, zeta, adjoint=True
    )


def _projectile_prepare_uv_model(
    num_bits_n: int,
    num_atoms: int,
    m_param: int,
    nu: ComponentCost,
    zeta: ComponentCost,
    *,
    adjoint: bool = False,
) -> ComponentCost:
    signature = prepare_uv_signature(num_bits_n, num_atoms, m_param)
    ports = int(signature.n_qubits())
    index_width = int(signature.get_left("l").bitsize)
    calls = (
        CostCall(nu, bystander_qubits=index_width),
        CostCall(zeta, bystander_qubits=ports - index_width),
    )
    return ComponentCost(
        (
            "PrepareUVFirstQuantizationWithProj†"
            if adjoint
            else "PrepareUVFirstQuantizationWithProj"
        ),
        ports,
        calls=calls[::-1] if adjoint else calls,
    )


def source_projectile_prepare_uv_adjoint_gate_cost(
    num_bits_p: int, num_bits_n: int, num_atoms: int, lambda_zeta: int
) -> GateCost:
    """Count adjoint gates separately from the undefined source peak width."""
    nu = source_projectile_prepare_nu_adjoint_gate_cost(num_bits_p, num_bits_n)
    zeta = source_prepare_zeta_cost(num_atoms, lambda_zeta, adjoint=True)
    return nu.plus(zeta.gates, 1)


def source_projectile_select_first_quantization_cost(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    *,
    m_param: int = 2**8,
    num_bits_nuc_pos: int = 16,
) -> ComponentCost:
    """Estimate the source top-level projectile SELECT graph."""
    _require_positive(num_bits_p, num_bits_n, eta, num_atoms, m_param, num_bits_nuc_pos)
    p, n = num_bits_p, num_bits_n
    controlled_exchange = source_controlled_multiplexed_cswap_3d_cost(eta, p, n, 1)
    electron_exchange = source_multiplexed_cswap_3d_cost(eta, p)
    coordinate_swap = ComponentCost("CSwap", 2 * n + 1, GateCost(cswap=n))
    kinetic = source_projectile_select_t_cost(n)
    potential = source_projectile_select_uv_cost(p, n, num_atoms, num_bits_nuc_pos)
    ports = int(projectile_select_signature(p, n, eta, num_atoms, m_param).n_qubits())
    one_momentum = ports + 3 * n
    both_momenta = one_momentum + 3 * p
    # QubitCount postpones the q and rl allocations until first use and
    # releases each projectile coordinate immediately after its last swap.
    return ComponentCost(
        "SelectFirstQuantizationWithProj",
        ports,
        calls=(
            CostCall(
                controlled_exchange,
                bystander_qubits=one_momentum - controlled_exchange.qubits,
            ),
            *(
                CostCall(
                    coordinate_swap,
                    bystander_qubits=one_momentum - coordinate_swap.qubits,
                )
                for _ in range(3)
            ),
            CostCall(kinetic, bystander_qubits=one_momentum - kinetic.qubits),
            CostCall(
                electron_exchange,
                bystander_qubits=both_momenta - electron_exchange.qubits,
            ),
            CostCall(
                potential,
                bystander_qubits=both_momenta + num_bits_nuc_pos - potential.qubits,
            ),
            CostCall(
                controlled_exchange,
                bystander_qubits=both_momenta - controlled_exchange.qubits,
            ),
            *(
                CostCall(
                    coordinate_swap,
                    bystander_qubits=both_momenta - axis * n - coordinate_swap.qubits,
                )
                for axis in range(3)
            ),
            CostCall(
                electron_exchange,
                bystander_qubits=ports + 3 * p - electron_exchange.qubits,
            ),
        ),
    )


def source_projectile_prepare_tuv_cost(
    num_bits_t: int, *, adjoint: bool = False
) -> ComponentCost:
    """Count the source term preparation on nine through wires."""
    _require_positive(num_bits_t)
    return ComponentCost(
        "PrepareTUVSuperpositions",
        int(projectile_prepare_tuv_signature().n_qubits()),
        GateCost(toffoli=0 if adjoint else 6 * num_bits_t + 2),
    )


def source_projectile_prepare_first_quantization_cost(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    m_param: int = 2**8,
    num_bits_t: int = 16,
    num_bits_rot_aa: int = 8,
) -> ComponentCost:
    """Estimate the forward top-level projectile preparation."""
    potential = source_projectile_prepare_uv_cost(
        num_bits_p, num_bits_n, num_atoms, m_param, lambda_zeta
    )
    return _projectile_prepare_model(
        num_bits_p,
        num_bits_n,
        eta,
        num_atoms,
        m_param,
        num_bits_t,
        num_bits_rot_aa,
        potential,
        adjoint=False,
    )


def projectile_prepare_first_quantization_adjoint_model(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    m_param: int = 2**8,
    num_bits_t: int = 16,
    num_bits_rot_aa: int = 8,
) -> ComponentCost:
    """Estimate the top adjoint using the native UV model's corrected ν flag."""
    potential = projectile_prepare_uv_adjoint_model(
        num_bits_p, num_bits_n, num_atoms, m_param, lambda_zeta
    )
    return _projectile_prepare_model(
        num_bits_p,
        num_bits_n,
        eta,
        num_atoms,
        m_param,
        num_bits_t,
        num_bits_rot_aa,
        potential,
        adjoint=True,
    )


def _projectile_prepare_model(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    m_param: int,
    num_bits_t: int,
    num_bits_rot_aa: int,
    potential: ComponentCost,
    *,
    adjoint: bool,
) -> ComponentCost:
    particles = source_prepare_ij_cost(eta, num_bits_rot_aa)
    if eta < 2:
        raise ValueError("Source projectile PREPARE requires at least two particles")
    if adjoint:
        particles = replace(particles, name=f"{particles.name}†")
    term = source_projectile_prepare_tuv_cost(num_bits_t, adjoint=adjoint)
    kinetic = source_projectile_prepare_t_cost(num_bits_p, num_bits_n, adjoint=adjoint)
    particle_ports = 2 * (eta - 1).bit_length()
    ports = int(
        projectile_prepare_signature(num_bits_n, eta, num_atoms, m_param).n_qubits()
    )
    calls = (
        CostCall(term, bystander_qubits=ports - term.qubits),
        CostCall(particles, bystander_qubits=ports - particle_ports),
        CostCall(kinetic, bystander_qubits=ports - kinetic.qubits),
        CostCall(potential, bystander_qubits=ports - potential.qubits),
    )
    return ComponentCost(
        (
            "PrepareFirstQuantizationWithProj†"
            if adjoint
            else "PrepareFirstQuantizationWithProj"
        ),
        ports,
        calls=calls[::-1] if adjoint else calls,
    )


def source_projectile_prepare_first_quantization_adjoint_gate_cost(
    num_bits_p: int,
    num_bits_n: int,
    eta: int,
    num_atoms: int,
    lambda_zeta: int,
    *,
    num_bits_rot_aa: int = 8,
) -> GateCost:
    """Count adjoint gates independently of the invalid nested source width."""
    particles = source_prepare_ij_cost(eta, num_bits_rot_aa)
    if eta < 2:
        raise ValueError("Source projectile PREPARE requires at least two particles")
    potential = source_projectile_prepare_uv_adjoint_gate_cost(
        num_bits_p, num_bits_n, num_atoms, lambda_zeta
    )
    kinetic = estimate_component_resources(
        source_projectile_prepare_t_cost(num_bits_p, num_bits_n, adjoint=True)
    )
    # The adjoint term preparation declares zero gates at every precision.
    return potential.plus(kinetic.gates, 1).plus(particles.gates, 1)


@dataclass(frozen=True)
class ProjectilePrepareDeclaration(PrepareOracle):
    """Declare the projectile PREPARE ports and forward resource schedule."""

    num_bits_p: int
    num_bits_n: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8

    @cached_property
    def signature(self) -> Signature:
        return projectile_prepare_signature(
            self.num_bits_n, self.eta, self.num_atoms, self.m_param
        )

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return self.signature[:-3]

    @cached_property
    def junk_registers(self) -> tuple[Register, ...]:
        return self.signature[-3:]

    def resource_model(self, *, adjoint: bool = False) -> ComponentCost:
        """Price the forward schedule or the native corrected adjoint."""
        model = (
            projectile_prepare_first_quantization_adjoint_model
            if adjoint
            else source_projectile_prepare_first_quantization_cost
        )
        return model(
            self.num_bits_p,
            self.num_bits_n,
            self.eta,
            self.num_atoms,
            self.lambda_zeta,
            m_param=self.m_param,
            num_bits_t=self.num_bits_t,
            num_bits_rot_aa=self.num_bits_rot_aa,
        )


@dataclass(frozen=True)
class ProjectileSelectDeclaration:
    """Declare the projectile SELECT ports and resource schedule."""

    num_bits_p: int
    num_bits_n: int
    eta: int
    num_atoms: int
    lambda_zeta: int
    m_param: int = 2**8
    num_bits_nuc_pos: int = 16
    num_bits_t: int = 16
    num_bits_rot_aa: int = 8

    @cached_property
    def signature(self) -> Signature:
        return projectile_select_signature(
            self.num_bits_p, self.num_bits_n, self.eta, self.num_atoms, self.m_param
        )

    @property
    def control_registers(self) -> tuple[Register, ...]:
        return self.signature[:3]

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return self.signature[3:-2]

    @property
    def target_registers(self) -> tuple[Register, ...]:
        return self.signature[-2:]

    def resource_model(self) -> ComponentCost:
        return source_projectile_select_first_quantization_cost(
            self.num_bits_p,
            self.num_bits_n,
            self.eta,
            self.num_atoms,
            m_param=self.m_param,
            num_bits_nuc_pos=self.num_bits_nuc_pos,
        )
