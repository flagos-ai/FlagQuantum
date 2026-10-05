"""Quantum algorithms and hybrid optimization workflows."""

from . import amplitude_estimation as amplitude_estimation
from . import arithmetic as arithmetic
from . import cdr as cdr
from . import core as core
from . import data_encoding as data_encoding
from . import error_mitigation as error_mitigation
from . import feature_selection as feature_selection
from . import folding as folding
from . import grover as grover
from . import kmedians as kmedians
from . import logical_resources as logical_resources
from . import nelder_mead as nelder_mead
from . import pca as pca
from . import pec as pec
from . import qarm as qarm
from . import quantum_kernel as quantum_kernel
from . import qubo as qubo
from . import readout_mitigation as readout_mitigation
from . import spsa as spsa
from . import svd as svd
from . import trotter as trotter
from .arithmetic import (
    AdderWires,
    adder_circuit,
    adder_wires,
)
from .cdr import (
    CDR_ASSUMPTIONS,
    CDR_LIMITATIONS,
    CDR_SNAP_OPCODES,
    CdrResult,
    CliffordFit,
    CliffordTrainingPoint,
    CliffordVariant,
    clifford_variants,
    run_cdr,
)
from .core import (
    AdaptVQEIteration,
    AdaptVQEResult,
    Hamiltonian,
    HamiltonianTerm,
    LayerwiseVQEResult,
    OptimizerFactory,
    VQEResult,
    hardware_efficient_ansatz,
    hardware_efficient_parameter_count,
    heisenberg_chain_hamiltonian,
    heisenberg_hva,
    heisenberg_hva_parameter_count,
    pauli_term,
    qaoa_circuit,
    qaoa_loss,
    run_adapt_vqe,
    run_hybrid_vqe,
    run_layerwise_vqe,
    run_vqe,
    transverse_field_ising,
    vqe_loss,
    zz_chain_hamiltonian,
)
from .data_encoding import (
    amplitude_encode,
    angular_encode,
    append_angular_encode,
)
from .error_mitigation import (
    ExtrapolationFit,
    ZneMeasurement,
    ZneResult,
    extrapolate_polynomial,
    extrapolate_richardson,
    run_zne,
    scale_noise_model,
)
from .folding import (
    FOLDING_ASSUMPTIONS,
    FOLDING_SCHEMA,
    FOLDING_STRATEGIES,
    FoldingPlan,
    fold_program,
)
from .logical_resources import (
    LOGICAL_RESOURCE_BASIS,
    SURFACE_CODE_MODEL,
    LogicalResourceReport,
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)
from .nelder_mead import (
    NELDER_MEAD_ASSUMPTIONS,
    NELDER_MEAD_LIMITATIONS,
    NelderMeadOptimizer,
    NelderMeadResult,
)
from .optimization import (
    HybridOptimizationResult,
    OptimizationRecord,
    OptimizationStage,
    optimize_hybrid,
)
from .pec import (
    PauliTwirlDecomposition,
    PecLocation,
    PecResult,
    pauli_twirl_decomposition,
    run_pec,
)
from .readout_mitigation import (
    MAX_CORRELATED_BLOCK_QUBITS,
    NORMALIZATION_TOLERANCE,
    READOUT_MITIGATION_ASSUMPTIONS,
    READOUT_MITIGATION_LIMITATIONS,
    READOUT_MITIGATION_SCHEMA,
    SINGULAR_VALUE_FLOOR,
    ReadoutBlock,
    ReadoutMitigationPlan,
    ReadoutMitigationResult,
    plan_readout_mitigation,
    run_readout_mitigation,
    run_readout_mitigation_counts,
)
from .spsa import SPSAOptimizer
from .trotter import (
    TROTTER_ORDERS,
    pauli_exponential_circuit,
    trotter_circuit,
)

__all__ = [
    "AdaptVQEIteration",
    "AdaptVQEResult",
    "AdderWires",
    "CDR_ASSUMPTIONS",
    "CDR_LIMITATIONS",
    "CDR_SNAP_OPCODES",
    "CdrResult",
    "CliffordFit",
    "CliffordTrainingPoint",
    "CliffordVariant",
    "ExtrapolationFit",
    "Hamiltonian",
    "HamiltonianTerm",
    "HybridOptimizationResult",
    "LayerwiseVQEResult",
    "LOGICAL_RESOURCE_BASIS",
    "LogicalResourceReport",
    "NELDER_MEAD_ASSUMPTIONS",
    "NELDER_MEAD_LIMITATIONS",
    "NelderMeadOptimizer",
    "NelderMeadResult",
    "OptimizerFactory",
    "OptimizationRecord",
    "OptimizationStage",
    "PauliTwirlDecomposition",
    "PecLocation",
    "PecResult",
    "SPSAOptimizer",
    "SURFACE_CODE_MODEL",
    "TROTTER_ORDERS",
    "VQEResult",
    "ZneMeasurement",
    "ZneResult",
    "adder_circuit",
    "adder_wires",
    "amplitude_encode",
    "amplitude_estimation",
    "angular_encode",
    "append_angular_encode",
    "arithmetic",
    "cdr",
    "clifford_variants",
    "data_encoding",
    "error_mitigation",
    "extrapolate_polynomial",
    "estimate_logical_resources",
    "extrapolate_richardson",
    "feature_selection",
    "FOLDING_ASSUMPTIONS",
    "FOLDING_SCHEMA",
    "FOLDING_STRATEGIES",
    "FoldingPlan",
    "fold_program",
    "folding",
    "grover",
    "hardware_efficient_ansatz",
    "hardware_efficient_parameter_count",
    "heisenberg_chain_hamiltonian",
    "heisenberg_hva",
    "heisenberg_hva_parameter_count",
    "kmedians",
    "nelder_mead",
    "logical_resources",
    "optimize_hybrid",
    "pauli_exponential_circuit",
    "pauli_term",
    "pauli_twirl_decomposition",
    "pca",
    "pec",
    "qaoa_circuit",
    "qaoa_loss",
    "MAX_CORRELATED_BLOCK_QUBITS",
    "NORMALIZATION_TOLERANCE",
    "READOUT_MITIGATION_ASSUMPTIONS",
    "READOUT_MITIGATION_LIMITATIONS",
    "READOUT_MITIGATION_SCHEMA",
    "ReadoutBlock",
    "ReadoutMitigationPlan",
    "ReadoutMitigationResult",
    "SINGULAR_VALUE_FLOOR",
    "plan_readout_mitigation",
    "qarm",
    "quantum_kernel",
    "qubo",
    "readout_mitigation",
    "run_readout_mitigation",
    "run_readout_mitigation_counts",
    "run_hybrid_vqe",
    "run_layerwise_vqe",
    "run_vqe",
    "run_adapt_vqe",
    "run_cdr",
    "run_pec",
    "run_zne",
    "scale_noise_model",
    "spsa",
    "surface_code_qubits_per_logical",
    "svd",
    "trotter",
    "trotter_circuit",
    "transverse_field_ising",
    "vqe_loss",
    "zz_chain_hamiltonian",
]
