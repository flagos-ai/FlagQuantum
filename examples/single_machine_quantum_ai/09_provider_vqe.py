"""Provider-owned, replayable H2 VQE execution."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
from flagquantum.services import (  # noqa: E402
    GroundStateVQERequest,
    PauliSum,
    PauliSumTerm,
    exact_sector_reference,
    run_ground_state_vqe,
)

hamiltonian = PauliSum(
    n_qubits=2,
    terms=(
        PauliSumTerm(-1.052373245772859, "II"),
        PauliSumTerm(0.39793742484318045, "ZI"),
        PauliSumTerm(-0.39793742484318045, "IZ"),
        PauliSumTerm(-0.01128010425623538, "ZZ"),
        PauliSumTerm(0.18093119978423156, "XX"),
    ),
)
request = GroundStateVQERequest(hamiltonian=hamiltonian, fixed_hamming_weight=1)

result = run_ground_state_vqe(request)
reference = exact_sector_reference(hamiltonian, fixed_hamming_weight=1)

if result.status != "completed":
    raise RuntimeError(result.unsupported)
if reference.status != "completed":
    raise RuntimeError(reference.unsupported)
assert result.provenance.package_version == fq.__version__

print(
    {
        "vqe_energy_hartree": result.energy,
        "exact_sector_energy_hartree": reference.energy,
        "parameters": result.parameters,
        "provider": result.provenance.provider,
        "package_version": result.provenance.package_version,
        "tool_version": result.provenance.tool_version,
        "execution_digest": result.provenance.execution_digest,
    }
)
