"""Run the existing code-family workflow through root and responsibility-package imports."""

import pickle

from flagquantum.qec import (
    RotatedSurfaceCode,
    SteaneCode,
    css_code_matrices,
    get_code,
    run_repetition_memory_noise_sweep,
)
from flagquantum.qec.codes.surface import RotatedSurfaceCode as SurfaceFamily
from flagquantum.qec.codes.surface.rotated import RotatedSurfaceCode as Owner

registered = get_code("rotated_surface", distance=3)
direct = Owner(3)
print("Registered family:", type(registered) is RotatedSurfaceCode is SurfaceFamily)
print("Single class:", RotatedSurfaceCode is SurfaceFamily is Owner)
print("Equivalent checks:", registered.checks == direct.checks)
print("Pickle path:", type(registered).__module__)
print("Pickle identity:", type(pickle.loads(pickle.dumps(direct))) is Owner)
matrices = css_code_matrices(SteaneCode())
print(
    "Steane matrices:",
    matrices.num_qubits,
    matrices.num_checks,
    matrices.num_observables,
)
point = run_repetition_memory_noise_sweep((0.0,), rounds=3, shots=8, seed=53)[0]
print("Logical error rate:", point.logical_error_rate)
