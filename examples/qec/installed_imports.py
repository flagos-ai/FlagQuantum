"""Verify existing QEC imports and results from an installed distribution."""

import pickle

import flagquantum.qec as qec
from flagquantum.qec.codes.surface import RotatedSurfaceCode

code = qec.get_code("rotated_surface", distance=3)
print("Public exports:", len(qec.__all__))
print("Registered class:", type(code) is qec.RotatedSurfaceCode is RotatedSurfaceCode)
print("Code pickle path:", type(pickle.loads(pickle.dumps(code))).__module__)
model = qec.DetectorErrorModel.from_stim_text(
    "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
)
print(
    "Correction:", qec.get_decoder(qec.AUTHORITY_NAME, model).decode((0,)).observables
)
point = qec.run_repetition_memory_noise_sweep((0.0,), rounds=3, shots=8, seed=53)[0]
print("Logical error rate:", point.logical_error_rate)
