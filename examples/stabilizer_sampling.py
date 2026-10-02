"""Sample a Clifford circuit that no amplitude store could hold.

The stabilizer engine tracks a circuit through the Pauli group instead of through
amplitudes, so its memory grows with the wire count squared rather than
exponentially. The last case here measures a thousand wires, which is the point
of the representation: a dense amplitude store for that many wires is not slow,
it is unrepresentable.

Run it from the repository root:

    python -m examples.stabilizer_sampling

The engine needs the optional `stim` distribution:

    pip install 'flagquantum[stim]'
"""

import time

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.simulation.stabilizer import sample_stabilizer

# A Bell pair: two outcomes, each perfectly correlated, and both wires measured.
bell = fq.Circuit(2).h(0).cx(0, 1)
samples = sample_stabilizer(bell, shots=8, seed=20260930)

print("wires measured:", bell.n_wires)
print("sample shape:", tuple(samples.shape))
print(
    "outcomes:", sorted({tuple(int(bit) for bit in shot) for shot in samples.tolist()})
)

# A subset keeps the requested order in the output columns. Wire 1 is in the
# first column here because it was requested first, not because it is the lower
# wire number.
subset = sample_stabilizer(bell, shots=4, wires=[1, 0], seed=1)
print("requested wires [1, 0] ->", subset.tolist())

# Determinism: one seed reproduces a run on one engine version and one machine.
assert (
    sample_stabilizer(bell, shots=64, seed=7)
    == sample_stabilizer(bell, shots=64, seed=7)
).all()

# A GHZ chain at a scale where the amplitude store is out of reach. The exact
# answer is still the two outcomes that differ in every bit, because every wire
# is correlated with wire 0.
n_wires = 1024
ghz = fq.Circuit(n_wires).h(0)
for wire in range(n_wires - 1):
    ghz = ghz.cx(wire, wire + 1)

started = time.perf_counter()
wide = sample_stabilizer(ghz, shots=8, seed=11)
elapsed = time.perf_counter() - started

print("GHZ wires:", n_wires)
print("sample shape:", tuple(wide.shape))
print("shots elapsed (s):", round(elapsed, 3))
print("all wires agree with wire 0:", bool((wide == wide[:, :1]).all()))
print("wire 0 outcomes:", wide[:, 0].tolist())

# For reference: a complex64 amplitude store over this many wires would need
# 2**1024 amplitudes, which is more than a million tebibytes.
tebibytes = 2**n_wires * 8 / 2**40
print("dense amplitude store would need (TiB):", f"{tebibytes:.3e}")

# A gate outside the Clifford group fails closed instead of being approximated,
# and the message names the gate set the engine does accept.
try:
    sample_stabilizer(fq.Circuit(2).t(0), shots=4, seed=1)
except CapabilityError as error:
    print("refused a non-Clifford gate:", str(error).split(";")[0])
