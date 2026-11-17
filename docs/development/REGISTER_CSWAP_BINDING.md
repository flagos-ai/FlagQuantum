# Register controlled-swap binding

K-0197 and K-0198 cover the locked controlled-swap implementation and
header at Git blobs `6063d4fdc14070a514abfece73931066e8f31597` and
`f70f6309ca38b5e20115c72990f641d29519dc3d`.

The binding resolves scalars and explicit arrays before delegating each
lane to the existing physical controlled-swap primitive. Both targets
must be quantum wires. Ordinary Boolean controls select SWAP or an
empty lane. Classical controls select guarded SWAP. Quantum controls
select CSWAP with the original control and target order. Those paths
cover the source's homogeneous columns and the heterogeneous fallback.
There are no operand pairs to validate in a width-zero broadcast.
Quantum operand collisions are refused at native instruction formation.

The 158 locked stream comparisons cover original scalar and broadcast
cases, strided arrays, mixed lanes and the zero-width path. Read-only
classical inputs and all 128 complex amplitudes agree on the mixed
three-lane append path for both bit values. This emits one CSWAP, one
guarded SWAP and one SWAP. There are no measurement outcomes or work
wires. Emitted operations are tracked separately from the source's
operational non-Clifford count.

The original SWAP/CSWAP test file also checks guarded CSWAP text,
simulator outcomes, KMB byte transport and reaction depth. The locked
two-gate fixture reports two non-Clifford operations and reaction depth one.
Native now
checks the guarded KMX stream and all 64 complex amplitudes for both bit
values. K-0211 remains open for the KMB and reaction-depth dependencies.

Run `python -m pytest -q tests/unit/test_kickmix_cswap_binding_file.py`.
