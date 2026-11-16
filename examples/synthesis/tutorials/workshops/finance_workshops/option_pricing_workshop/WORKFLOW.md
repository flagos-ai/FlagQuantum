# Native static option-payoff model

The model retains all 32 original probabilities and all 32 raw rotation angles
in the application physics module. `effective_angles(strike)` derives zero
angles for selector words below the cutoff; the original cutoff is 13. This
is a finite static model calculation. Host Python never reads a qubit, and
`select_rotation` remains the quantum operation on the complete superposition.
A zero table angle means identity on both indicator basis states, preserving
nonzero marker inputs. General control predicates on unrelated storage are
outside this transformation.

The qfunc loads the original full prefix-conditioned distribution unitary,
applies the derived rotation table and composes inverse/reflection helpers for
k iterations. Application and workshop entrypoints retain separate provenance
while sharing that quantum model. Users declare a five-bit unsigned asset,
choose bounded classical k/strike and read the named indicator distribution;
no gate design, physical wire numbers or bitstring decoding is required.

Tests compare both unchanged sources, native forms and independent full
loading/Grover matrices for k=0/1/2 without phase alignment. Additional tests
check raw cutoff boundaries, indicator 0/1, two nonzero coherent inputs with
cutoffs 12/14, workspace and real width/operation/depth/memory caps. Integer
cutoffs 0..32 cover the all-active and all-identity endpoint tables. The k
helper accepts 0..16 subject to the compiler's final construction/resource
limits. Default native k=1 uses 6 physical bits, zero work, 1724 operations and
depth 1498; the source uses 6 bits, 1724 operations and depth 1497. No resource or
performance improvement is claimed.

See the [shared amplification guide](../../../../amplification_guide.md) for phase and lifecycle
semantics. Dropped asset data is retained and marginalized. This accepts the
quantum program, not a full pricing Notebook, estimator or financial result.
