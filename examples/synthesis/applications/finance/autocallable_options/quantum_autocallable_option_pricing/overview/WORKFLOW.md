# Autocallable workflow checks

The grid uses the mass of a standard normal between the original edges in
`[-3, 3]`, normalized over that interval. The original left endpoints and
coefficient-by-coefficient truncation yield raw increments `(-2, -1, 0, 1)`,
in steps of one quarter. The first call takes priority over the second. The
down-and-in put requires a strict breach in any period and a terminal return
strictly below the strike. All other paths have zero payoff.

The quantum circuit binds the newly generated grid weights, normalized
exponential reference weights, and comparison thresholds into the existing
native preparation. Its three fixed rotations implement this specific contract.
The branch reference computes those rotations from discounting and normalization.
It independently builds a five-wire Circuit for each of the 64 paths, combines
their amplitudes in canonical named-register order, and checks Grover reflection
as a rank-one operation. The comparison includes absolute phase, the full joint
state, and zero population of every nonzero compiler work word.

Each circuit retains eleven logical wires. The source's hosted inspector and
width optimization are replaced by the local compiler's measured resource
report under an explicit 24-wire bound. The reported width is the executed
circuit's physical width. The preparation's decoded probability equals the
enumerated finite-grid payoff. The first Grover iterate verifies coherence and
oracular behavior separately. The original adaptive execution remains disabled;
its stored estimate and interval are displayed with their source attribution.
