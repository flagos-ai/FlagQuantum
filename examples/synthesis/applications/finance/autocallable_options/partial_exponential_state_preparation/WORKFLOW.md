# Partial exponential preparation

Run from the repository root:

```sh
python -m examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.main
python -m examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.comparisons.circuit
```

The native program prepares an increasing exponential input on five bits,
marks x >= 21, applies one ancilla-tuned exact amplitude-amplification round and
subtracts 18 modulo 32. The retained interval [21,31] becomes [3,13]. Input
amplitudes are proportional to exp(x/4) for rate -0.5; the program does not
load a precomputed truncated final state. The attenuation output and every
compiler work bit must return to zero.

The frozen original declares overlap 0.996625424765961 and is correctly refused.
Successful acceptance uses the existing approved overlap 0.99795457844373736.
The native builder checks the declared overlap against the input distribution;
it never silently replaces an incorrect declaration. Additional rates require
a matching overlap and the bounded one-round profile.

Independent Circuit preparation uses product RY gates from analytic bit odds,
direct oracle/reflection matrices and a modulo permutation. Acceptance checks
three nonzero rates, every complex amplitude and absolute phase; inverse
restoration on complex nonzero input; all modular wrap columns through a
nonuniform coherent state with a spectator; alias, input, overlap and budget
refusals. Original and approved source identities remain test-side.

The native default uses 8 physical bits including 2 compiler work bits, an
explicit clean attenuation port, 2,399 operations and depth 2,046. Generic
numeric predicate/phase synthesis is an optimization gap, not a resource
improvement claim. This certifies the quantum program, not the complete Notebook.
