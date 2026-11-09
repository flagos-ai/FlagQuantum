# Native fermionic science

Owns operator encodings, static decompositions and circuit synthesis. Reuses the
existing public FermionOperator. Core Pauli values contain no execution objects.
Observable conversion lives here, not in Core. Import concrete modules instead
of expanding the stable root API.

Dependencies: standard library and PyTorch. This domain must not import Cirq,
OpenFermion, external chemistry drivers or runtime internals. Consumers execute
ordinary circuits through fq.run. Static preprocessing refuses autograd inputs;
time-dependent circuit builders preserve their declared tensor parameters.

Run `python -m examples.algorithms.fermion_energy` to see the same measurement
consumer use JW, parity and BK, each reporting 1.3. Modify a number-operator
coefficient and run `python -m pytest tests/unit/fermion -q` to verify a change.

Dense matrix helpers are explicitly bounded inspection tools. They do not
establish large-system capacity, distributed training or accelerator support.
