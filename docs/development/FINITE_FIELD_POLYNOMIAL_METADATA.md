# Finite-field polynomial numerical register metadata

Simulation connects `FiniteFieldPolynomialEncoding.register_type()` to the
existing domain, truth-table, bit-codec, typed-input and output-collection paths.
The projection uses the existing internal `RegisterType` record. It introduces
no Core type, public export, compiler dependency or executable admission.
`executable_register_width` continues to reject every polynomial declaration.

## Representation and ownership

The encoding string is `finite_field_polynomial`. Quantum storage uses
`quantum_bits = field.bitsize * (polynomial_degree + 1)` and zero classical
width; classical storage uses the converse. Parameters are ordered as follows:

1. `("storage", "classical")` for classical storage only;
2. `("polynomial_degree", degree)`;
3. the unchanged `FiniteFieldEncoding.register_type()` parameters, including
   characteristic, extension degree, order and optional defining-polynomial
   and coefficient-field identity.

`finite_field_polynomial_metadata` resolves concrete nonnegative built-in integer
degrees and widths. It checks storage shape, then delegates coefficient-field
reconstruction and identity validation to `finite_field_metadata`. It does not
reimplement field algebra, coefficient packing or array vectorization.
Standalone polynomial codecs retain their existing symbolic/negative-degree
behavior; those declarations are outside this concrete numerical resolver.

Qualtran v0.7.0 stores its QGFPoly/CGFPoly codec objects directly. The neutral
projection and inverse are FlagQuantum representation adapters, not copied
upstream methods. Numerical dispatch beneath them follows the unchanged pinned
source at `bcb17bb25360ed68ce3bc084645db3e929799389`.

Input validation reuses the existing polynomial codec through the shared
classical binding owner, preserving scalar symbolic bypass and distribution
resolution. Collection uses object arrays for both quantum and classical
polynomials, matching the source. Domain iteration and generic dense tables use
the same numerical owner. Scalar and shaped bit decoding retain the original
source differences, including zero-quantum-width classical shaped failures and
invalid binary words in odd fields. No modulo repair or silent fallback occurs.

## Verification and resources

The 88 new tests compare six small field/degree configurations with original
QGFPoly and CGFPoly objects: full domains, truth tables, every small binary word,
scalar/vector/matrix/empty codecs, binding and collection, nominal field identity,
and default dense outputs or exact source failure causes. Malformed projections
are rejected before optional field imports. Core refusal is checked independently.

A replacement test feeds reconstructed metadata into the existing polynomial
addition consumer without changing that consumer. It checks the complete
256-by-256 complex matrix, complex-superposition `circuit.append(op)`, double
append restoration and a clean spectator, plus the original source primitive
stream. Actual execution emits **4 CX**, zero T, zero measurements and no work
bits. The separate source QEC model estimates **4 Clifford**, zero T and zero
measurements. Numerical projection and codecs emit zero gates. This is coherent
execution with one deterministic branch, not a measured-branch claim.

Source-wide register support and executable polynomial graph ports remain open;
this numerical connection does not close either inventory parent or start G3.

Local checks: 88 new cases, 3974 concentrated regressions and 84 CPU/TN tests passed. Strict typing of six files, Black, Ruff, scope, architecture, internal IR and protected API checks passed. Polynomial encoding (57 statements, 10 branches) and metadata resolution (16 statements, 4 branches) each have 100% coverage. Nine source blobs match the pinned inventory.

Default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings in 141.62s (0:02:21)`. Observed failures all belong to the seven-case baseline; 0 baseline cases were not observed before the configured stop. This is a bounded run, not a green full suite. Local evidence prefix: `g2-field-polynomial-metadata-`.
