# Symbolic metadata in classical numerical composition

Simulation's existing typed value and domain adapters now use
`value_register_metadata`. Widths, bounds, Montgomery moduli and fractional
widths can remain SymPy expressions. Their existing numerical owners decide
whether a requested operation succeeds. No substitution, integer coercion or
symbolic solver is added. Core executable admission remains unchanged.

The shared metadata owner still exposes `concrete_register_metadata` to binary
codecs. It retains its original concrete-width and parameter refusal messages.
Both readers share parameter traversal; neither duplicates dtype algorithms.
Symbolic support lazily imports SymPy only when noninteger metadata is inspected.

```python
from sympy import Symbol
from flagquantum.core.ir.program.register_types import bounded_unsigned_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.simulation.classical_domains import register_classical_domain
from flagquantum.simulation.classical_typed_values import bind_typed_register_values

width = Symbol("n", integer=True, positive=True)
register = Register("word", bounded_unsigned_register_type(width, 3))
assert list(register_classical_domain(register)) == [0, 1, 2]
value = Symbol("value")
assert bind_typed_register_values(register, {"word": value})[()] is value
```

The previous concrete-only composition refused both examples before reaching the
source-equivalent numerical owners. Symbolic scalar input still follows shared
binding's Expr bypass; sampled symbolic objects still undergo validation. Scalar
collection returns its original object without selecting an integer array dtype.
A symbolic width does not imply that concrete numeric values can be validated:
source comparisons can fail with TypeError. Symbolic-width quantum integer array
storage can also fail while classical object storage succeeds. Empty arrays keep
source-specific validation and allocation behavior.

Domain enumeration is not uniformly symbolic. A bounded integer can enumerate a
concrete bound independently of symbolic width. Montgomery with a symbolic
modulus falls back to its stored-word domain, provided width is concrete. Fixed
point can enumerate concrete stored width with symbolic fractional width. Other
symbolic ranges or shifts retain the original failures. The native adapter may
raise during domain acquisition where the source generator raises on iteration;
its diagnostic context remains native. Symbolic finite-field projection and
symbolic array shapes are outside this slice.

## Evidence and resources

`test_classical_symbolic_metadata.py` compares unmodified Qualtran v0.7.0
`bcb17bb25360ed68ce3bc084645db3e929799389` dtype domains and real state assignment /
collection against native adapters. Seven encoding families, both storage
identities and symbolic width/parameter modes cover scalar identity, numeric
validation, vector/empty arrays, source failures and successful symbolic domains.
Explicit codec refusal and unknown metadata checks keep the binary boundary.

The existing measured Add consumer still checks all 64 inputs, its complete
complex matrix and all 16 measurement branches with both work bits zero. Actual
primitives remain CX40, H12, T8, Tdg8, S4, measurement4, CZ4 and reset4. The separate
source QEC model reports AND4, Clifford22 and measurement4. Metadata and domain
composition emit zero primitives and allocate no quantum work bits. No new
physical decomposition or symbolic executable Program is claimed. The reference
files are unchanged. G2 remains open; G3 is not started.

Local checks: 87 new cases, 3794 concentrated regressions and 84 CPU/TN tests passed. Strict typing of five Python files, Black, Ruff, Simulation scope, architecture, internal IR and API protection passed. Shared metadata (31 statements/16 branches), typed values (53/22), domains (30/16) and concrete codec (51/28) reached 100%; the state owner remains 97%. Seven source blobs match the inventory; three captured old/source/after examples confirm previously refused symbolic domains.

Default smoke/unit: `7 failed, 22385 passed, 300 skipped, 1048 warnings in 119.71s (0:01:59)`. Observed failures are all in the seven-case baseline; 0 baseline cases were not observed before the bounded stop. This is not a green full suite. Evidence uses `g2-classical-symbolic-metadata-` in the local reference directory.
