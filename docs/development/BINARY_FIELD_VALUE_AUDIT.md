# Complete binary field construction and scalar algebra

## Source and ownership

The immutable Kickmix revision is
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0.
K-0302 through K-0305 cover the complete `util/gf2_cyclotomic_factors.h`,
`gf2_field.cc`, `gf2_field.h` and `gf2_field.test.cc` files. The committed fixture
records all eleven original source and direct dependency Git blobs. Reference
files remain unchanged.

Private Core values own concrete scalar construction and arithmetic. They reuse
accepted `_binary_polynomial.py` and `_binary_matrix.py`. Compiler retains gate
construction and numerical codecs retain their existing field defaults.
Coefficient k is integer bit k. Matrix column j is the image of x**j. Physical
wire ordering remains the existing Compiler contract. Values own no quantum
workspace, measurement state or persistent allocator.

## Complete source mapping

| Original definitions | Native owner and retained behavior |
| --- | --- |
| CYCLOTOMIC_U64_OFFSETS | `_binary_field_factor_offsets.py`, all 513 original offsets |
| CYCLOTOMIC_U64_FACTORS and CYCLOTOMIC_BIG_FACTORS | `_binary_field_factors.py`, 840 original factors and 95 original degree/limb records |
| gf2_mersenne_prime_factors | `mersenne_prime_factors`, divisor order, cyclotomic stripping, distinct factor order and 192-bit limb reconstruction |
| spread_bits_32 | `_spread_bits_32`, original five masks and shift sequence |
| GF2PolyMod constructor, degree/modulus/mod_terms/sparse_reduction | Immutable `PolynomialModulus`, descending non-leading terms and original sixteen-term threshold |
| ireduce, mul, square, pow | `reduce`, `multiply`, `square`, `power`, source sparse/dense choice, comb multiplication, 32-bit spreading and positive exponent scan |
| is_irreducible, is_primitive_element | Corresponding modulus methods, Ben-Or test and ordered multiplicative factor tests |
| is_primitive_poly, search_primitive_poly, gf2_default_irreducible_poly | `default_polynomial`, exact ascending trinomial then lexicographic pentanomial search |
| gf2_is_irreducible | `is_irreducible`, zero/constant, degree-one and even-polynomial checks before the maximum-degree refusal |
| GF2Field five constructors | `BinaryField`, degree/default, explicit modulus, inferred degree, textual modulus and optional primitive element |
| degree, modulus, primitive_element | Immutable fields with the original equality identity |
| zero, one, x | Integer zero/one and `mod(2)` values, including degree one |
| is_element, is_primitive_element, mod, ireduce | `is_element`, `is_primitive`, `mod`, `reduce` |
| add, mul, square, frobenius, both pow overloads, invert, div | `add`, `multiply`, `square`, `frobenius`, `power`, `inverse`, `divide` |
| frobenius_matrix, constant_mul_matrix, linear_map_matrix | `frobenius_matrix`, `constant_matrix`, `linear_map_matrix`, accepted matrix values and exact column orientation |
| equality and inequality | Dataclass value equality over degree, modulus and primitive element |
| Original test consumers | Complete C++ suite plus committed unchanged-source observation driver and native execution regression |

Factor data retains the original Cunningham Project/galois provenance. It is
used to reconstruct group-order factors. Default moduli are searched. The two
private data files separate offset indexing from factor payloads and keep each
within the architecture size limit.

Python immutable values replace mutable C++ inline limbs and output references.
Degree and input bounds have explicit native ValueError failures. Nonnegative
scalar exponents retain the C++ domain. The signed-exponent Python binding is a
separate source owner. Bounded 512-entry caches retain immutable results. Duplicate
concurrent misses may recompute a deterministic value.

Default degree eight produces modulus 0x11D with primitive element 2. Explicit
AES modulus 0x11B selects primitive element 3. A missing requested leading term
is implied as in the original. Zero inverse and division by zero return zero.
These source conventions are covered explicitly.

## Source observations and execution

The fixture generator compiles unchanged field, matrix, polynomial and
fixed-width implementations with an I/O driver. It records 2355 complete output
observations, including factor requests for degrees 0 through 513, original
sampled default searches through degree 512, every polynomial below 1024,
constructor errors, all degree-one through degree-four operand pairs, wider
samples, 160-bit exponents and AES primitive candidates 0 through 256. Each
request compares a digest of all scalar results, construction metadata, matrix
outputs and error sentinels. The dense degree-32 candidate 0x1F2C32B6D is accepted
by the unchanged original constructor and exercises dense reduction.

All 27 unchanged tests in `gf2_field.test.cc` pass with the original helpers and
GoogleTest commit `f8d7d77c06936315286eb55f8de22cd23c188571`. These include
factor reconstruction and probable-prime checks for every degree 1 through 512,
large-degree searches and 512-bit arithmetic. Original build retains a
GoogleTest char8_t conversion warning. Logs are
`ftqc-reference-env/g2-binary-field-original-20261122.log` and
`g2-binary-field-original-build-20261122.log`.

```sh
python tests/unit/fixtures/generate_binary_field_values.py \
  --source-root ../ftqc-reference-env/kickmix \
  --output tests/unit/fixtures/binary_field_values.json
python -m pytest tests/unit/test_binary_field_values.py -q
```

The normal `Circuit.append(FieldOperation(...))` entry executes constant and
Frobenius matrices for degrees 1, 2, 4 and 8. Complete primitive streams agree
with existing source-audited Compiler wrappers. Complete complex amplitudes
match an independent basis permutation, including an entangled spectator, and
the exact inverse restores the input. Existing field product/inverse/division,
measured erasure and work cleanup evidence is reused in the concentrated suite.
Scalar construction has no measurement branches. Small applicable measured
branches remain covered by the existing execution tests.

## Resources, verification and retained boundaries

Scalar and matrix values emit zero quantum primitives and allocate zero work.
Physical consumers retain existing gate streams and resource accounting.
The previously accepted four-coordinate map (9,4,5,6) emits forward five CX and
one SWAP, inverse six CX and two SWAP, with zero measurements/work. Its analytical
forward PLU estimate separately records five Clifford operations and four logical
qubits with a free output permutation. No scalar execution count is inferred
from that analytical estimate.

Batch verification uses the locked reference wrapper, actual Kickmix Simulator
and preimported torch. It comprises 491 concentrated tests, 29 local integration
tests, strict typing on six changed Python files, Black, Ruff, Integration scope,
architecture, IR, protected API and whitespace checks. Logs use the
`ftqc-reference-env/g2-binary-field-*-20261122.log` prefix.

Default searches are sampled through degree 512. The original factor checks
cover all supported degrees. Complete Python GF binding K-0226 through K-0228,
whole FixedWidthInt/word-operation parents, original comparator/mixed-adder
failures and pending Proposals 116 through 119 remain separate open scope.
