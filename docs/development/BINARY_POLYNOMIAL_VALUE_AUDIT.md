# Binary polynomial value dependency closure

## Source and native ownership

The frozen Kickmix revision is
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0.
The complete source group is `util/gf2_poly.cc`, `gf2_poly.h` and
`gf2_poly.test.cc`, ledger parents K-0309, K-0310 and K-0311. Git blob
identities appear in `tests/unit/fixtures/binary_polynomial_values.json`.
The source and its tests remain unchanged.

Core `_binary_polynomial.py` owns private backend-neutral polynomial values.
The representation remains an ordinary nonnegative integer with coefficient
of x**k in bit k. The module exports no stable API or IR vocabulary. Compiler
imports its remainder operation through the existing `_remainder` name.
Existing Compiler callers retain unbounded polynomial width. Simulation field
construction can consume the same value functions without importing Compiler.
Numerical arrays, field-library objects and circuit emission remain at their
existing domain boundaries.

## Complete definition mapping

| Source surface | Native owner and adaptation |
| --- | --- |
| Zero constructor, from_u64 | Integer zero and nonnegative 64-bit values |
| monomial | Integer `1 << k` after the source coefficient-capacity check |
| from_fixed_width_int, to_fixed_width_int | Existing integer storage with explicit declared-width check and `truncate`; conversion refuses widths above 1024 |
| from_str and trim_ws | `parse`, ASCII whitespace, original binary/hex declared-digit accounting and algebraic repeated-term rejection |
| bit, set_bit, xor_bit | Integer bit tests and masks, bounded mutation admission at the source binding boundary |
| is_zero, degree, num_bits_in_use, weight | Integer zero, bit_length and bit_count; the source unsigned zero-degree sentinel is converted only by the observation adapter |
| truncate | `truncate`, saturating the requested storage mask at 1024 bits |
| equality, inequality and XOR | Integer value comparison and XOR |
| left/right shifts | Integer shifts with the 1024-bit mask on left shift; shifts at or above capacity produce zero |
| ixor_shifted | `xor_shifted`, pure value ownership also preserves self-alias semantics |
| mul | `multiply`, original larger-operand table, 16-entry four-bit comb, descending nibble and ascending word traversal, original degree-sum overflow refusal |
| imod | `remainder`, original shift-XOR reduction; zero modulus is rejected before iteration |
| gcd | `gcd`, original Euclidean traversal |
| invert_mod | `invert_mod`, original cofactor-only extended Euclid, bounded shifted XOR, zero for nonunits |
| str and ostream insertion | Integer uppercase hex formatting with `0x` prefix |
| algebraic_str | `algebraic`, descending coefficients and original highest-four/lowest-two abbreviation, including max_terms below six |

C++ inline allocation, mutable references and size_t storage are language-level
implementation details. Pure integer values preserve their observable results
without introducing a second register representation. Source storage admission
remains 1024 bits. Existing unbounded Compiler reduction is deliberately retained
through the shared algorithm. The remaining complete GF binding will own its
field-degree and scalar port admission.

## Reproduction and evidence

Generate the committed observations with the accepted reference Python wrapper
and the following fixture generator arguments.

```sh
python tests/unit/fixtures/generate_binary_polynomial_values.py \
  --source-root ../ftqc-reference-env/kickmix \
  --output tests/unit/fixtures/binary_polynomial_values.json
python -m pytest tests/unit/test_binary_polynomial_values.py -q
```

The generator verifies all three source Git blobs and compiles unchanged source
`gf2_poly.cc` and `fixed_width_int.cc` with a small input/output driver. It records
2,639 observations, including 300 broad random operand pairs for multiplication,
reduction, GCD, inverse and XOR, all 256 AES-field inverses, parsing positive and
negative cases, 64-bit word boundaries, 512/1024-bit capacity boundaries, shifts,
truncation, self aliases, metadata and rendering. These are executed original
C++ results, independent of the native implementation.

All 23 tests in the complete unchanged `gf2_poly.test.cc` pass with original
`test.test.cc` RNG helpers and GoogleTest 1.14.0 commit
`f8d7d77c06936315286eb55f8de22cd23c188571`. The GoogleTest compiler emits its
existing char8_t-to-char32_t warning. Tests and reference implementations were
unchanged. The original test log is
`ftqc-reference-env/g2-binary-polynomial-original-20261122.log`.

The source degree-zero sentinel is UINT64_MAX in the C++ observation format.
The native value representation uses bit_length()-1. The test adapter converts
that sentinel explicitly. C++ std::invalid_argument is represented by native
ValueError. Error wording is owned by FlagQuantum.

## Execution and resources

The shared value kernel emits zero quantum primitives, measurements or work
qubits. It supplies scalar arithmetic to existing circuit synthesis. Physical
complex-amplitude, measurement-branch and clean-work evidence is reused from the
field multiplication, inverse, division and erasure owners and rerun in the
concentrated batch. The normal user entry remains `circuit.append(op)` through
the existing field operation and reversible-phase binding paths.

Primitive operation counts from these execution owners remain separate from
analytical resource models. This value utility has no source circuit resource
formula of its own. No arithmetic gate synthesis algorithm changed.

## Remaining dependency group

K-0303 through K-0308 field construction and binary matrices remain open, along
with K-0226 through K-0228 complete Python GF bindings. Default primitive
polynomials use the original ordered search; cyclotomic factor tables support
primitive-element testing. The earlier 78 unchanged original GF binding tests
are reusable intake evidence. Pending Proposals 116 through 119 and previously
recorded upstream defects retain their own boundaries.

## Batch result

355 concentrated field/value regression tests and 29 local integration tests
pass. Strict typing passes for all four changed Python files. Black, Ruff,
Integration scope, architecture, internal IR contract, public API snapshot and
whitespace checks pass. The environment receipt verifies the actual Kickmix
0.1 extension and Simulator, plus all six source/dependency blobs against the
immutable inventory. FixedWidthInt and word_ops remain independent open source
parents; this batch accepts their polynomial conversion and parse dependency
edges only. The full G2 acceptance remains open.

## Complete uint64 bit-length dependency

K-0289 (`util/bit.h`, blob
`ea1c426299ab2ed0a7da662f629b133a106228e2`) and K-0290
(`util/bit.test.cc`, blob `ead0a4db46236891ee1304a5b0e5ee99a7aa6ce1`)
are the complete unsigned 64-bit length dependency. Both immutable source blobs
were checked against the original inventory. The only source definition is
`bit_length(uint64_t)`. It returns zero for zero and the highest set-bit position
plus one otherwise, using descending 32/16/8/4/2/1-bit tests.

Existing native `int.bit_length()` supplies that exact value throughout the
source uint64 domain. Core `_binary_polynomial`, `_binary_matrix` and
`_binary_field_modulus` already consume this builtin. Their accepted parsing,
capacity, construction and physical field consumers remain authoritative. There
is no additional production helper, wrapper, representation or circuit schedule.
Native arbitrary-width integer callers keep their existing wider domain.

The unchanged original `bit.bit_length` C++ test passes against the original
header. Its full assertions cover zero through seventeen, every power-of-two
boundary from two through sixty-three and seven alternating-bit examples. A
separate compiled original-header driver matches all 5,282 native values from
zero through 1,023, all 64-bit power boundaries, the maximum uint64 value and
4,096 deterministic random words. This is sampled evidence across the domain,
with all power boundaries included. The exact algorithm and builtin semantics
establish the general correspondence.

The probe's deterministic request set is reproduced below. Each requested word
is passed as a decimal uint64 to a C++ driver including the unchanged `bit.h`.
The driver emits `unsigned(kickmix::bit_length(value))`. The complete receipt is
`ftqc-reference-env/g2-bit-length-intake-20261122.json` in the shared evidence
workspace. Original C++ test output is retained alongside it.

```python
import random

rng = random.Random(289)
values = sorted(
    {
        *range(1024),
        *(
            min((1 << 64) - 1, (1 << k) + delta)
            for k in range(64)
            for delta in (-1, 0, 1)
            if (1 << k) + delta >= 0
        ),
        *(rng.getrandbits(64) for _ in range(4096)),
        (1 << 64) - 1,
    }
)
expected = [value.bit_length() for value in values]
```

The shared scalar utility emits zero physical primitives, measurements and
workspace and declares no analytical circuit formula. Full complex-amplitude,
measured cleanup and resource evidence remains with the accepted field/lookup
execution owners. Their 530 finite-field and 499 lookup concentrated checks,
plus the 29 integration checks, are reused from the same verified branch.
Only this definition mapping and evidence record changes in the dependency
closure commit. FixedWidthInt, word operations and phase-angle whole-file
boundaries remain separately open.
