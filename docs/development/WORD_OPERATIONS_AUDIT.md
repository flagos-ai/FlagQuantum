# Complete scalar word operations

The frozen Kickmix revision is `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
The unchanged `util/word_ops.h` and `util/word_ops.test.cc` map to parents
K-0322 and K-0323. Their Git blobs are checked before generating observations.

## Ownership and correspondence

Core owns the private `_word_operations.py` value functions. Operands are
unsigned 64-bit built-in integers. Tuple results replace source output pointers.
Low and high limbs are ordered least significant first. These scalar utilities
have no quantum register, compiler, numerical state, RNG or backend ownership.
The declared-width integer product owner consumes these same functions.

| Complete source definition | Native owner and result |
| --- | --- |
| add_carry_u64 | add_carry returns low word and outgoing carry |
| sub_borrow_u64 | subtract_borrow returns low word and outgoing borrow |
| extract_bits_u64 | extract_bits packs mask-selected bits in increasing input-bit order |
| full_fma64 | full_multiply_add returns low and high words of the product plus both offsets |

Python arbitrary-width intermediate arithmetic implements the exact bounded
result. The source platform intrinsics, unsigned-128 route and 32-bit product
route have the same mathematical contract. No production foreign dependency or
public API is added. Negative, overwide and noninteger native operands refuse
before calculation. Source C++ implicit operand coercion has its language owner.

## Acceptance

All three unchanged original C++ tests pass. The checked-in generator verifies
both frozen blobs, compiles the unchanged header twice, and compares the ordinary
unsigned-128 and forced 32-bit product routes on identical inputs. All 3,472
observations agree across both source routes and the native owner. They cover
all combinations of the six original boundary words, both incoming-bit values,
every mask position and seeded random words. Each observation compares all four
functions, including both full multiply-add output limbs. The fixture is streamed
as JSON lines. The source header and test remain unchanged.

Regeneration uses the isolated frozen source tree.

```bash
python tests/unit/fixtures/generate_word_operations.py \
  --source-root /path/to/locked/kickmix/src \
  --output tests/unit/data/word_operations_reference.jsonl
python -m pytest tests/unit/test_word_operations.py -q
```

These scalar operations emit zero primitives, measurements and work qubits.
Quantum consumer resource accounts and analytical circuit models keep their
existing owners. Source x86 intrinsic instruction selection and throughput have
no native Python instruction-count claim. The observed source lane is Darwin
arm64. The forced fallback covers the source 32-bit multiplication branch.
