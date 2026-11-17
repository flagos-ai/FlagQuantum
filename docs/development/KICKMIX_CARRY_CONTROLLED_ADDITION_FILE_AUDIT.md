# Carry-controlled addition file audit

The locked Kickmix source is commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Ledger | Source | Frozen blob | Native owner |
| --- | --- | --- | --- |
| K-0057 | `gen_iadd_control.cc` | `ee8062ae063ce241383bdb633ead74c4dafc5db1` | `compiler/registers/arithmetic/carry_controlled_addition.py` |
| K-0058 | `gen_iadd_control.h` | `6225f00ab5269a622ffd009049fff8222b52ae9b` | Same compiler owner |
| K-0059 | `gen_iadd_control.perf.cc` | `79c67d74d6231c6e26995e12d24be93d0ab44b7a` | `benchmarks/benchmark_physical_carry_controlled_addition.py` |
| K-0060 | `gen_iadd_control.test.cc` | `6d34dc55f06c82c0007b05f72912899d93be8edf` | `tests/unit/test_kickmix_controlled_addition*.py` |

The logical entry consumes big-endian target and offset words, a quantum enable
and a constant, classical or quantum carry. It changes only the target word.
Source span order is little-endian. A wider offset retains its unused high
bits. Subtraction conjugates the shared false-carry adder with target-wide X.
Widths zero and one borrow no work. Larger widths require the source
admission minimum of n clean wires, use n-1 and reuse one result slot. The caller
owns the classical carry slot and must supply a distinct result slot.

The existing append path admits pre-cleared carry wires. The physical export
uses the same carry schedule, adds an R at each used wire's first write and
emits the original block marks even for the zero-width empty call. A classical
carry contributes a scalar correction whose outer guard is the HMR result,
with the carry bit tested inside that scope. Subtraction retains the nested
add mark. Unused bound wires remain declared in the native artifact.

The physical fixture covers 180 operand and workspace cases at widths 0–8 plus
the locked width-256 workload. The digest covers ordered operands, resets,
both guard records, phase operations and marks. Only source register
declarations and the native transport header are outside the comparison.
Regenerate against the locked checkout with
`python tests/unit/fixtures/generate_kickmix_controlled_addition.py --physical --source-root <locked-kickmix-root> --output tests/unit/fixtures/kickmix_controlled_addition_file.json`.

Both unchanged upstream fuzz tests, `icadd.fuzz` and `icsub.fuzz`, passed.
The accepted append tests cover the constant, quantum and classical carry
routes, all small HMR histories and native basis/phase checks. The new
physical tests exhaust the reset and HMR histories for coherent two-bit
registers, including subtraction and both values of a classical carry in a
slot above the HMR destination. They check unnormalized complex amplitudes,
all input restoration and zero clean work.

## Width-256 workload

| Observed physical operation | Count |
| --- | ---: |
| CX | 1,277 |
| CCX | 511 |
| R | 255 |
| HMR | 255 |
| CZ | 255 |
| Z | 255 |
| Block marks | 2 |

The workload binds 256 work wires, touches 255, and uses one result bit.
The locked C++ benchmark reported 1.2 microseconds for mutable record
generation. The native median over 25 iterations after three warmups on the
same checkout host was 17.79 milliseconds, covering instruction generation,
validation and physical lowering. The timed regions measure different work.
The table reports emitted records. No resource model or inferred T count was
used for those counts.

Run the native workload with
`python -m benchmarks.benchmark_physical_carry_controlled_addition`.
The separate K-0056.04 mixed-adder prefix source defect remains open.
