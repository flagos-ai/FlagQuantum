# Kickmix mixed modular addition and subtraction parity

This audit covers K-0113 through K-0116 at Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

| Source file | Git blob | Native evidence |
| --- | --- | --- |
| `gen_iadd_mod.cc` | `ba7e69f03ad225a13ca996bb44f23c36a6f4f28c` | Independent add and subtract schedules, with shared add, compare and measured cleanup |
| `gen_iadd_mod.h` | `a65f953052beb38773700ece8f9502ad47a11e78` | Big-endian target, offset and mixed modulus, with a coherent optional enable |
| `gen_iadd_mod.perf.cc` | `029c34224f3d5755c222152efe1d7f7dae3970cb` | All three locked width-256 workloads |
| `gen_iadd_mod.test.cc` | `5ab68cac667fb715669f4e5a19b83bb96b0019fc` | Both unchanged fuzz tests and the existing exhaustive small-history tests |

The production emitter is unchanged. It acts in place on a target in
`[0, modulus)` with an offset in the same range. The caller supplies clean
work and preserves the offset, modulus and enable. The compiler owns
reused scratch result slots. Source quantum-modulus comparison with
insufficient clean workspace can read outside its span; the native
admission rejects that case. The independent mixed-addition
extra-clean-prefix source defect remains tracked as K-0056.04.

The existing add/subtract acceptance checked 976 source effective streams
and exhaustively verified the signed arithmetic on all reset and
measurement histories of small instances, including full complex states,
work cleanup and the circuit append entry. The two original C++ fuzz tests
and 17 focused native tests passed in the current batch. The 5,417-test
concentrated regression, strict typing, format, scope, architecture, internal
IR and public API checks also passed. In a separate
wide basis check, the locked Simulator ran 64 deterministic inputs for each
workload on each implementation, including overflow, underflow and wrap
boundaries. Both implementations produced the expected modular values,
preserved all input ports and returned zero work and exact zero phase on
every sampled history.

## Workload records and feedback

Both mixed workloads use 254 classical coefficients with literal high and
low one bits, plus a 256-wire offset and 256 clean wires. The constant
workload uses the source prime ending in `FFFFFC2F`; the source's
`approx` benchmark passes infinite bit tolerance. Input layout construction
is outside the timed region.

| Workload | Raw source records | Raw native records | Matched effective quantum effects | HMR on each side | CCX on each side | Source generation | Native generation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mixed add | 13,434 | 13,119 | 11,588 | 785 | 1,295 | 40 μs | 184.946 ms |
| Mixed subtract | 15,697 | 15,641 | 14,111 | 1,037 | 1,545 | 54 μs | 241.813 ms |
| Constant add | 13,368 | 11,529 | 11,522 | 785 | 1,295 | 27 μs | 132.290 ms |

Raw source records include block marks and exclude register declarations.
The native representation retains scoped execution and omits resets whose
precondition already guarantees a zero target. Source resets number
784, 1,036 and 784 respectively; the native emitter retains 1, 2 and 1.
These counts are physical records, separate from expanded runtime
instructions or a fault-tolerance resource model. Source figures time C++
builder generation; native figures are the median of seven Python
generation and artifact validation runs after one warmup, measured with
other local checks active.

The effective-stream certificate handles classical operations by evaluating
their Boolean expressions over independent input and X-measurement
variables. It snapshots the predicate at scope entry and keeps the previous
bit value on inactive HMR lanes. It compares every remaining quantum
opcode, ordered operand and exact symbolic guard. This establishes that the
different `BIT_STORE0` destinations and explicit versus implicit zero
initialization do not change any effective quantum condition on these
workloads. One test perturbs a live initialization and another changes a
guard after its scope is entered. The certificate distinguishes both
changes. Raw byte identity is not asserted.

Run the benchmarks or regenerate the source fixture:

```sh
python -m benchmarks.benchmark_mixed_modular_addition
python tests/unit/fixtures/generate_kickmix_modular_addition_workloads.py \
  --source-root <locked-kickmix-root> \
  --output tests/unit/fixtures/kickmix_modular_addition_workloads.json
```
