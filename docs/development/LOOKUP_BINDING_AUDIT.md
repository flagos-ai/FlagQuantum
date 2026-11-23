# Complete lookup Python binding intake

## Frozen scope and module ownership

K-0232 is `src/kickmix/py/circuit/subs/lookup.pybind.cc` at Kickmix
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, blob
`227392313663d4c2c06987e47d93c226869bf4be`. K-0233 is its complete Python test
file, blob `9ce2ab152b0c4e533297f363feb5a8a93b1f1ecc`. Both local source blobs
were verified against the immutable inventory before implementation.

`compiler/registers/lookup/lookup_binding.py` owns typed argument binding,
full-table admission and initialization boundaries. Existing `recursive_lookup`
and `measured_unlookup` own physical schedules and component traces. Existing
layout utilities bind native quantum operands and encode integer words. Existing
lowering, Runtime and Simulation own execution. This path is classified as
`single_device_fast_path`.

All native words are MSB-first. Each source LSB-first word is reversed once at
its boundary. The caller owns target storage and supplies clean workspace in
source lease order, with one lease per address bit. Initialization resets target
wires in source order before entering the lookup mark. Erasure consumes computed
target values and returns target and temporary workspace in zero. Phase slots
start after declared external classical storage. A reused prefix records the
source clean-bit allocator's additional zero stores. No new allocation API or
serialized vocabulary is introduced.

## Complete definition mapping

| Original definition or dependency | Native owner and disposition |
| --- | --- |
| `append_init_lookup` | `lookup_binding(erase=False)` binds quantum words, checks the complete table, prefixes target resets and reuses the exact recursive lookup schedule. |
| `append_del_lookup` | `lookup_binding(erase=True)` reuses measured unlookup, unary conversion, recursive phase lookup and source feedback. |
| `read_python_int_list_into_table_data` | The binding indexes entries from zero through `len(table)-1`, applies Python `int`, checks the converted nonnegative target range and uses shared big-endian integer encoding. Full utility-file disposition remains independent. |
| `ConvertedArrayXZ` | Accepted shared native operand binding supplies typed/sliced quantum words. Foreign register objects remain outside the native Compiler. |
| `gen_lookup` / `gen_unlookup` | Accepted recursive lookup and measured unlookup owners are reused unchanged. |
| `test_init_lookup_diagram` | The original diagram assertion passes unchanged. Full marked stream comparison checks every operation and operand independently of diagram rendering. |
| `test_fuzz_init_lookup` | All original 16 parameterized cases pass unchanged. Native forward streams cover address widths zero through six and target widths zero through four. |
| `test_fuzz_del_lookup` | All original 16 parameterized cases pass unchanged. Native erase and joined streams cover all source-success cases in the same range. |

## Evidence

The fixture generator requires actual locked Kickmix version `0.1`, verifies its
source blob and checks that `Simulator` exists. It compares 1,030 complete marked
streams byte-for-byte after removing only register declarations. Cases cover
zero, full and mixed tables, string and floating-point integer conversion,
fresh/reused clean pools, declared unused classical slots, initialization,
erasure and joined calls. Full KMX marked roundtrips and primitive histograms
are checked against those source receipts.

Native branch checks exhaust address widths zero through two, target widths one
and two and reversed physical layouts. They use complex address/spectator
superpositions, initially dirty targets, every reset and erasure history, raw
unnormalized amplitudes including reset-dependent global signs, total channel
probability and zero support on dirty target/work lanes. Existing deeper
recursive lookup and measured-unlookup branch evidence is reused. A normal
`DynamicCircuit.append(FieldOperation(...))` initializes and erases a table through
existing lowering and Runtime.

The original 33 tests pass unchanged. Source admission probes also cover bool,
float truncation, strings, bytes, NumPy integers, `__int__`, `__index__`, invalid
ranges and conversion errors. Native admission retains the same integer
conversion and indexed-table behavior. Invalid native construction returns no
partially mutated Program.

## Resource accounts

Observed primitive histograms are stored per complete fixture case. Component
entry/exit INFO_TAG records are counted separately from physical primitives.
Workspace consists of the wrapper's address-width clean lease pool, with phase
slots and one reusable measurement slot on erasure. The source Python wrappers
provide no independent analytical model. Existing QROM/QROAM logical estimates
remain separate from these concrete recursive lookup primitives and feedback.

## Source defects, adaptations and remaining boundaries

- The original six-address-bit, one-output-bit erasure exhausts its clean pool.
  `gen_unlookup` needs four unfolding leases and three recursive lookup leases,
  while the wrapper allocates six. The native wrapper retains the same failure
  with its six-lease pool. This is a recorded source defect, so K-0232 remains
  open. No source correction is substituted into parity evidence.
- Both original wrappers allocate clean qubits before table validation and omit
  cleanup on failure. Eight invalid-table probes show the next allocation is
  q3 after address q0 and target q1, retaining the leaked q2. Native immutable
  construction rejects the input without mutating caller storage. This failure
  atomicity adaptation is explicit.
- Source Python admission permits up to 32 address bits. Empty output returns
  before the recursive C++ shift, and native empty-output admission retains this
  behavior through 32. Nonempty kernels retain their accepted portable 30-bit
  limit. Dense 31/32-bit tables have not been allocated or claimed equivalent.
- Empty targets accept any number of zero entries because original table length
  validation multiplies both sides by zero. Native behavior retains this case.
- Target allocation stays caller-owned. Original bindings require an explicit
  target and expose no `target="alloc"` route here.

K-0233's complete original tests and native replacements are accepted. K-0232
retains the exact work-shortfall and large-table boundaries above. Pending API
Proposals 116 through 119 remain outside this delivery.

## Batch verification

All 499 concentrated lookup/dependency checks and 29 local integration checks
pass. Strict typing covers all three changed Python files. Black, Ruff,
Integration team scope, architecture, internal IR, public API migration and
whitespace checks pass. Type annotations in the fixture generator were corrected
and strict typing rerun after the concentrated regression. No physical schedule
changed in that repair.

## Complete conversion-utility follow-up

The complete utility audit now records cross-byte target widths through 129,
actual wide packed execution and a sequential conversion failure-order repair.
Ordinary int/string/float/bytes conversion retains the established behavior.
The source dynamically honors int-subclass bit_length/to_bytes overrides,
while the native owner normalizes these values first. This explicit source
boundary keeps K-0183 open. See LOOKUP_TABLE_CONVERSION_AUDIT.md and the
immutable source conversion observations. The existing K-0232 source-workspace
and address-portability boundaries remain independently open.
