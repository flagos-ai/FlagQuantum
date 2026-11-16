# Native unsigned modulo by powers of two

The user authorized missing native language capabilities for the five-category
example migration. This narrow unreleased extension admits `expression % d`
where d is a literal positive power of two at most 2**256, including one. The
left operand has unsigned integer semantics. The compiler rewrites this to the
existing finite-width `qint.bitwise` AND with d-1; it creates no new arithmetic
kernel, serialized opcode, runtime path or public export. Stable Core and frozen
historical snapshots are unchanged. Release authorization remains false. This
is the `single_device_fast_path`.

Quantum divisors, nonliteral divisors, zero, negative divisors and non-powers of
two are refused. General remainder/division and signed/fractional remainder are
not admitted. The result is the exact nonnegative remainder, with the existing
AND width inference; modulo one uses a one-bit zero encoding. Fresh assignment
ownership, mutation dependencies and all construction/execution budgets remain
in force. The pure-expression frontend and typed qfunc share this translation.

## Acceptance

- Migrate both retained arithmetic-expression paths and the original modulo
  example with its five-bit input domains. Preserve all original bytes test-side.
- Compare raw complex statevectors with independently calculated Circuit XOR
  permutations, covering every small input pair, divisor-one and wider-mask
  boundaries, nested arithmetic and both workspace schedules.
- Reject unsupported divisors and unchanged ownership/budget failures. Preserve
  all 216 frozen original source identities and resource records.

## Static typing of fixed-width examples

`Output[Annotated[QNum, 5]]` is admitted as the standard typing spelling of the
existing five-bit unsigned domain. `QNum[5]` remains accepted by capture. Width
metadata must be one literal accepted by the existing width validator; unknown
metadata, QBit/QArray payloads and extra metadata are refused. This lets examples
retain source domains while passing strict Python type checks. No representation
or domain semantics change. The same domain is checked under both spellings in
the acceptance tests, with malformed metadata rejected at decoration time. QNum equality annotation methods explicitly retain
Python's object-to-bool signature; capture still produces a quantum comparison,
never a host Boolean. These methods only raise if invoked outside capture.
