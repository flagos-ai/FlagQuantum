# Basis-aware register controls

K-0185 and K-0186 cover the locked `control_helper.pybind.cc` and header
at blobs `59ae141218b4d980fd15d4991c23f38fbe12a57d` and
`e40f74210fba4b4d2c63f739442759825bb414f2`. The compiler binding
normalizes the source operand families before invoking a child emitter.
It reuses the existing wire, classical slot and X-basis representations.

| Source behavior | Native binding |
| --- | --- |
| Z quantum wire and true/false literals | Pass the wire or true to the child. A false enable produces an empty body. |
| Z classical bit | Pass the true sentinel and retain the classical enable. The typed native boundary takes Boolean literals explicitly and interprets an integer as a wire index. |
| X quantum output | Pass the target wire to the child. |
| X plus and minus states | Skip for plus. Select scalar/controlled phase output for minus. |
| X classical bit | Select phase output and guard it on the same classical slot after the source's basis reinterpretation. |
| Constructor, conversion and destructor | A resolved native input is immutable. The body owns its conditions. Guard lifetime ends with the body, including when admission fails before construction. |
| Deleted copy, move construction and move assignment | Native code has no mutable guard owner to copy or move. Source move construction clears the moved-from owner. Source move assignment also clears the moved-from owner and handles self-assignment. Assigning over a live destination replaces its owner without first balancing its old push. That source edge is outside the native value interface. |

The equal-comparison binding applies both outer guards to the full body
including HMR and its correction. The result slot must be distinct from
the guards and defined before guarded measurement, so an inactive branch
preserves its previous value. Two identical guard reads form one condition.
Rotation reuses the same guard composition and retains its specific entry
snapshot for an enable whose storage is temporarily overwritten.
Previously accepted field addition supplies the independent Z-enable
consumer evidence.

The oracle captures all 16 controls/output combinations from locked
`flip_if_equal`. Ordered primitive operands and effective guards match
through the native append path. Source reset markers establish the clean
work contract. The native caller supplies those two wires in zero.
Source push/pop records and runtime bookkeeping are counted separately
from the effective gate stream. The comparison uses up to two HMR
operations and one reusable result slot. The control binding creates no
quantum workspace and declares no analytical resource model.

For each applicable classical value, the native certificate exhausts the
two-HMR histories and checks the signed basis mapping, including both
work wires returning to zero. Local append execution compares complete
complex amplitudes and preserves the guard inputs. The unchanged source
also balances a classical X scope when the following Z-enable conversion
raises. A subsequent unrelated gate remains outside the scope.

Run `python -m pytest -q tests/unit/test_kickmix_control_binding.py` for
the fixture and `python -m examples.algorithms.register_bit_rotation`
for the existing append example.
