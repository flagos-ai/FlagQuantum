# Complete distillation channel utilities

Q-0927/Q-0928 cover the full frozen source and original tests at Qualtran v0.7.0
revision `bcb17bb25360ed68ce3bc084645db3e929799389`.
The source blob is `934703ab8dc9e7cde7df42b50ba4892a265dd605` and original test blob
is `f40e12d12434db519cb200ae9bbbf5572ffa8846`.

The existing `qec/resources/distillation_channels.py` already owns both channel
constructors. Source `_rotation_unitary` is implemented directly in the native
rotation builder using the authoritative Pauli basis. The Cirq gate object is
represented by an existing native channel instruction. Its arity is the explicit
wire tuple, and its ordered mixture is encoded as four complex Kraus operators.
There is no second gate class, SDK adapter or simulation implementation.
Source Cirq wire-symbol rendering is an SDK presentation protocol. Native IR
retains explicit operands and its channel name. Cirq diagram text is not a
native display-parity claim.

The complete four-branch order is 1, 5, -1, 3 times pi/8 with the source's positive
imaginary exponent. All Pauli words of widths one through three are checked,
including Y and identity factors, zero-error, each deterministic error branch,
and a nontrivial mixture. Full complex matrices use absolute tolerance 1e-15
and zero relative tolerance. The four mixture entries are retained even at zero
probability.

Storage uses existing independent X/Z channels. Native channels omit Cirq's
extra zero-probability Pauli slots. This is an explicit representation difference,
with Kraus and full density semantics already covered. Empty schedules, invalid
kinds and strict probability/wire zip failures are compared. Native rotation
admission requires a nonempty Pauli word with matching wire count and valid
probabilities. The source mixture object itself admits invalid weights, while
its execution protocol later fails. These validation timing differences are
retained and separately tested by the existing focused suite.

Combined noisy rotations and storage execute through the normal native density
executor on wires (3, 0), with complex coherent inputs. Every density entry is
compared with the original simulator at absolute tolerance 2e-15 and zero relative
tolerance. Untouched wires remain exactly zero in every affected row and column.
The schedule emits one channel instruction per rotation and one per storage
operand. Three preparation gates plus one rotation and two storage channels
produce six native instructions. These primitive channel counts are distinct
from the existing factory's analytical T-state demand and physical estimates.
Kraus alternatives are incoherent noise branches, not measurement-feedback
histories or clean unitary branches.

The original two tests are executed unchanged. Existing 100-operation factory
schedule/Kraus comparisons, complete five-qubit density outputs, postselection
and consumer replacement evidence are reused without changing implementations.

All 423 concentrated resource regressions and both unchanged original tests pass.
The added complete-file evidence passes strict mypy, Black and Ruff. Core scope,
architecture, IR and protected API checks pass. No production changes are needed
because the previously accepted native constructors already implement the full
source's executable channel behavior.
