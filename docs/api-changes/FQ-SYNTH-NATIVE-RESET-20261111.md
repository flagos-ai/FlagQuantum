# Native scalar RESET admission

Status: scalar admission implemented and locally verified under the ongoing
user-directed native migration. Ten new tests plus twenty-three related/API
tests pass; strict typing covers eight changed Python files, and repository
quality checks pass. Thermal corpus coverage remains pending at 207/216. Classification:
`single_device_fast_path`. This is an unreleased domain-extension candidate,
not a Stable Core signature change or a release approval.

Add `RESET(target: QBit, /) -> None` to the existing synth and Services facades,
retaining the Qmod spelling. The captured operation lowers to the existing
`qreg.reset` effect, verifier, amplitude-damping-at-one channel and density
matrix execution authority. It prepares the target in zero and traces its prior
state, including entanglement; it is not an X gate, uncompute, free, hidden
postselection or pure-state replacement. Allocation and register ownership stay
unchanged. Calls outside captured programs remain refused.

Accept initialized writable scalar bits and static array elements, including
borrowed writable helper parameters. Refuse whole arrays, Const and uninitialized
storage. Existing pure-effect validation must refuse reset under adjoint,
within/apply and coherent control. Inverse contexts cannot silently discard a
reset or treat it as identity. Actual density matrix memory and channel counts
must be charged; do not label a noisy wide circuit as exact interval MPS.

Acceptance includes reset of half a Bell pair yielding the correct mixed
state, nonzero coherent inputs, repeated reset, helper composition, early
ownership/context refusals and memory/operation limits. Preserve ordinary
unitary examples and Qmod RESET semantics. This admission alone does not accept
the thermal example: its fifty complete delta cycles still require compact
static-channel replay and independent complete density-matrix validation.

The new candidate contract and active export/signature checker precede Compiler
and Services implementation. Preserve all historical contracts. No new
numerical kernel, executor or Runtime public option is introduced. Full216 and
approved-six replay, production pin disclosure, catalog and ledger updates are
required before the thermal example can count toward coverage.
