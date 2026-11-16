# Shared reset-cycle templates and native static replay

Status: implementation contract; native coverage remains 207/216.
Classification: `single_device_fast_path`.

Thermal's fifty native delta cycles cannot be naively expanded past the current
4096-effect construction budget. The Qmod adapter already packs an invariant
reset cycle into the checked `qreg.replay` representation. Extract only its
owned-IR template construction into Compiler and demonstrate replacement of the
Qmod constructor without changing importer consumers, source outcomes, resource
counts or numerical execution. No source syntax or SDK object enters Compiler.

Then admit ordinary Python `for ... in range(...)` loops with a loop-independent
closed reset cycle by capturing one checked cycle and referencing that same
owned template. Reuse existing effect verification and lowering. At least one
actual reset must be present, and a cycle remains limited to 512 effects.
Keep the existing static range/count, definition, alias, ownership, instruction,
work, depth and memory constraints. Expanded runtime operations/channels count
in full. A template must not allocate, release, drop or escape storage; targets
must already be initialized writable disjoint owners. No new public function,
option, numerical kernel, executor or arbitrary Python execution is introduced.

Index-dependent or unsupported loops retain existing checked expansion/refusal;
never treat their bodies as invariant. Nested templates, effectful inverse or
coherent control, changing classical bindings and empty/non-reset cycles cannot
silently qualify. Preserve exact density evolution and reset epoch semantics.

Acceptance requires Qmod replacement tests with unchanged complete thermal
results/resources, generic native repeated Bell-reset density comparisons,
cycle identity, invariance and ownership negative cases, and operation/channel,
state memory and construction budget refusal. Only full original fifty-round
thermal native/Circuit/Kraus acceptance, fresh216/approved6 replay and evidence
updates may increase corpus coverage. This contract alone earns no acceptance.
