# Native quantum helpers with fresh scalar outputs

Date: 2026-11-01. Continue the sixth plan after #985 (57 successful checks,
three configured skips; merge 03ddcc63fcd24d0fee04efe576bfdafd3c8fb172).
This Ecosystem slice admits composition using the existing fresh-output
Compiler contract. No public signatures, exports, schemas or Compiler operation
contracts change. The Python-first user interface remains unchanged.

## Semantics

A native source may contain at most 32 named qfunc definitions, exactly one
named main. Other definitions accept only the already admitted output scalar
parameters. Every output must start fresh and return initialized. Helpers may
compose existing allocation, gates, arithmetic, sign phases and other helpers;
there are no locals or captured external names. Calls use positional output
variables only, with distinct actual storage, exact qbit/qnum kinds and exact
resolved widths. Bare qnum widths may be inferred from a verified helper's
output, but a conflicting caller declaration must be refused without truncation
or widening. Prepared outputs are transferred to the caller, not cloned.

Compile every definition, including unused ones, using the existing owned
register model and Compiler verifier. Resolve an acyclic call graph independent
of definition order; reject recursion, missing/duplicate main, duplicate symbols,
intrinsic redefinition, shadowed helper calls, unknown callees and unsupported
roles. Calls to main are not helpers. Intrinsic spellings keep their existing
meaning. Each definition is checked independently before being used to infer a
caller output: an invalid helper must not become valid by borrowing caller state.

Inline only already verified register effects, substituting formal output
references with caller references and threading the existing linear effect.
Reuse immutable arithmetic callees under definition-qualified names and dedupe
them across repeated calls. Do not substitute source strings, evaluate source,
invent another canonical call IR, or add a register-role manager. Run the
existing Compiler verifier again on the complete main. Its initialization checks
reject passing an initialized actual, and its node/width budgets still apply.
The existing 4096-effect expansion cap applies to every expanded definition and
the final main; source byte/token/depth limits remain unchanged. Call-site
coordinates locate expanded effects; arithmetic definitions retain their own
source coordinates.

Output here is a fresh storage obligation, not Input, Const, classical const,
arbitrary mutable borrowing or automatic destruction. General role combinations,
arrays/views, higher-order functions, locals, free/drop and within/apply remain
outside admission. Native sign phases keep #985's one-bit/pi/logical-target
boundary. No generic Classiq or all-216 compatibility claim is added.

## Reuse and acceptance

Ecosystem owns the source definition table, bounded call resolution and owned
effect substitution. Compiler keeps freshness, complete initialization, phase
cleanup, arithmetic schedules, target checks and selection. Existing Services,
Program, Runtime and named result reading remain unchanged. No new dependency,
public class, serializer or execution path is needed.

Prove forward/nested/repeated calls, formal-name isolation, inferred/exact widths,
invalid unused definitions, recursion and expansion refusal. Compare helper and
flat producers through the same unchanged execution consumer. Check complete
complex amplitudes, phase/interference, correlated outputs, zero workspace,
both schedules, text/Path, source metadata, plan identity and failures before
planning. Give a complete gate-free user example and actual resources. Preserve
original source/license hashes and the 3/3 count; independent trials remain 0/2
and release_authorized=false.
