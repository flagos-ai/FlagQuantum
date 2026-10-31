# Predicate phase public implementation authorization

Date: 2026-10-31. After PR #974 merged as
1afca2cbc9128f070b6637f75b2818e0f6b2ae61, the human asked what to do next.
The response proposed implementing output_mode="phase" on the existing
synthesize / Program / run flow, limited to Python one-bit predicates and native
logical CPU targets, with full phase, workspace, interference and resource
acceptance. The human explicitly replied "do". This authorizes that concrete
additive API and its name; it is not an invented GitHub CODEOWNER review or
independent developer trial. Repository review and release gates still apply.

The [historical proposal](FQ-SYNTH-PREDICATE-PHASE-20261031.md) and its JSON
remain byte-for-byte unchanged. The separate
[implementation contract](../../contracts/synthesis-phase-v1-candidate.json)
records activation. The active Program/run candidate signature advances by
exactly the approved final keyword; the historical Qmod admission stays intact
and its source/value behavior remains protected. No baseline snapshot is
regenerated, and no root export, run/read signature, executor or serialization
changes.

Value remains the default. Phase requires a one-bit unsigned return; it does not
coerce arbitrary integers to truth values. Native string/file phase and explicit
basis/topology requests fail before capture, source acquisition or planning.
Preparation remains outside the oracle. Reuse the same selected forward prefix
and literal inverse, existing constraints, plan validation and named reader.
The public phase report has input registers only and omits arithmetic_resources.

Independent developer trials remain 0/2 and release authorization remains false.
Native Qmod phase, general routed phase, arbitrary angles, controlled-block DSL,
GPU/QPU/gradient admission and full-corpus compatibility are outside this change.
