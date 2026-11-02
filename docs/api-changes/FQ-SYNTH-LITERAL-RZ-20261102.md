# Literal RZ and paired FlagQuantum-native workflows

Follow #1002 at 6df1f2ae6296334d087fdf39918f8b1e46559a19. Run the pinned
RZ.qmod unchanged, with original-size file/text full-state verification. Target
coverage is eight original paths, seven distinct blobs, only after execution.

FlagQuantum-native high-level usage is primary: an ordinary Python radians
function passed to synth.synthesize, without gates or physical indices.
Gate-level fq.Circuit and Qmod migration are separate reference baselines.
Compare all three on basis and coherent states, including global phase; also
check native gate/Qmod reference-entangled states. The first high-level profile
supports known basis/uniform preparation only, not unknown entangled inputs.
The same synth.synthesize entry gains a bounded native high-level profile:
output_mode="phase" plus an explicit float return annotation means radians,
with one UInt(1) parameter and arithmetic literals/name/+/-/*/unary signs only.
Existing bool/int/unannotated phase functions retain their pi-predicate meaning.
No function body is executed: the shared source reader checks both values of
the complete one-bit domain through a bounded AST interpreter. This is complete
one-bit diagonal synthesis, not state-dependent sampling or a general real
arithmetic engine. Preserve integer arithmetic exactly and float64 operation
order; convert validated final phases to float64. Require finite literals/intermediates/phases with magnitude
<=2**53 and depth<=32. Reject helpers, constants bindings, assignments, loops,
custom preparation IR and wider/multiple inputs in this first native profile.
Use existing named basis/uniform InputState preparation. Pure constant phases
are valid and global phase is preserved. Select an RZ for exact antisymmetric
basis phases, a PHASE for zero phase on basis zero, otherwise use existing X
and PHASE gates to preserve both basis phases. No tolerance-based simplification.
Reuse existing register/effect HybridProgram, gate verifier, selection, checked
Program, preparation validation, memory planning and named result reader. No
new public export, argument, mode value, quantum real-data type or execution path.
Report output_mode=phase and result_width=0; logical inputs are the only outputs.
A wider/multi-input/general real phase language remains a gap. Each future
compatibility feature must document its native high-level counterpart or gap.

Compiler reuses the existing qreg.gate angle contract for phase and rz, with one
private closed gate set: exact gate/index/angle, finite Python float radians,
initialized target, validated index, unchanged effects/Const/mutation/budgets.
Emit existing Core rz with named theta. RZ(theta)=diag(exp(-i*theta/2),
exp(i*theta/2)); it differs from PHASE by a global phase. No new opcode, public
API, gate matrix, allocator, executor or numerical kernel.

Ecosystem admits nonnegative integer or contiguous fixed-decimal angle literals
only as first arguments to PHASE/RZ. Reuse existing tokens and expression records;
do not add decimal data arithmetic or an alternate IR. Max literal length is 78
characters. Compare the decimal value exactly against 2**53 before float64
conversion, preventing rounded boundary overshoots. Decimal angles have ordinary
float64 numerical semantics; integer values in range still convert exactly.
Preserve parenthesized integer PHASE inputs. Negative, exponent notation,
leading/trailing decimal point, whitespace/comment-split decimal, symbolic,
computed, quantum, nonfinite and out-of-range angles remain refused. RZ becomes
a reserved intrinsic. PHASE decimal admission deliberately replaces its previous
literal-decimal refusal; old non-decimal behavior is preserved and tested.

Acceptance: native/Qmod/analytic amplitude agreement, unmodified original
source/license pins, both objectives, global/relative phases, helper composition,
old integer grammar and phase behavior, malformed IR and source numeric/role/
scope/shadowing guards, exact decimal bounds, tiny nonzero angles, supported
named basis/routing and explicit unsupported-target refusal, actual resources.
No general CReal expressions, native high-level real phase API or hardware claim.
Trials remain 0/2 and release_authorized=false.
