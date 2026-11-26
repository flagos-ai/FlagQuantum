# Shared phase-window transformation ownership

The complete locked phase_estimation/qpe_window_state.py and
kaiser_window_state.py dependencies retain their
[constructor disposition](RESOLVER_PHASE_WINDOW_AUDIT.md). Compiler now owns
rectangular_window_program in registers/phase/window_component.py and the two
stored constructors in formats/legacy_phase_window_factories.py. The obsolete
private Ecosystem window owner is removed and every direct consumer is updated.
Algorithms retains the existing rectangular_window_program signature as a
thin delegate, plus its mathematical declarations and user binding interface.
Kaiser coefficients and numerical execution remain in their existing owners.

The RIGHT qpe_reg port retains unsigned fractional storage QFxp(width,width),
MSB order and caller ownership of the allocated output. The literal rectangular
Program allocates once and invokes the existing OnEach/H dependency. Symbolic
widths retain a signature-only OnEach declaration. Negative rectangular widths
retain their refusal. Kaiser validates its original alpha metadata and declares
the same output without a body. It does not invoke its tensor or approximate
rotation child during parsing.

Before moving code, eighteen complete Program/bound-IR/Kaiser declaration
records were captured from published cccc4c295. The replacement must equal every
record, including operation/value IDs, dependency names, parameters, output
aliases and lifetime ownership. The records cover zero, positive and symbolic
rectangular widths, reversed physical allocation and a borrowed live carrier;
Kaiser covers zero/positive/symbolic widths and negative/zero/positive/symbolic
alpha. Source constructor graph, full complex state/operator, clean output and
append evidence is reused only after that complete-record equality. Dedicated
checks refuse Algorithms, Simulation and SDK imports during Compiler resolution
and replace the shared Program owner without modifying its Algorithms consumer.

Measured rectangular width n emits n H, no private work, measurements, reset or
feedback. The original model independently reports n Clifford and n output
wires. Kaiser has no original physical body and thus no measured primitive
count. Its coefficient tensor and approximate preparation model remain separate.
Original zero-width tensor failure, Kaiser small-precision/overflow failures,
symbolic physical limits and the existing controlled/scalar boundaries are
unchanged. No protected public API or frozen source is changed; constructor
ownership migration alone adds no complete-parent acceptance.

Final verification passes 166 concentrated cases in each Cirq 1.7/1.6 profile,
145 local integration cases, strict typing of six Python files, Black/Ruff,
Compiler/Algorithms/Ecosystem scopes, English/link, architecture, IR/API and
diff checks. Seven frozen source blobs and actual Kickmix 0.1 Simulator are
verified. Five zero/positive-width resource rows retain separate primitive and
source-model categories. The eighteen complete baseline records match.
