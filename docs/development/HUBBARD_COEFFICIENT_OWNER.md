# Shared Hubbard coefficient numerical ownership

The complete frozen prepare_hubbard.py/test pair at Qualtran v0.7.0 commit
bcb17bb25360ed68ce3bc084645db3e929799389 retains
[its full constructor disposition](HUBBARD_PREPARATION_CONSTRUCTOR_AUDIT.md).
Simulation numerics/hubbard_coefficients.py now owns normalization and the two
preparation angles. Existing Algorithms private helper signatures delegate to
that owner. Ecosystem coefficient conversion calls the numerical owner directly,
removing its last Algorithms import from packed constructor conversion.
Compiler continues receiving owned angles through the existing callback.

The source normalization is 2*N*t + (N*u)//2 with N=2*dimension**2. Integer
coefficients reach that arithmetic before any float conversion. The first angle
retains t*N numerator order, math.sqrt and NumPy arccos; the conditional angle
retains NumPy sqrt(1/5) and arccos. No algebraic reassociation, clipping, dtype
coercion, extra validation or approximation is introduced. Numerical warnings
and zero-division/domain failures remain with their original operations.
The physical operation's nonnegative finite-weight validator and the interchange
converter's finite-angle/symbolic rules remain distinct and unchanged.

Before edits, 70 complete numerical/warning/refusal/primitive-stream/model
records were captured from published 78fac06cd. They include dimensions zero,
one, two, three and nine; fractional/end-point/negative/large-integer weights;
zero denominators, nonfinite values and invalid square roots. Twenty records
retain complete forward/inverse emitted gates, angles, operands, feedback and
source models. All records must match after migration. The prior 22 complete
packed/bare constructor Program/binding/failure records are also compared, so
existing full complex amplitude, measurement-history, logical-junk and private
work-zero evidence is reused only after identity proof.

The new direct source checks consume the actual first two yielded rotation
operations and compare their literal radian parameters, normalization value and
runtime numeric type. They cover both negative source coefficients with real
angles and integers above exact float precision. Replacement checks demonstrate
that Algorithms delegates to the numerical owner and that Ecosystem can reload
and convert coefficients with Algorithms imports blocked. Normalization needs
no NumPy; angle evaluation retains its lazy import failure.

This numerical module emits no gates or model nodes itself. Existing PREPARE
and packed primitive/model inventories remain separately owned and unchanged.
Atomic ModAddK, singleton source bodies, source enum serialization, symbolic
physical execution and broader scalar boundaries remain as recorded in the
constructor audit. No protected public interface or reference source changes.

Final concentrated regression passes 72 cases in each Cirq 1.7/1.6 profile and
145 local integration cases. The 22 direct numerical/source checks pass again
after optional-dependency guards, without skips. Strict typing covers four Python
files; Black/Ruff, Algorithms/Ecosystem/Simulation scopes, English/link,
architecture, IR/API and diff checks pass. Eight source/test blobs and real
Kickmix 0.1 Simulator are verified; the retained source fixture also checks its
notebook blob. Four directional packed primitive/model resource rows are
unchanged. Repository dependency policy remains nonzero at 733 findings, with
none on the batch. This is not a repository-wide dependency-policy pass.
