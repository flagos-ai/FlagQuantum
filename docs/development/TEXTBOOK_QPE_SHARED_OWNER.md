# Textbook QPE constructor transformation ownership

The complete frozen phase_estimation/text_book_qpe.py and its test file retain
[their original source disposition](RESOLVER_TEXTBOOK_QPE_AUDIT.md). The pure
stored constructor transformation now lives in Compiler formats under
legacy_textbook_qpe_factories.py. The obsolete private Ecosystem module is
removed and all repository consumers import the authoritative replacement.

Per-compilation records retain exact ZPow, rectangular preparation, Fourier
reversal/direction and QPE adjoint context. They are not a process registry.
Unknown same-signature children retain their explicit refusal. A new test proves
that children created in one factory mapping cannot be used through another
mapping. Compiler neither imports numerical or Algorithms owners nor chooses an
execution backend. Source/SDK/Algorithms/Simulation-blocked resolution is tested
on the actual serialized small source example.

Before moving the owner, 32 complete Program and bound-IR records were captured
from published ac3b1c38b. They cover every width-one/two/three configuration of
both Fourier reversal settings and both Fourier and QPE directions, plus all
four original concrete/symbolic examples in both directions. Every operation,
operand, parameter, feedback, helper, output alias and lifetime record must remain
identical after replacement. Existing source recursive-call, full absolute
complex isometry, prepared-image inverse and work-zero evidence is reused only
after that full-record identity proof. Applicable raw histories remain exhaustive
at widths one/two; width three retains its recorded independent certificate and
zero-history inverse check. There is no larger inverse-channel claim.

The phase output remains RIGHT QFxp(m,m), MSB-first, owned by the caller. Targets
remain THRU. Powers traverse phase bits in reverse order, one/two/four and so on.
Inverse LEFT phase storage retains the prepared-image clean-release promise;
it is neither arbitrary projection nor reset. Unknown exact adjoints, arbitrary
unitary/window constructor contexts, atomic Kaiser bodies, symbolic physical
execution and pending scalar behavior retain their existing separate scope.

The 48 original directional resource rows are rerun independently of full-record
comparison. Primitive instructions, measured feedback and physical width stay
separate from source QEC models and logical peaks. For the default three-bit
forward example, physical counts are 40 T-family, 60 CX, 36 H, 10 phase gates and
10 measurements on five wires. The source model instead has one T, ten And,
19 Clifford, seven rotations and ten measurements with a five-qubit logical
peak. Phase output is logical storage; the private scratch bit returns zero.
This ownership migration does not alter those counts or add parent acceptance.

Final concentrated regression passes 137 cases in each Cirq 1.7/1.6 profile;
145 local integration cases pass. Strict typing covers both changed Python
files; Black/Ruff, owner scopes, English/link, architecture, IR/API and diff
checks pass. Dependency policy remains nonzero repository-wide at 735 findings,
with none on the changed implementation or test. This is not a claim that the
whole repository dependency policy passes. Frozen source locks and actual
Kickmix 0.1 Simulator availability are verified by the 48-row resource receipt.
