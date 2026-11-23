# Complete Kaliski constructor dependency chain

## Source and ownership

The complete mod_arithmetic/mod_division.py and its unchanged original tests
are locked to Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Existing
whole-file and layer acceptance evidence is reused. The format owner
legacy_modular_inverse_factories.py covers all nine classes. Steps 1 through 6
delegate respectively to inverse_termination_component, inverse_parity_component,
inverse_ordering_component, inverse_exchange_component,
inverse_arithmetic_component and inverse_scaling_component. The iteration,
algorithm and public wrapper delegate inverse_iteration_component,
inverse_algorithm_component and modular_inverse_component. The adapter owns
constructor and ordered-signature validation only.

All word storage is MSB first. Steps 1 through 5 use width-only Montgomery
words. Step 6, iteration and algorithm use modulus-bearing words. Individual
step flags are through-going logical inputs. The iteration owns its temporary
a/b flags and cleans them only on source-reachable inputs. The algorithm retains
two independent 2n-bit histories. The wrapper joins these into a 4n-bit RIGHT
output, or consumes the matching LEFT input when uncompute is true. Exact native
owner equality preserves physical result aliases and their clean-work ownership.
Concrete consumers retain circuit.append(op). Every symbolic constructor keeps
a bodyless signature and its original ordered dtype, width and side metadata.

## Original wire and explicit construction

Steps 1 through 5 resolve from the original serialized library and reproduce
the unchanged source primitive gates, operands, measurement identifiers and
feedback. Each equals its previously certified owner after physical binding.
The original built-in resolver dictionary omits these private classes, so their
explicit native constructor mapping is distinct from upstream automatic loading.

The last four classes reconstruct Montgomery modulus metadata from their mod
argument. The original dtype codec drops that metadata in stored signatures.
Original-wire resolution refuses the mismatch. Explicit native constructor
resolution preserves the full source signature and equals the existing owner.
The public inverse identifier remains partial. The dtype schema and generic
resolver retain their accepted contracts. Upstream wrapper uncompute is encoded
as integer zero/one in the wire, and either encoding normalizes to the existing
boolean owner direction. Other direction values refuse.

## Phase, history and resources

The accepted complete-source layer streams and raw-history certificates remain
applicable through exact bound Program equality. Existing inverse wrapper tests
cover every unit and zero in the recorded residue grids with complete complex
spectator amplitudes, retained history, three selected primitive trajectories,
certified And channels and clean returned work. These scopes remain separate
from an exhaustive enumeration of the whole inverse's raw histories.

Actual exact swaps consume seven T gates per bit. Independent upstream symbolic
QEC formulas, including 26n squared plus 9n minus 1 Toffoli equivalents, retain
their original category assumptions. Emitted primitive counts are scanned
separately. The source p+1 loader failure at an all-ones modulus remains a
refusal, with existing nonunit/exterior and unreachable-flag witnesses preserved.
Generic Adjoint and Controlled dispatch remain separate open work.

This complete constructor disposition adds no complete built-in identifier.
Local catalogue coverage remains 77/303, pending publication. Whole-file
acceptance and Q-0876 retain their previously registered generic, domain and
wire boundaries. Receipts use ftqc-reference-env/g2-modular-inverse-constructor-*.

## Local batch acceptance

The concentrated native regression passes 218 cases, including all 23 focused
constructor cases. Local integration passes 145 cases. The unchanged original
source file passes 54 cases, with one original Montgomery metadata skip, one
expected count failure and one notebook deselected. The complete earlier source
intake including its notebook remains reusable. Seven frozen blobs and the real
Kickmix 0.1 Simulator are verified. Both Python files pass strict typing, Black
and Ruff. Compiler scope, language, architecture, IR and protected API checks
pass. The dependency baseline is identical at 720 findings. Git HTTPS remains
blocked after an independent ten-second TLS handshake timeout, recorded in
g2-git-transport-recheck-20261123.json. Publication remains pending.

## Publication verification

The accepted module is published in the seven-commit range ending at
05031474d26059a3c531269738aa7b16fa07a923 on PR #1084. The GitHub Git
database API preserved every local blob, tree and commit SHA and updated the
integration branch with force disabled. The verified publication supersedes
the historical local-only transport status above. The combined published
catalogue covers 85/303 constructor identifiers, with 218 remaining.
The per-module execution, symbolic, codec and model boundaries remain recorded
at their original scopes. Exact publication evidence is retained in
ftqc-reference-env/g2-git-object-publication-20261123.json.
