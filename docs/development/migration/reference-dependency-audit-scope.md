# Use the same default test scope for dependency and reachability audits

Owner: Integration. Commit this declaration before implementation.

At PR #1360 head2e19847ea, Main46156119545 unit0 job141013233459 executes
real tests and fails only test_every_integration_a_test_probes_is_installed_by_a_lane_that_runs_it.
Its independent filesystem scan includes the relocated Qualtran references,
although those same tests were explicitly invoked outside testpaths before the
move and remain outside default recursive collection after it. The actual full
old/default selection was preserved; the dependency audit was not synchronized.
Retain this failing run and original log.

Change only tests/unit/test_lane_dependency_policy.py plus this declaration.
Reuse the existing _test_files helper from test_test_reachability_policy for the
_audit enumeration. That helper checks the actual conftest ignore declaration
and requires every excluded reference file to be an existing whole-file move
from tools/qualtran_parity in the protected source ledger. Do not copy its
filter, add another exception list, skip a policy test, or change lane selectors,
installed extras, import/device/environment probes or any assertion.

Keep every original test function AST exact. Only the shared-helper import and
_audit loop iterator may change. The caller must still enforce all five existing
dependency invariants on every original default test. Explicit reference source,
16 direct collection nodes and documented pinned environment stay unchanged.
No dependency, image, runner, scientific case/input/budget or product change.

Run both complete policy modules, existing related lane/offline/source consumers,
source quality and static gates. Compare the complete default collection and
original node order/multiplicity. The resulting head still requires independent
own CI/source/default/scientific/install/resource/elapsed qualification. Preserve
all prior failures and misses; no merge, force push, trial rerun, new compatibility
layer, runner administration or Rust.
