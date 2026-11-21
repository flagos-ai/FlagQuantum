# Generated-notebook checkout check source closure

## Complete source and ownership

Q-0024 covers the complete unchanged shell script check/autogenerate-notebooks
at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Its Git blob is
verified against the immutable source inventory. The native owner is
tools/development/notebook_publication_check.py. It calls the accepted publication
module in a child process, then queries Git status for the entire checkout.
Publication, notebook rendering, Git-root discovery and output-file ownership
remain in their existing modules.

Run the supported workflow from the checkout whose generated output is being
checked.

```bash
python -m tools.development.notebook_publication_check
```

This command returns zero only after successful publication and an empty Git
porcelain status. A generator failure retains its exit status and bypasses the
Git query. A failed Git query or any staged, modified or untracked file returns
one. Ignored files retain Git's normal handling. The tool does not stage files
or change the checkout's tracked-output or ignore policy. Publication currently
creates docs/components copies of the accepted native notebook assets. Newly
created output therefore causes a clean-tree check to fail until the owning
checkout has an explicit baseline or ignore policy for it.

## Exact comparisons and native process adaptations

The unchanged original shell script runs in a real temporary Git checkout with
only its generator import replaced by the accepted native publication owner.
The native module runs in a second equivalent checkout. Both publish all 120
distinct notebooks and the index. They agree on exit codes and final status
reports for initial untracked output, a committed output baseline, repeated
publication, ignored local files, changed authored inputs, unrelated staged
files and unrelated untracked files. Generator exit statuses 3, 7 and 37 are
compared against actual original shell invocations. A separate injected Git
failure verifies the native failure path.

The source's bare python command maps to the current Python interpreter and the
accepted native publication module. Native generation and Git queries have
explicit 120-second and 15-second limits. Process launch and timeout errors
propagate. Status output keeps the original whitespace folding. Literal glob
characters are preserved, avoiding the shell's incidental pathname expansion in
its unquoted echo. The check resolves the existing checkout root before running
the generator, allowing invocation from a nested checkout directory.

This is classical command tooling. Existing accepted publication and notebook
evidence retain the quantum semantics and measured-primitive versus model-cost
distinction. This batch makes no new gate-count or scientific-notebook execution
claim. Broad default-suite timeout, Algorithms NumPy dependency ownership and
the preceding-execution copyright interaction remain separate open debt.

## Batch verification

The concentrated regression passed 304 tests with 915 visible warnings and zero
skips. Strict mypy passed both changed Python files. Black, Ruff, Integration
scope, architecture, IR, API and whitespace checks passed. The initial sandboxed
regression had five Jupyter local-port permission failures. The same complete
regression passed with the required local-kernel permission; no failing test was
removed or replaced with a mock to obtain that result.
