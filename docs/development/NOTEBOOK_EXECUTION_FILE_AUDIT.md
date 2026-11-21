# Notebook execution and export source closure

## Source and ownership

Q-0078 and Q-0079 cover the complete original notebook_execution.py and its
unchanged test file in dev_tools/qualtran_dev_tools at Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. Both Git blobs are checked against the
frozen inventory. Google copyright and Apache-2.0 notices remain in the ports.

Testing owns execute_notebook_node, the shared optional Jupyter execution path.
The existing Ecosystem Cirq helper delegates to this owner with the same
signature, defaults and caller-relative file resolution. Its dependency remains
lazy. Tools owns tracked discovery, modification-time selection, ordered
Markdown substitutions, path derivation, export and serial/process summaries.
These internal changes pass Verification, Ecosystem and Integration ownership
checks. NotebookNode and existing Jupyter facilities represent and execute
notebooks. Actual quantum programs still use the Compiler and Simulation owners.

## Source comparisons and executable path

Both unchanged original tests run against native providers. Four output-flag
combinations, missing/stale/fresh sources, repeated ordered substitutions,
Markdown-only replacement, exact exported bytes, execution errors, serial and
fork-pool ordering/error summaries, empty runs and actual tracked Git discovery
compare to the original. Reference link targets are explicitly supplied for
byte comparisons. Native defaults link owned names to existing native pages.

Real Jupyter execution compares original and native executed notebook contents
and Markdown bytes. Native spawn runs use standalone main-guarded scripts with
two tracked notebooks, covering both complete success and one notebook failure.
Successful outputs remain available, failed notebooks produce no exports, the
batch exits with status 1 and kernel connection files are reclaimed. All real
execution runs use isolated Jupyter configuration, runtime and kernelspec paths.

A real kernel runs an ordinary circuit.append(FieldOperation(...)) with H and S,
checks both complex amplitudes [1, i]/sqrt(2), and checks the two observed IR
instructions. Existing component/workspace and measured-branch evidence is
reused. This classical documentation workflow introduces no new decomposition.
Its two observed instructions are independent of resource models, and it makes
no additional model estimate.

## Deliberate lifecycle and publication corrections

With the locked nbconvert 7.16.6 and nbclient 0.11.0 environment, the original
precreates a kernel and client under FileLock before calling preprocess.
Preprocess resets its NotebookClient state and starts another manager. The real
reference test records two managers and one surviving kernel after return, then
explicitly shuts down the reference leftovers. Native execution puts the lock
around entry to the actual Jupyter setup context. It creates one manager and
reclaims it on return. Kernel startup failures are returned as execution task
errors, while original pre-start failures propagate before its execution catch.

Original parallel empty input raises AssertionError. Native empty serial and
parallel runs both report successful completion. Rendering completes before
writing destinations and each destination uses same-directory atomic replacement
with temporary cleanup. If replacement of the second destination fails after
the first succeeds, the first remains updated. Cross-file rollback is outside
this workflow's contract. Original source writes the notebook before rendering
Markdown. The regression verifies preservation of old outputs on render failure
and cleanup on destination replacement failure.

Original MarkdownExporter resource behavior is retained. Image attachments are
returned separately by Jupyter and the workflow writes only Markdown and the
optional executed notebook. Asset materialization remains a documented boundary.
Input parsing and export errors continue to propagate. Repeated linkification
retains the original substring replacement behavior, including nested links.

## Acceptance and remaining scope

Run tests/unit/test_notebook_execution.py with the existing notebook-tools,
Cirq notebook-resource, example metadata, docstring, discovery and graph tests.
Run strict mypy on the four changed Python files, Black, Ruff, scope checks,
architecture, IR and protected API checks. Reference tests require the pinned
optional tool environment and actual kernel execution requires loopback ports.
Fork comparisons emit the Python multi-threaded fork deprecation warning.

Q-0080 notebook_specs.py remains open. Its full explicit catalog needs native
module/documentation mappings. This execution workflow's tracked discovery does
not establish catalog equivalence. Pending Proposals 116, 117 and 118 remain
unimplemented.
