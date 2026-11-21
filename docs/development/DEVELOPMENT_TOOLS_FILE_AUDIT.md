# Development file and process helper closure

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Tests verify all five complete
source Git blobs against the frozen inventory before behavioral comparisons.

| Ledger | Complete source | FlagQuantum owner |
| --- | --- | --- |
| Q-0054 | `dev_tools/qualtran_dev_tools/__init__.py` | `tools/development/__init__.py` and its local README |
| Q-0060 | `dev_tools/qualtran_dev_tools/git_tools.py` | `tools/development/repository.py` |
| Q-0087 | `dev_tools/qualtran_dev_tools/shell_tools.py` | `tools/development/process.py` |
| Q-0088 | `dev_tools/qualtran_dev_tools/shell_tools_test.py` | Complete original tests executed against the native provider |
| Q-0090 | `dev_tools/qualtran_dev_tools/write_if_different.py` | `tools/development/files.py` |

## Interfaces and ownership

These internal stdlib helpers belong to Integration tooling. File comparison,
subprocess calls and checkout discovery have separate owners and require no
runtime, compiler, ecosystem SDK or simulation dependency. The source initializer
contains a package-purpose docstring and no exported implementation. The native
package and local README retain that purpose under FlagQuantum ownership.

`WriteIfDifferent(Path)` owns an in-memory text buffer and commits on close.
It preserves unchanged file timestamps, writes changed Unicode text, reports the
source messages and implements the complete original stream protocol. Context
exceptions still write partial buffered content and propagate. Missing parents
fail, repeated close and closed writable() raise ValueError. Closed flush() returns None.
The source does not provide atomic replacement or concurrent-writer coordination.
Callers own output-path coordination and buffer size.

`run` preserves the source checked/text defaults, string or list commands,
subprocess keyword forwarding and exact stderr command logging. Highlighting and
argument abbreviation retain empty-command IndexError and lone-dash behavior.
`output_of` removes exactly one final newline. Requesting bytes from this text
helper retains the original TypeError. The lower-level run supports bytes.

`get_git_root` preserves the checked Git command, absolute existing Path and
non-checkout failure. The native command has a deliberate 15-second timeout to
bound developer-tool stalls. The source has no timeout. This is an explicit
execution-policy difference covered separately by a subprocess-boundary test.

## Acceptance

`tests/unit/test_development_tools.py` executes all four original shell tests
without rewriting their bodies, substituting the native provider and bounding
subprocess execution. Additional source/native comparisons cover the stream
lifecycle, context failure, unchanged timestamp, command failures, binary output,
newline trimming, cwd, timeout propagation, invalid inputs and actual temporary
Git checkout discovery. Blob verification covers all five complete files.

Quantum amplitudes, measurement feedback, work qubits, primitive gate counts and
resource models do not apply to these classical development helpers. Existing
quantum evidence remains attached to its respective modules.

Run the tests in the locked G2 reference environment, strict mypy on all new
Python files, Black, Ruff, Integration scope, architecture, IR and API checks.
The package README provides the executable file-generation golden path.

The final batch run passed all 65 concentrated cases. Strict mypy passed all five
new Python files. Black, Ruff, Integration scope, architecture, internal IR,
public API baseline and whitespace checks passed. Earlier incorrect assertions
about closed StringIO methods were corrected against both untouched source and
native behavior before acceptance.
