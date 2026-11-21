# Copyright checker source closure

## Frozen scope and ownership

Q-0084 and Q-0085 cover the complete pylint_copyright_checker.py and its original
test suite at Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389. Both source
Git blobs are verified against the immutable inventory. The dev.env.txt lock and
pylint dependency profile are fetched at that exact revision and retained in the
reference environment. The source notices and Apache attribution are retained.

Tools owns the optional copyright_checker.py Pylint plugin. It validates retained
Google Apache-2.0 source notices, with the original year wildcard and diagnostic
identity. This is an opt-in imported-source audit. Each file's license and holder
remain determined by its provenance. The existing repository policy remains the
owner of required project-wide checks. Its registration hook delegates to the
installed Pylint host. Core, Compiler and Runtime have no dependency on this tool.

## Complete comparisons and native correction

The original unchanged test class runs with both source and native checker
providers. Both providers also compare every non-newline header character,
trailing spaces/tabs/CRLF, all 13 extra-text positions, correct/wrong holders and
years, shebangs, intervening blanks, body suffixes and disabled checks. Message
identities and physical line/column locations agree for complete input. Byte-level
cases supply a real Astroid node with a raw byte stream, allowing malformed Python
comments to reach the same raw-file checker boundary.

The original nested zip loops silently accept an empty file and every complete
line prefix of the correct notice. All 13 truncated prefixes and shebang-only
inputs reproduce that failure. Native consumes every expected notice line and
reports the first missing line, column zero. The reference source remains
unchanged and its incorrect acceptance stays asserted separately. The source's
two year positions accept arbitrary characters, including 20ab. That permissive
source behavior remains explicitly preserved.

Both registration functions install the expected checker in a real PyLinter.
The native public plugin-loading path executes through Pylint Run for a valid
file, invalid file, truncated notice and file-level opt-out. Actual exit statuses
are 0, 8, 8 and 0 respectively. Standard Pylint owns suppression and reporting.

## Optional dependency boundary

The isolated reference lint-deps directory contains the frozen dev.env.txt
versions Pylint 3.3.9, Astroid 3.3.11, dill 0.4.0, isort 5.10.1, mccabe 0.7.0,
platformdirs 4.5.1 and tomlkit 0.13.3. Distribution metadata and included license
files identify GPL-2.0-or-later for Pylint, LGPL-2.1-or-later for Astroid, BSD for
dill, MIT for isort/platformdirs/tomlkit and Expat for mccabe. Installation uses
the existing resolver, explicit versions and no package cache. The resolver's
warnings about malformed historical version metadata remain in the install log.
These are optional developer/test tools. No runtime dependency is added.

Pylint owns AST parsing, suppression, registration and diagnostic publication.
The checker receives only its documented raw-file boundary. Replacing Pylint
requires rerunning the unchanged original suite, raw diagnostic comparisons and
real plugin entry tests. Removing the opt-in plugin and isolated environment is
the exit path if repository tooling adopts another license-header auditor.

## Acceptance and resource scope

Run tests/unit/test_copyright_checker.py with the accepted development tooling,
reference generation, notebook, example and graph regression. The base environment
can collect the tests while excluding ftqc_reference without Pylint or Astroid.
Explicit reference execution requires the locked dependencies and performs every
selected case. Strict typing, Black, Ruff, Integration scope, architecture, IR,
protected API and diff checks cover the two changed Python files.

This batch is classical lint tooling. Existing circuit.append(op), complete
complex amplitudes, work-register cleanup and separate measured primitive/model
resource evidence are reused. Quantum gate counts are inapplicable to this tool.

The explicit notebook catalog Q-0080 and older TensorFlow-docs pipeline Q-0086
remain separate source closures. Pending Proposals 116, 117 and 118 remain open.

Final concentrated regression passes 224 tests with zero skips and 885 visible
upstream parser/fork warnings. Strict typing covers both changed Python files,
and all formatting, scope, architecture, IR, API and diff checks pass. A final
focused run rechecks the five complete checker scenarios after the Pylint message
mapping annotations were aligned with the host. Reference-free collection includes
the new file with 14 selected and 73 deselected tests.
