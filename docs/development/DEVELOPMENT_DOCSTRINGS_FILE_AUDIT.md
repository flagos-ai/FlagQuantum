# Development docstring parsing closure

The unchanged reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Frozen Git blobs are checked against
the inventory for the complete `parse_docstrings.py` (Q-0081) and
`parse_docstrings_test.py` (Q-0082) files.

## Ownership and interfaces

Integration owns `tools/development/docstrings.py`. It owns immutable structured
and fallback references, reference splitting, Google-style class docstring
rendering, custom Registers/Parameters/References sections and Markdown headers.
Inputs are ordinary Python classes or strings. Outputs are native text records
and lists of Markdown lines. Sphinx objects stay within this development-tool
boundary. Framework runtime packages do not import this module.

The source regular expression, newline handling, section formatting, whitespace,
inline-function substitutions and unparsed-reference fallback are retained.
The inherited Sphinx constructor is forwarded with its full positional/keyword
protocol because Sphinx recursively constructs the parser from a list of lines
and a positional configuration. Any annotations occur only at this external
constructor boundary. Immutable records retain attrs keyword-only and frozen
behavior. No new framework API or serialization vocabulary is introduced.

## Dependency review and replacement boundary

The frozen source docs requirement declares unpinned Sphinx, while the parser
also uses attrs. The isolated acceptance environment pins Sphinx 7.4.7 and uses
attrs 26.1.0 with Docutils 0.21.2. Both native and untouched source execute in
that same environment. This pin is an acceptance-environment choice.

Sphinx supplies the complete Google-style section parser and attrs supplies the
source record semantics. The existing repository has no equivalent authoritative
docstring renderer. These are optional development-tool dependencies, installed
in the reference environment. Core runtime dependencies and package manifests
are unaffected. Missing dependencies fail on explicit tool import. Reference
tests import the tool through fixtures so marker-filtered collection does not
require Sphinx.

The installed wheel licenses were read locally. Sphinx is BSD-2-Clause with
incorporated-software notices in LICENSE.rst. attrs is MIT. Docutils describes
its public-domain files and individual exceptions in COPYING.txt. Dependency
versions and license hashes are recorded in the batch environment evidence.
The environment was installed from package-index wheels with an explicit Sphinx
pin. Parsing uses local text and performs no network requests. This review
covers ownership, provenance and licenses. It is not a vulnerability-scan claim.

Replacement is bounded by the native text/reference inputs and outputs. A future
parser may replace Sphinx only after passing the same complete rendering and
reference corpus, including whitespace and failure tests. The attrs dependency
can be removed only after preserving immutable record construction, equality,
hashing and failure behavior. Integration owns both replacement decisions.

## Acceptance and source boundaries

Both complete original tests run without changing their bodies. The discovery
context of the original full-reference test is the extracted frozen source tree,
using the original finder. All 340 discovered source component classes have
exactly matching Markdown lines and reference records in both implementations.
The test asserts a nonempty corpus and independently compares every class.

Additional comparisons cover missing/empty docstrings, complete register and
parameter sections, multiline references, repeated reference blocks, HTTP/HTTPS,
unsupported links, Unicode titles, fallback text, frozen records, equality,
hashing and keyword-only construction. Unsupported URL/title forms preserve
the original UnparsedReference result. No broader parser support is claimed.

Quantum primitive/model counts, amplitudes, measurement feedback and workspace
cleanup do not apply to these classical documentation tools.

The complete batch passed 129 concentrated cases. Both original docstring tests
and all 340 source classes passed. Strict mypy passed both new Python files;
Black, Ruff, Integration scope, architecture, internal IR, public API baseline
and whitespace checks passed. Reference-free marker-filtered collection also
completed without importing the optional parser dependencies.
