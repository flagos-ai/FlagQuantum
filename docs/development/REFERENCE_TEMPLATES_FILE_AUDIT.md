# Template reference generator source closure

## Complete source group and ownership

Q-0086 covers every function, page-info class, builder hook and constant in the
frozen reference_docs.py at Qualtran v0.7.0
bcb17bb25360ed68ce3bc084645db3e929799389. Q-0113 through Q-0118 cover its six
Jinja assets. All seven Git blobs match the immutable inventory. The native
assets retain the exact original bytes. The original Google Apache-2.0 notice
is retained in the Python adaptation. The local other-member formatting also
credits the TensorFlow Authors for its Apache-2.0 upstream helper.

Tools owns three responsibilities. reference_template_pages.py owns the optional
SDK's concrete page plugins, declaration-order filtering and local description
formatting. reference_templates.py owns native package imports, canonical alias
policy and generation orchestration. reference_output.py owns Markdown link
repair, metadata removal and Sphinx contents. It reuses WriteIfDifferent for
owned output. The existing static Griffe renderer and its Markdown TOC retain
their distinct object and file-format responsibilities. No runtime domain imports
these developer tools.

The source's dynamic template mixin and four SDK page classes become four
concrete plugin types assembled at the optional SDK boundary. Their typed hook
functions preserve the SDK's actual subclass contract. The SDK's page records
and constructors remain external objects. No parallel Python API model is added.
Source-specific type alias names become the five actual native aliases from
core.value_math and core.ir.program. Each declared location is imported and
checked in the acceptance test.

## Actual generation and differential evidence

An independent package fixture exercises real modules, classes, inheritance,
properties, constructor lifting, declaration-ordered methods and type aliases.
Its eight Markdown pages compare byte-for-byte between the unmodified source
builders and native builders. All four concrete page categories and all six
original template assets execute in that workflow. Inherited unoverridden
methods are excluded, while constructor and original method order are retained.

Both complete configured build entry points also run on the actual frozen
Qualtran package. For this reference comparison, the native generator receives
that package's original title, source-link prefix and alias-policy data. The
reference source remains unchanged. Every final generated file, including the
Sphinx TOC, compares byte-for-byte. A fresh controlled process produces 50 files.
Already imported submodules can expand this source profile, so the test verifies
the full equal file set and contents rather than treating 50 as a universal count.

The actual native default entry executes against this checkout with its native
package, imports and source links. Only its destination is redirected to the
reference evidence directory. It completes generation, fixups and TOC with
175 Markdown pages and one Sphinx TOC. The runtime generator follows already
imported public namespaces. The static Griffe walk's 1,520 records and its setter
refusal remain a separate scope. The 176 template outputs certify the measured
loaded namespace profile.

All suffix heuristics compare directly, including external URLs, fragment
anchors, quotes, bare filenames, whitespace and unmodified Markdown links.
Metadata cleanup and removal of all_symbols.md are exercised through actual
files. Native package spelling is the only normalization in the separate
postprocessing fixture. Direct TOC output and full frozen workflow outputs use
exact bytes. Repeated native TOC output preserves its timestamp through the
shared generated-file owner.

## Global-state correction and preserved boundaries

Importing the original module replaces two global TensorFlow-docs parser
formatters. The original mutation is explicitly reproduced, restored by the
source test fixture and enabled only around original reference calls. Native
adapts copied TitleBlock instances in the current page's owned documentation
fields, and supplies local class/module other-member formatting. Parser
configuration and documented Python objects retain their identities. Shared
blocks and global parser tables retain their original state.

Concurrent ordinary/native formatting gives independent HTML-table and native
description-list results. An injected rendering failure also preserves the
original block and global tables. Both DescriptionListItem and full TitleBlock
formatting compare against the source across empty descriptions, continuations,
anchors, empty titles, absent titles and empty/custom title formats.

The frozen description-list table template omits parsed freeform block text.
That source limitation remains explicit and tested. The source's autoescape=True
policy is retained, including escaped HTML emitted by SDK builder helpers. These
formatting semantics stay visible in the exact generated-page comparisons.
Existing WriteIfDifferent exceptional-buffer behavior remains its owner's
contract. This batch adds unchanged-file timestamp preservation to fixups and
TOC generation.

## Isolated dependencies and replacement

The optional template-docs-deps directory installs tensorflow-docs
2023.5.24.56664 and its 17 dependencies with the frozen docs.env.txt constraints.
The actual template test entry gives that directory precedence. Measured active
direct versions include Jinja2 3.1.6, protobuf 6.33.2, attrs 25.4.0, absl-py 2.3.1,
astor 0.8.1 and PyYAML 6.0.3. Dependency metadata and bundled license files are
recorded in g2-template-docs-license-review-20261121.json. TensorFlow-docs and
absl use Apache-2.0, the remaining packages use BSD, MIT or PSF licenses. Jinja's
included license has the three BSD conditions.

An initial environment probe used an earlier restored protobuf package. The final
template entry prioritizes the locked directory and asserts protobuf 6.33.2.
The host environment advertises a newer platformdirs pytest plugin absent from
the frozen platformdirs 4.5.1 package. The isolated test wrapper explicitly disables
that host-only plugin with -p no:platformdirs. All selected reference cases still
execute. Runtime manifests, repository pytest configuration and source files are
outside this environment correction.

TensorFlow-docs owns reflection, traversal, signatures, source links and page
publication. Jinja owns template inheritance and escaping. The native boundaries
are the supplied SDK plugin map and local template assets. A replacement must
pass the same original/source-profile output comparisons and real native build.
The removal path is the optional template adapter and isolated dependencies when
another owned renderer covers those outputs. No dependency is added to Core.

## Verification and resource scope

Run tests/unit/test_reference_templates.py with the existing template-free
reference, notebook, SDK helper, discovery, component and graph regression.
Concentrated regression passes 234 tests with zero skips and 902 visible upstream
parser/fork warnings. Strict typing covers four Python files. Black, Ruff,
Integration ownership, architecture, IR, protected API and diff checks cover the
batch. All six assets also pass Integration ownership checking.

This is classical developer documentation tooling. Previously accepted
circuit.append(op), complete complex amplitudes, measurement branches,
work-register cleanup and separate observed primitive/model resource evidence
are reused. Quantum gate counts are inapplicable to the documentation generator.

The explicit notebook catalog Q-0080 and its one-to-one native example mappings
remain open. Pending Proposals 116, 117 and 118 remain separate open contracts.

## Broader default-tier findings

The bounded default smoke/unit run selects 80,494 tests from 98,026 collected
items, with 17,532 deselections and 51 collection skips. It remains in progress
at publication. Its first Algorithms dependency failure reproduces independently
as one failed and two passing boundary tests. Existing G2 Algorithms modules
import NumPy, while the current repository dependency policy and Algorithms
syntactic guard require a Torch-only core installation. The affected numerical,
array and seeded-RNG contracts require domain-owned repair and renewed source
comparison. This template batch does not change those modules or weaken the
guard. Evidence is retained in
g2-template-docs-default-failure-isolated-20261121.log. The concentrated regression
above is green, while the complete default tier has no passing claim.

An initial sandboxed concentrated run failed four real-kernel notebook cases
because binding Jupyter ports was denied. The authorized local-port run executes
the same 234 cases successfully. No kernel test was skipped or replaced.
