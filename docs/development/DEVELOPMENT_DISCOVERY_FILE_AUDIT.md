# Complete component discovery source closure

Q-0057 maps the complete frozen `dev_tools/qualtran_dev_tools/bloq_finder.py`
from Qualtran v0.7.0, commit `bcb17bb25360ed68ce3bc084645db3e929799389`, into
`tools/development/component_discovery.py`. Its Git blob is checked directly.

## Ownership and adapted boundary

Integration owns source discovery. The tool reuses existing
`ComponentExample` and `ComponentDocumentation` records from Testing. It does
not introduce a component base class, registry, execution policy or framework
export. Class discovery takes an explicit domain predicate because FlagQuantum
components have no required common superclass. Module imports are ordinary
Python imports, and failures propagate to callers.

The complete source file's responsibilities are mapped as follows.

| Source responsibility | Native owner |
| --- | --- |
| Tracked/untracked Python source paths | `_get_git_paths`, `_get_paths` |
| Implementation/test file selection | `get_component_module_paths`, `get_component_test_module_paths` |
| Module namespace conversion | `_module_name` with an explicit package |
| Owned public component classes | `module_components` with a domain predicate |
| Example and documentation records | `module_examples`, `module_documentation` |
| Library-wide collection | `get_component_classes`, `get_component_examples`, `get_component_documentation` |

Default collection inspects tracked `flagquantum/algorithms` modules. An explicit
root includes local untracked modules, matching the original root convention.
The native module namespace is configurable. Native `test_*.py` files are
excluded from implementation discovery as well as the original `*_test.py`
convention. Git calls have the accepted 15-second timeout. These are explicit
adaptations to the native layout and execution policy.

Class ownership, private-class exclusion, alphabetical module-member order,
imported example/documentation inclusion and lazy factories preserve the source
behavior. Source file enumeration is not sorted or deduplicated, matching the
original. No factory is called merely to discover its record.

## Evidence and limits

A real temporary Git tree covers tracked and untracked modules, both test
conventions, nested namespace conversion and the default checkout route.
Unchanged source discovery functions run on the same tree. At the metadata type
boundary their foreign types are substituted with the existing native records.
This checks discovery logic without requiring native classes to inherit a
foreign framework class.

The native class discovery result matches all 340 classes discovered by the
original finder in the frozen source tree, using an explicit source-class
predicate solely for this reference test. A real native example catalog returns
its existing lazy swap factory and documentation record. Import errors propagate.

Q-0055/Q-0056, the complete library call-graph tool and its library-wide test,
remain open for a following dependency-closed batch. Their physical and modeled
resource acceptance cannot be inferred from class discovery. Quantum gate
streams, amplitudes, workspace and resource counts are inapplicable to this
source-introspection-only module.

The complete batch passed 153 concentrated cases, including the existing
component-example full-complex-state execution scenario and the original
metadata tests. Strict mypy passed both new files. Black, Ruff, Integration
scope, architecture, internal IR, public API baseline and whitespace passed.
