# Optimization package ownership audit

Both complete package files are frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Source file | Frozen blob | Native disposition |
| --- | --- | --- |
| optimization/__init__.py | 528a9789f62243bf6e64ad2ea3929dc4e663cbf7 | Empty namespace, owned by Algorithms |
| optimization/k_xor_sat/__init__.py | 79b00094a6874a9f8395876d6672201303951add | Six complete imports mapped to existing owners |

The empty source file contains copyright and license comments only. The k-XOR
package has 12 imported names and 10 unique classes. Matrix-entry and neighbor
classes are imported twice, once directly and once via block encoding. Reference
identity checks prove these duplicate imports resolve to the same source classes.

## Complete native mapping

| Source class | Native module and class |
| --- | --- |
| KikuchiNonZeroIndex | algorithms.kikuchi_adjacency.KikuchiNonZeroIndex |
| KikuchiMatrixEntry | algorithms.kikuchi_matrix.KikuchiMatrixEntry |
| KikuchiHamiltonian | algorithms.kikuchi_block_encoding.KikuchiHamiltonian |
| GuidingState | algorithms.kikuchi_guiding_state.GuidingState |
| SimpleGuidingState | algorithms.kikuchi_guiding_state.ScopeGuidingState |
| Constraint, KXorInstance | algorithms.kxor_instance, same class names |
| LoadConstraintScopes, LoadUniqueScopeIndex, PRGAUniqueConstraintRHS | algorithms.kxor_loading, same class names |

Every unique native class is defined by its responsible module and has a distinct
identity from the SDK reference. All eight operation classes construct against
the same concrete constraints and match full signature names, widths, shapes and
sides. Existing complete-source evidence supplies primitive, amplitude, feedback,
work-zero and QEC comparisons for their supported paths.

Q-0599 and Q-0600 close as complete package ownership mappings. Component parents
retain their own concrete physical gaps, including sorting decomposition,
omitted phase-gradient ports and original
count/serialization failures. Package closure makes no additional execution or
resource claim. Native consumers import each class from the domain owner listed above.

The batch passes 972 regression cases, strict mypy for the complete new test
file, Black, Ruff, team scope, architecture, IR and protected API checks. Three
existing regression warnings remain visible.

## Tutorial follow-up

The untouched Q-0613 tutorial executes all 17 original cells successfully, as
recorded in `g2-kikuchi-guiding-tutorial-original.log`. Four original cells require
symbolic or concrete root component diagrams, including flattening. Native root
Program construction now admits bounded permutation registers and symbolic
word widths with concrete scope count under the existing Core contract.
Symbolic and concrete twelve-call roots and the twenty-one-call concrete
expansion match the original connections. Concrete HasDuplicates has an
original body and now reuses its literal native component; its alias defect
is retained. Scope preparation and sorting are the bodyless prefixes.
Q-0613 remains open for the remaining complete-cell and resource-graph
verification, not a register admission blocker. Its working source run is
reused and will not be rerun to manufacture progress.
