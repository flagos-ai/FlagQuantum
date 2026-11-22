<!-- Adapted from Qualtran v0.7.0 qualtran/Autodoc.ipynb, Google LLC, Apache-2.0. -->

# Documenting register components

The class docstring is the authoritative narrative for a component. Describe its
purpose, construction, mathematics and decomposition there. Native operation
bodies, typed registers and resource owners continue to define executable
behavior. This guide covers the complete four-cell upstream authoring notebook
at revision `bcb17bb25360ed68ce3bc084645db3e929799389`.

## Docstring basics

Start with a one-line summary, a blank line and explanatory paragraphs. Use
Markdown for the body, `$...$` for inline mathematics and `$$...$$` for displayed
equations. Use a Python raw docstring when LaTeX backslashes would otherwise
become escape sequences. Keep constructor parameters and decomposition context
on the class docstring so the renderer can produce one cohesive component page.
Use the repository's Google-style docstring sections.

## Docstring sections

### Args

Document each constructor parameter or class attribute by name and description.
A uniform-preparation component's class documentation can describe its count
and control values with this raw docstring. The original illustration describes
an $O(\log n)$ T-gate construction, whose cost remains owned by the existing
uniform-preparation resource model.

```python
r"""Prepare a uniform superposition over the first n basis states.

Args:
    n: Number of basis states in the uniform superposition.
    cvs: Control value for each control bit, when controls are present.
"""
```

### Registers

The renderer recognizes a Registers section. Describe logical input/output
roles alongside register names. The conjunction example uses a two-bit control
and a freshly produced target.

```python
"""Compute the conjunction of two control bits.

Registers:
    ctrl: A two-bit control register.
    target [right]: The output bit.
"""
```

### References

Separate references with blank lines. Begin each with a Markdown title and
abstract-page link, followed by optional additional information. Keep that link
on one source line when readability requires it. Put author names, publication
year and relevant figures or equations on the following line. Prefer arXiv
abstract links. The table-lookup example uses these two references.

```python
"""Load data indexed by a selection register into a target register.

References:
    [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity](https://arxiv.org/abs/1805.03662).
    Babbush et al. (2018). Figure 1.

    [Compilation of Fault-Tolerant Quantum Heuristics for Combinatorial Optimization](https://arxiv.org/abs/2007.07391).
    Babbush et al. (2020). Figure 3.
"""
```

These strings demonstrate documentation syntax. Existing uniform-preparation,
conjunction and lookup implementations retain their domain-owned interfaces.

## Example factories

The existing `ComponentExample` metadata and `component_example` decorator
retain lazy example construction. Supply multiple examples for different
parameter choices, including symbolic declarations. The factory needs a
concrete return-class annotation. Its name begins with an underscore followed
by the local variable assigned before returning. The source extractor renders
that construction body into a notebook. Prefer placing factories beside the
component they document.

The source notebook's three modular-exponentiation examples map to the existing
native constructor and factoring factory with their original parameters.

```python
import sympy

from flagquantum.algorithms.primitives import modular_exponentiation
from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    component_example,
)
from tools.development.notebook_autogen import NotebookSpec


@component_example
def _modexp_small() -> ModularExponentiation:
    modexp_small = ModularExponentiation(3, 15, 3, 2048)
    return modexp_small


@component_example
def _modexp() -> ModularExponentiation:
    modexp = ModularExponentiation.for_shor(15 * 17, 9)
    return modexp


@component_example
def _modexp_symb() -> ModularExponentiation:
    g, N, n_e, n_x = sympy.symbols("g N n_e n_x")
    modexp_symb = ModularExponentiation(g, N, n_e, n_x)
    return modexp_symb
```

The original small factory uses base 3 modulo 15. Those values are not coprime,
so the unchanged upstream constructor raises AssertionError and the native
constructor raises its existing ValueError when that lazy factory is called.
The example is retained as a documented source counterexample. The original factoring factory also chooses base 9 for 15 * 17, which shares
a factor of 3 and fails both constructors. The source runtime module itself
uses base 4 for its small example and 13 * 17 for factoring with base 9.
Those valid native choices are executable directly.

```python
valid_small = ModularExponentiation(4, 15, 3, 2048)
assert valid_small.classical_values(2) == (2, 1)
valid_factoring = ModularExponentiation.for_shor(13 * 17, 9)
assert valid_factoring.modulus == 221
```

These examples construct declarations. The 2048-bit and symbolic cases retain
their existing declaration and resource scope. Physical execution follows each
component's supported concrete binding and `circuit.append(op)` path.

## Automatic notebook documentation

Group the component class with its ordered example metadata in the existing
`ComponentDocumentation` record. The native field names express the same class,
example list and import responsibility as the upstream documentation record.
The explicit import line points to the defining native module.

```python
_MODEXP_DOC = ComponentDocumentation(
    component_type=ModularExponentiation,
    examples=(_modexp_symb, _modexp_small, _modexp),
    import_line=(
        "from flagquantum.algorithms.primitives.modular_exponentiation "
        "import ModularExponentiation"
    ),
)

_MODEXP_NOTEBOOK = NotebookSpec(
    title="Modular Exponentiation",
    module=modular_exponentiation,
    component_specs=[_MODEXP_DOC],
)
```

A component has one documentation record containing its examples. A module
notebook can contain multiple such records. Register its authored notebook in
`tools/development/notebook_catalog.py`, or extend the existing module's
`NotebookSpec` when regenerating its documentation cells. The native catalog
owns the explicit publication order. `notebook_autogen.render_notebook` merges
rendered class documentation and factory bodies while retaining authored cells.
The owning diagram and resource modules supply graphical-signature and call-graph
code through its existing callbacks. Choose an explicit output directory when
rendering outside the defining module, then add the resulting notebook to Git.

Run `python -m tools.development.notebook_publication` to publish the registered
library through the existing owner. The [documentation build](../README.md#html-documentation)
records its Sphinx workflow. No package import implicitly generates files.

## Complete source disposition

| Source cell | Native content and existing owner |
| --- | --- |
| 0 | Class-docstring authority, summary, Markdown, mathematics, raw strings and cohesive class-level documentation in Docstring basics. |
| 1 | Args, Registers and References syntax with all three examples and both original references. Existing tools/development/docstrings.py owns parsing. |
| 2 | Three original modular-exponentiation choices, factory annotation, naming, local assignment and colocated-authoring conventions. Existing testing/component_examples.py owns metadata. |
| 3 | Ordered documentation grouping, module notebook specification, catalog registration, generation and Git publication. Existing notebook_autogen and notebook_publication own execution. |

The original notebook has four Markdown cells and no executable cells or stored
outputs. Their teaching code blocks are validated as native examples. The
frozen source blob is `87a1784e4d68804069f3c4a5e8565ba4e12c5ec1`.
Physical gate streams, full complex-amplitude and work-zero validation remain
under the accepted modular-exponentiation and compiler owners. This authoring
module defines no primitive gate count or resource-model formula.


The source notebook's two non-coprime factory defects keep parent Q-0139 open under the
integration acceptance rules. The guide and existing implementation evidence
are delivered, with these specific source inconsistencies recorded for final
disposition. The frozen notebook and runtime module remain unchanged.
