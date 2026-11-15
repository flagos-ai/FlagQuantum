# Unitary block encoding source files

Reference: Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen files under
`qualtran/bloqs/block_encoding/` are Apache-2.0, copyright Google LLC.

| Item | Source file | Frozen Git blob | Disposition |
| --- | --- | --- | --- |
| Q-0305 | `unitary.ipynb` | `cca05e99e365416bada5a7785a43ac1ba3d57555` | Closed by the full native tutorial. |
| Q-0306 | `unitary.py` | `bbcff63d78cbd58979bee3670597a778fdd4bdd2` | Open for symbolic and shaped child binding, generic external objects, and controlled specialization. |
| Q-0307 | `unitary_test.py` | `4bead193a26eb24303fb96a15bd2233774924e0e` | Closed for the original runnable scenarios and notebook. |

## Interface, bit order and ownership

The Algorithms wrapper accepts the existing typed signature and admits
concrete scalar quantum through ports for a physical binding. It rejects
input-only and output-only children before emission. The Compiler's shared
partition layout concatenates the child system words in signature order.
Each word is big-endian. The outer signature declares system, ancilla and
resource in that order, with zero-width entries omitted. Ancilla and
resource pass through untouched. Caller-provided work and classical slots
belong to the child's decomposition. The wrapper owns no allocator, gates,
or feedback of its own. The signal state delegates to identity preparation.

The single directional child contributes the direct call and gate model.
Auxiliary ports contribute bystanders to the logical peak. The plain and
overridden T fixtures each emit one T and no measurements, and reserve
one and four physical wires respectively. The independent model declares
one T equivalent for each with logical peaks one and four. The earlier
[execution audit](UNITARY_BLOCK_ENCODING.md) covers 48 model comparisons
across both profiles, complete source gate sequences and all small
measurement histories, including clean work and coherent auxiliary states.

## Original scenarios

The default and overridden T examples preserve the complete packed
signature and the declared scale and error. Their full 2-by-2 and
16-by-16 complex matrices agree with the locked tensor contraction;
the forward stream has T on the system only and the inverse restores
all basis columns. Tests verify identity signal preparation and the
source's refusal to wrap a one-sided integer state. The concrete typed
adapter uses the existing register metadata; it refuses shaped, symbolic,
and classical child registers because the binding uses flat quantum words.

The unchanged original non-notebook run has eight passes and two
expected source serialization failures. The original 11-cell and native
13-cell notebooks both pass independent kernels. The native tutorial
preserves both fixtures, their graphical signatures, and the single-call
graph, then exercises append and adjoint with full complex matrices. The
native notebook runner rejects reference imports and checks both SVGs.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_unitary_block_encoding.ipynb
```
