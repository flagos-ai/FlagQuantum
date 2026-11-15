# States and effects tutorial

Q-0255 is `qualtran/bloqs/basic_gates/states_and_effects.ipynb` at
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`, Git blob
`cabb21c54a8e8444972f1a687b295c37b562c8f1`. The locked asset is unchanged.

The 89-cell native notebook retains the topics in the original 87 cells.
It includes all ten signatures and call graphs with leaf counts, the
prepared Bell graph and the physical circuit diagram. The added example
exercises all ten vectors on the ordinary append and tensor effect paths.

| Original examples | Signature and executable counterpart |
|---|---|
| Zero and One state/effect | `q` is a scalar QBit. States produce the right port; effects consume the left port. Append applies zero or one X to a clean output wire. Tensor effect selects the corresponding complex amplitude. |
| Plus and Minus state/effect | The same signature uses X-basis vectors. State preparation applies H, with a preceding X for Minus. Effect contraction preserves the signed complex overlap without normalization. |
| Integer 55 state/effect | `val` is one eight-qubit opaque word with the matching directional port. The native circuit loads the bits most significant first, while the native effect contracts in the same order. |
| Bell composition | The original Plus/Zero/CNOT dependency graph is displayed with native identities. The existing typed Bell tensor workflow and the two-wire H/CX append circuit both produce the same complete complex vector. |

The eight scalar basis examples each declare one atomic leaf with zero QEC
gates. Integer state and effect each declare one `ArbitraryClifford(8)`
child. Preparing integer 55 emits five X gates at bit positions 2, 3, 5, 6,
and 7 in the eight-wire register. A basis effect is a tensor contraction
and emits no circuit instructions. Bell preparation emits H and CX, produces
two logical output wires, and needs no extra work or measurements.

The source Cirq Bell visualization is represented by the native circuit
drawer in the teaching asset. Source implementations and their existing
boundaries remain in the separate basis and vector inventory entries.

Both notebooks ran in independent kernels. The native kernel rejected
Qualtran, Quimb and Cirq imports. The 21 SVGs and the Bell circuit PNG were
inspected. All 628 concentrated and 111 integration cases passed. Strict
typing, formatting, team scope, architecture, multilevel Program IR and
the public API baseline passed.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_states_effects.ipynb
```
