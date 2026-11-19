# Complete signed bit-addressing tutorial

Q-0945 maps every cell of the unchanged Qualtran v0.7.0
`tutorials/quantum-programming/3-addressing-bits.ipynb`, revision
`bcb17bb25360ed68ce3bc084645db3e929799389`, blob
`a0f1acfbe5dd8c0bc05ced68b3255bcbcabad717`.
The native `examples/notebooks/register_bit_addressing.ipynb` retains all 11
cells in order and adds one provenance cell with Apache-2.0 attribution.

## Interface and ownership

The user executes `fq.Circuit(8).append(WordSwap())`. Core owns two signed,
four-bit, MSB-first through-registers and their linear ownership. Compiler's
existing builder splits each word, records four two-bit swaps and rejoins the
same signed types. Its existing physical register binder derives the emitted
circuit directly from that stored graph. Simulation interprets stored calls
using shared split/join codecs and the existing raw swap action. Drawer consumes
the same graph for connected and musical-score views.

This source file exposed a missing prerequisite in finite execution admission.
Proposal 103 prerequisite 1.18 and its machine contract landed first as
`bde4c4fe2`, followed by Core implementation `6ba71799d`. They admit the existing
unparameterized signed quantum descriptor without adding public fields, exports,
opcodes, codecs or schemas. Source-backed stand-in/native replacement scenarios
retain exact ordered connections and single consumption. Malformed profiles
remain refused. Zero-width words preserve logical ownership. Physical binding
continues to require at least one physical wire in the supplied layout.

## Complete source coverage

| Source cells | Native execution |
| --- | --- |
| 0–2 | Explain MSB-first signed bit addressing and construct the complete stored graph |
| 3–4 | Verify the typed graph and physical circuit |
| 5–6 | Render all actual producer/consumer connections |
| 7–8 | Render split, bit swap and join lanes from the same graph |
| 9–10 | Evaluate stored classical actions, including (5, 6) to (6, 5), and report separate gate/model counts |

All original code cells execute unchanged. Native cells execute with imports of
Qualtran, Cirq and Kickmix rejected. The source sets `n=4` while two register
comments incorrectly say eight bits. Native prose follows its executable
four-bit definition. Notebook-local dataclass and explicit append replace the
source builder's circuit interface. Graphviz, IPython and Matplotlib are optional
presentation dependencies. No production numerical or Compiler implementation
is added by the tutorial.

## Acceptance evidence

All 256 signed input pairs from −8 through 7 match the original classical
execution. The complete 256-by-256 original tensor acts on both spectator
columns of a nine-wire state containing every input basis coefficient with
nontrivial real and imaginary parts. Native appended execution agrees exactly,
including phase and the entangled spectator. The routine has zero workspace and
zero measurements, so there are no measurement histories or work wires to
enumerate. Physical binding returns both original wire words with zero lifetime
events.

At the tutorial's atomic level, both exports emit four SWAP instructions with
identical operands and order. Requesting CNOT-only native basis translation and
fully decomposing the unchanged source both produce the same ordered twelve
CNOT instructions. These are separately tested granularities. The independent
source QEC model and native resource formula both report four Clifford gates,
eight logical wires, zero T and zero measurements. Four model Cliffords do not
mean twelve model Cliffords after primitive basis expansion.

The complete graph has two Split, four TwoBitSwap and two Join calls. Native
musical-score lane geometry, generations, glyphs and labels match the source.
Boundary labels retain native lifetime identities and QAny width captions under
the existing renderer contract. The shared comparison normalizes those boundary
captions only. SWAP titles and x/y labels use the original tutorial spelling.

The batch's 195 concentrated cases pass without skips, including signed/field
execution, rebinding/lowering, both preceding complete tutorials and musical
scores. Strict typing passes for extracted notebook code and its test. Each code
cell is individually Black-checked, and notebook-aware Ruff, scope, architecture,
the machine IR contract and 74 IR/API cases pass. The optional Black Jupyter
formatter is absent. Full dependency diagnostics retain 706 existing findings,
with zero findings in this batch's paths. Whole-repository dependency-policy and
full smoke/unit acceptance remain separate. Physical resource dashboard/factory
numerical parents Q-0910/Q-0911/Q-0930/Q-0931 remain open.
