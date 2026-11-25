# Hardware-aware synthesis of MCX

This walkthrough builds the original 10-control MCX under a 20-qubit ceiling.
It also compiles the original 12-control MCX for two 18-qubit devices. One has
a 3 × 6 grid and the other has a star. Both use the CX and U basis.

Open [walkthrough.ipynb](walkthrough.ipynb) or run from the repository root.

```bash
python -m examples.synthesis.community.basic_examples.hw_aware_synthesis.hw_aware_synthesis.overview.main
```

The native qfunc expresses one multi-control operation. Synthesis selects the
base decomposition under the available workspace. Each topology tries three
initial placements and two native router strategies. The workflow counts
actual CX instructions after swap lowering and selects the lowest observed
count, breaking ties on depth then candidate name. It verifies every
two-qubit instruction against its device edges.

The validation uses one complex coherent input with active target zero and
one, a disabled control, and the all-zero branch. An independent all-ones
truth permutation checks every output amplitude. All hidden wires return to
zero to numerical precision. These measurements describe the evaluated
routes in a local ideal simulator.
