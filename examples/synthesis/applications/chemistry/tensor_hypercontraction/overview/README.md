# Tensor-hypercontraction walk resource experiment

Open [the walkthrough](walkthrough.ipynb) or run the shared CLI from the repository root.

```bash
python -m examples.synthesis.applications.chemistry.tensor_hypercontraction.overview.main
```

The notebook generates the six original 31-row QROM tables and all sixteen sine-state amplitudes from the original inputs. The native Circuit implementation emits both PREPARE stages, SELECT, the fifteen-leaf clock traversal, its controlled reflections, and an inverse QFT. All 71 ELL bits, eight spin bits, and four clock bits stay in the full program.

The compiler reports declared width, used wires, operations, and dependency depth from native Circuit IR. The original notebook reports transpiled resources. The resource representations differ, and the full circuit is compiled without dense simulation. Independent traces verify all address-arithmetic inputs, controlled and unconditional lookup cases, all clock words, comparison boundary cases, and borrowed-bit restoration for the wide reflection.

The companion native quantum example uses the initialized-subspace simplification. This overview keeps the full walk circuits so readers can inspect their resource costs.
