# Your experiment notebook

Running a cell produces a number. An experiment explains what that number means.
Use these prompts in a Markdown cell or a separate file under `outputs/`. Short answers are enough.

## Before you run

1. **Question:** What am I trying to find out?
2. **Prediction:** What result do I expect, and why?
3. **Settings:** Which circuit, parameters, backend, precision, seed, and shot count am I using?
4. **Reference:** What will I compare against: an analytic answer, exact diagonalization, another implementation, or a recorded measurement?

## After you run

```text
Question:
Prediction and reason:
Settings and software version:
Observed result:
Reference and numerical difference:
Data source (local / recorded / live):
One conclusion supported by this result:
One limitation:
One change to try next:
Saved result file or task ID, if applicable:
```

Change one setting at a time when investigating a surprising result. Keep the original run so that you can compare them.
A seed controls some random choices; it does not guarantee identical answers across all hardware and software versions.

## Questions for the three featured experiments

### Lab 07: one loss, two learning systems

- Before training, which parameters do you expect to receive gradients?
- After training, what evidence shows that both parts of the model changed?
- Do the held-out points test interpolation or extrapolation?
- Could different parameter values describe the same fitted function?

If you get stuck, inspect the parameter names, gradient norms, and before/after parameter differences.
The encoder bias and quantum offset enter the same total angle. A good fit does not uniquely identify them.
This small cosine-fitting problem illustrates joint differentiation; it does not demonstrate quantum advantage.

### Lab 18: one problem, three simulators

- What stays the same when the simulation mode changes?
- Why compare gradients before running the optimizer?
- If energies agree but gradients disagree, would you trust the training comparison?
- What additional measurements would you need to make a speed or memory claim?

If you get stuck, compare the initial parameters, observable, execution options, and gradient vectors.
The small two-qubit calculation checks consistency. It cannot establish which representation scales best for a larger circuit.

### Lab 13: from a learned angle to a measurement

- Is the trained circuit's ideal prediction exactly equal to the training target?
- What changes when an exact expectation becomes a finite set of counts?
- Does a difference from the ideal prediction identify its physical cause?
- Which saved fields establish whether a result came from live hardware?

If you get stuck, separate the target, the frozen circuit's ideal expectation, and the measured estimate.
The plotted sampling error bar estimates variation from finite shots; it does not include every source of device or calibration error.
The default local sample is an exercise result, not a hardware measurement.

## When a notebook surprises you

| Symptom | Useful next step |
| --- | --- |
| A variable is missing | Restart the kernel and run the preceding cells in order. Each lab initializes its own state. |
| Repeating a gate cell changes the answer again | Recreate the circuit before adding gates; circuit objects retain previous operations. |
| Training behaves differently after a rerun | Recreate the parameters and optimizer, then repeat the training cells. |
| A gradient is absent | Check that the parameter requires gradients and that `.item()` or `.detach()` did not remove it from the computation before the loss. |
| A numerical assertion fails | Keep the error and settings; check the kernel and package source with preflight before changing a tolerance. |
| A remote call stops before returning | Inspect the saved receipt or task record and platform status before submitting again. |

Finish by explaining your chart to another participant without reading the code aloud.
They should be able to identify the question, the evidence, and what your experiment leaves unresolved.
