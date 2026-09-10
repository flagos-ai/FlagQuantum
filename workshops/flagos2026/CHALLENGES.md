# Challenge board

Choose one question. Write down your prediction, make a controlled change, and explain what you observed.
Use the lab notebooks as starting points rather than importing their completed implementations.

## First discoveries

- **Bell or not Bell?** Build three two-qubit circuits with different probability distributions. Explain each gate. Start with 01.
- **A noisy coin.** Estimate a rotation angle from sampled counts and show how your estimate varies with shots. Start with 09.
- **A learning-rate detective.** Compare three learning rates from identical initial parameters. Plot the full curves, including failures. Start with 03.

Deliver a circuit or small function, a plot, and an explanation of one surprising result.

## AI and scientific computing

- **Does the quantum layer help this model?** Compare lab 07 with a small classical baseline using separate training and test inputs. Match the comparison budget and report parameter counts. Do not infer quantum advantage from a toy fit.
- **How expressive is your ansatz?** Remove and add layers in VQE. Compare final energy with the exact solution across several seeds. Separate representational limits from optimization failures. Start with 08.
- **Can QAOA find good cuts?** Change the graph, add a layer, and report the probability of sampling an optimal cut as well as expected cut size. Start with 15.
- **Noise versus sampling.** Combine the ideas in 09 and 10. Explain which errors shrink with more shots and which do not.

Deliver a reference calculation, a comparison across seeds or settings, and clearly stated limitations.

## Systems and framework development

- **Where does MPS stop being compact?** Hold qubit count fixed and vary circuit depth and entangling structure. Plot error versus stored elements. Start with 11.
- **Catch a broken gradient.** Introduce a deliberate `.detach()` or `.item()` mistake, explain the failure, and repair it. Start with 14.
- **A compiler rule you can trust.** Extend the adjacent-X rule to another narrowly defined case. Test positive and negative cases and compare states before and after. Start with 17.
- **Recover an interrupted sweep.** Describe how to resume collecting results without duplicate submissions. Use local records first, then a small approved cloud run. Start with 16.

Deliver a failing case, a fix or experiment, and a repeatable correctness check.

## Real-device investigations

- **Does the learned setting survive deployment?** Repeat lab 13 for several reachable targets and record the hardware task ID, calibration context when available, and shot count.
- **Mapping matters.** With instructor approval, compare two currently valid physical mappings of the same circuit. Record the returned physical circuits and results. Start with 12 and 05.
- **Measure in another basis.** Add the basis rotation needed to estimate X as well as Z. State what this adds to your knowledge and what it still cannot prove about the state.

These projects require device access. Agree on a submission budget before running them. A replay may support analysis, but label it as a replay.

## Show your work

For any challenge, include:

1. The question and your prediction.
2. Code and configuration, including seeds and software versions.
3. An exact reference, a baseline, or a justified consistency check.
4. A plot or table that answers the question.
5. What remains uncertain and the next experiment you would run.

Keep credentials and personal platform details out of shared files.
