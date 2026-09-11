# Instructor guide

The aim is for each participant to understand and modify a working example.
Leave time for reading results and trying changes, rather than asking everyone to run every cell as quickly as possible.

## Let participants choose their route

Use the catalog in README.md to introduce the tracks. Do not make everyone complete all 18 labs in order.
Ask participants what they want to build and whether they are comfortable with PyTorch, quantum circuits, or numerical methods.
Offer a beginner table, an AI/science table, and a systems/extensions table when staffing allows.
All tables can share short demonstrations of real-device results.

Encourage advanced participants to work on CHALLENGES.md and explain their result to another group.
Have each group report its question, one numerical check, and one limitation. There is no shared completion deadline.

## Prepare the event

Fill in these details before sharing the materials:

| Item | What to arrange |
| --- | --- |
| Liangzhi Cloud | Event URL, sign-in instructions, environment entry point, and support contact |
| Software | FlagQuantum commit, QSteed plugin version, and notebook kernel |
| Jiuding | Full image address, Python executable, shared project path, and each participant's output directory |
| Compute resources | Expected attendance, reserved CPU/GPU capacity, group quotas, and who will clean up |
| Quafu | A currently available backend, credentials, shot allowance, and submission schedule |

Plan for the number of people who will submit at the same time. If hardware access is limited,
let each group submit one task and discuss its result together.
A *shot* is one repetition of a circuit followed by a measurement; the default hardware exercise uses 1024 shots.

Use current platform checks when planning capacity. A CPU job has previously completed through the Jiuding adapter,
and a Bell calculation has run on an A100 in an existing development environment.
An earlier request for a new GPU job remained queued for 180 seconds and was cancelled.
That observation alone does not establish whether GPU jobs will start promptly at the event.
Check the current [Jiuding guide](../../docs/guides/JIUDING.md) and repeat the test with the event configuration.

## Rehearse the participant journey

1. Record the software and image versions using the [environment notes](environment/README.md).
2. Sign in with a fresh participant account. Check that the exercises do not depend on an instructor's private files or credentials.
3. Run notebooks 01–03. Expect Bell probabilities near `[0.5, 0, 0, 0.5]` and a training loss below `0.01` after 80 steps.
4. Submit the CPU job in notebook 04. Confirm both `Succeed` and a readable result. Practice checking and cancelling a queued job.
5. If you will teach GPU submission, complete a new GPU job using the event image and confirm that its result reports CUDA.
6. Submit the 1024-shot task in notebook 05. Save its task ID and counts. Select a currently available backend rather than assuming a past mapping still works.
7. Load that result in notebook 06. Check its source label, task ID, and total count.
8. Repeat with the expected number of simultaneous users, or in the groups you plan to use on the day.
9. Run the local labs 07–12, 14–15, and 17–18; check their numerical assertions and charts. Lab 12 can be completed without hardware compilation.
10. Rehearse lab 13 with a newly trained parameter and a real hardware result, and lab 16 with its two-job CPU sweep. Confirm the appropriate submission switches are initially off.
11. Check that unwanted jobs have stopped and that participants know how to close their environments.

Record what passed and what still needs attention. Local notebook checks do not replace this rehearsal.

## Pilot with people who have not seen the notebooks

Invite a quantum beginner, a PyTorch user, and a scientific-computing or systems developer to try their chosen route.
Start from SETUP.md with a fresh kernel. Let each person read and attempt the task before offering an explanation;
record the first point where they need help, the exact error or confusing sentence, and the assistance you supplied.
Use [field notes](FIELD_NOTES.md) for their predictions and conclusions.

For each pilot, check whether the participant can:

- Explain the input, the output, and the numerical reference in their own words.
- Make one deliberate change, predict its effect, and interpret the observed result.
- Restart the notebook and reproduce the result without hidden state.
- Distinguish a local calculation, recorded data, and a live remote result.

For lab 07, ask which parameters learned and how they know. For lab 18, ask why gradient agreement matters beyond energy agreement.
For lab 13, ask why hardware counts should be compared with the frozen circuit's ideal prediction as well as the training target.
If a participant cannot answer, revise the relevant explanation and repeat that section with another reader.
Record the pilot outcome separately from automated execution results. Human usability testing is still pending until these sessions take place.

## Teach around queue delays

Submit hardware tasks before a discussion or another local exercise so the group has something useful to do while waiting.
Use hardware for a small number of deliberate experiments; do not ask participants to run a hardware optimization loop.
If a live result is unavailable, use the recorded example in notebook 06 and say clearly that it comes from an earlier run.
A replay keeps the analysis lesson moving, but it does not count as a successful live submission.

## Troubleshooting

| Symptom | What to check or do |
| --- | --- |
| Wrong FlagQuantum version | Check the `source` reported by preflight and the selected notebook kernel. |
| CUDA unavailable | Continue on CPU, then check the allocated device, driver, and image. Mark the GPU exercise as skipped. |
| Jiuding stays Pending | Keep the receipt and check status. Cancel if necessary; do not submit another job just to check progress. |
| File or Python executable missing | Check shared mounts and the executable path inside the job image. |
| Authentication fails | Ask platform support to check credentials and permissions privately. |
| QSteed compilation fails | Check the plugin version, selected backend, and current calibration. Read the reported error before retrying. |
| Quafu wait is interrupted | Keep the output file and check the platform task list. The submission may already have succeeded. |
| No hardware result arrives during class | Use the labeled replay and record which live exercise remains unfinished. |

## Help participants interpret results

A tiny Bell circuit illustrates device selection, not GPU speedup.
The counts in the computational (Z) basis show measurement probabilities; they do not by themselves establish entanglement or state fidelity.
For MPS extensions, explain that memory and accuracy depend on the circuit's entanglement and truncation settings.
Do not describe a favorable large-qubit example as support for arbitrary circuits of that size.

## Finish with something participants can keep

Ask each group to save its circuit, loss curve, cloud job ID, hardware task ID or replay label, and a few sentences explaining the result.
Keep personal outputs outside the course source. Before publishing a notebook, clear credentials, private paths, account details, and execution outputs.
