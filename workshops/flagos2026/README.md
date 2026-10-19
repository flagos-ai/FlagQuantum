# FlagQuantum: six experiments to keep

A small collection of complete examples, not an eighteen-stop API tour.
Learn a task here, then reuse the same FlagQuantum code in your own Python project.

## Choose a learning route

- First circuit: **01**, then **02** when you know basic PyTorch gradients.
- Hybrid training: **01 → 02 → 03**; familiarity with Linear, Sequential and batches helps.
- Simulation choices: **01 → 02 → 04**; keep the default mode until you have a reason to compare representations.
- Remote execution: **01 → 05 → 06**; the training lessons are not prerequisites.

01–04 run on a local CPU without a cloud account. 05 requires Jiuding access,
a verified job image/project/queue and shared storage. 06 requires Quafu access,
an available hardware target and shot quota. Follow [Setup](SETUP.md) before
either remote lesson. The published switches stay off; the event deployment
explicitly enables real submissions after configuration is verified.

Open the English or Chinese link in the table below. Use a fresh kernel, read
the learning goal, and run cells in order with Shift+Enter. Each lesson explains
the expected output and how to rerun safely; an assertion passing is a numerical
check, not proof that the explanation is clear to a new reader.

## The main workshop

Start with the circuit example. Choose the next experiment by the question you
want to answer; every notebook creates its own state and runs in a fresh kernel.

| Example | The question | Standard APIs to remember | Evidence you should obtain |
| --- | --- | --- | --- |
| [Build and measure a Bell circuit](notebooks/01_first_circuit.ipynb) | What do amplitudes and probabilities mean? | `fq.Circuit`, gates, `fq.run` | The expected Bell probabilities and a checked plot |
| [Train a quantum parameter](notebooks/02_quantum_training.ipynb) | How does a circuit learn a target? | `fq.Module`, PyTorch gradients and optimizer | A finite gradient and decreasing loss |
| [Combine classical and quantum layers](notebooks/03_hybrid_model.ipynb) | Can one optimizer train both? | `fq.Module`, `torch.nn.Sequential` | Held-out predictions and updates in both layers |
| [One objective, three simulators](notebooks/04_one_problem_three_simulators.ipynb) | Can I change representation without changing the problem? | `fq.run`, `fq.expectation`, `fq.ExecutionOptions` | Independent energy reference, gradient agreement and convergence |
| [Run a circuit on Jiuding](notebooks/05_jiuding_jobs.ipynb) | How is remote execution different from local execution? | `fq.submit`, `job.status`, `job.wait`, receipts | A real CPU job and local/remote agreement |
| [Measure on Quafu](notebooks/06_quafu_hardware.ipynb) | How do hardware counts differ from an ideal prediction? | `fq.submit`, `fq.counts`, `job.result` | A real task ID, shot accounting and interpretable counts |

The first four examples need only a laptop CPU. The final two require access to
the named service; submitting consumes real resources. A disabled switch is a
local rehearsal, not evidence that a cloud job ran. Queue waits are not failures,
and a task ID alone does not prove completion.

The six notebooks sit directly in notebooks/, numbered 01–06 in learning order.
Optional material lives separately in reference_notebooks/; its original numbers
are reference identifiers, not additional course steps.

## How to use each example

### Choose your teaching language

The English edition remains in `notebooks/`; the Chinese teaching edition lives
alongside it in `notebooks_cn/`. Both have the same six numbered examples,
identical executable cells, and identical take-home Python snippets.

| Example | English | Chinese |
| --- | --- | --- |
| 01: Bell circuit | [Open](notebooks/01_first_circuit.ipynb) | [Open](notebooks_cn/01_first_circuit.ipynb) |
| 02: Quantum training | [Open](notebooks/02_quantum_training.ipynb) | [Open](notebooks_cn/02_quantum_training.ipynb) |
| 03: Hybrid model | [Open](notebooks/03_hybrid_model.ipynb) | [Open](notebooks_cn/03_hybrid_model.ipynb) |
| 04: Three simulators | [Open](notebooks/04_one_problem_three_simulators.ipynb) | [Open](notebooks_cn/04_one_problem_three_simulators.ipynb) |
| 05: Jiuding CPU job | [Open](notebooks/05_jiuding_jobs.ipynb) | [Open](notebooks_cn/05_jiuding_jobs.ipynb) |
| 06: Quafu hardware | [Open](notebooks/06_quafu_hardware.ipynb) | [Open](notebooks_cn/06_quafu_hardware.ipynb) |

Changing language does not reset an experiment: both editions use the same
personal output directory and matching cloud receipts. This avoids submitting
the same saved task again just because you switched languages. New personal
copies include both editions; existing edited copies are never overwritten.
Setup guides and code output remain English. Existing workshop images need a
separate template update before the Chinese edition becomes available in them.

### Run an experiment

1. Read the question and predict the answer.
2. Run the cells in order with Shift+Enter in a fresh kernel.
3. Inspect the numerical check and figure, not just the absence of an exception.
4. Change one parameter, rerun from initialization, and explain the difference.
5. Copy the **Take it home** example into your own Python file.

Environment diagnostics and file exports are folded and labeled as setup or
optional material. They are not another quantum API. The main computation stays
in the notebook and uses `import flagquantum as fq`; there is no `lesson.submit`
or alternate workshop execution interface.

Follow [Setup](SETUP.md) for installation and the event environment. Notebook
cells never install packages or accept credentials. The event image supplies
platform configuration; a local user configures service access outside source.
The workshop directory/identity helper is only for the venue's file layout.

## Quality bar for a main example

A main example must have one clear question, a complete experiment, and a result
the reader can interpret. It must run independently, show the standard API,
include an independent numerical check where applicable, and end with a small
reusable example. Local computation must work without cloud credentials or
accelerator hardware. A failure must remain visible; do not silently substitute
a backend or label recorded data as a live run.

Cloud examples additionally identify the exact submission cell, save a receipt
before waiting, distinguish queued/failed/completed states, and prevent accidental
duplicate submissions. Supporting storage and recovery code stays in folded
cells, with its purpose explained rather than hidden behind a new execution API.

Validation records must state source/version, kernel, hardware, live task IDs,
retries and limitations. A local preview, a mocked provider test and a real
cloud submission are different evidence. See [Validation](VALIDATION.md).

After installing [the teaching tools](environment/requirements.txt), verify inline
plot rendering with `python scripts/verify_inline_plots.py --kernel python3` from
this workshop directory. This starts bounded Jupyter probes for the six plotting
examples in both editions (use `--language en` or `--language cn` for just one),
checks for PNG output even when the environment defaults to Agg, and
never submits a cloud task or modifies the source notebooks. Ordinary core unit
tests check the plotting setup without depending on Jupyter packages.

## Reference library, not required coursework

The remaining notebooks are retained for deeper questions; they are not extra
steps beginners must complete. No existing learning material is deleted.

- Devices and finite shots: [02](reference_notebooks/basics/02_cpu_gpu.ipynb), [09](reference_notebooks/basics/09_shots.ipynb).
- Physics applications: [08 VQE](reference_notebooks/applications/08_vqe.ipynb), [15 Max-Cut](reference_notebooks/applications/15_maxcut.ipynb).
- Noise and compact states: [10](reference_notebooks/simulation/10_noise.ipynb), [11 MPS](reference_notebooks/simulation/11_mps.ipynb).
- Gradient verification: [14](reference_notebooks/training/14_gradients.ipynb).
- Compilation and extensions: [12](reference_notebooks/compilation/12_compilation.ipynb), [17](reference_notebooks/compilation/17_extension.ipynb).
- Hardware comparison and deployment of a trained angle: [06](reference_notebooks/cloud/06_compare_results.ipynb), [13](reference_notebooks/cloud/13_trained_hardware.ipynb).
- Multiple cloud experiments: [16](reference_notebooks/cloud/16_cloud_sweep.ipynb).

[Cloud job field notes](CLOUD_JOBS.md) cover complete Python programs, accelerator
selection, raw public-API examples and recovery. [Field notes](FIELD_NOTES.md)
help explain a scientific result; [Challenges](CHALLENGES.md) suggest extensions.

## Instructor responsibilities

Before teaching, ask someone new to FlagQuantum to follow a route without help.
After each lesson, ask them to explain what the circuit does, what the printed
result means, and what rerunning the relevant cell will do. For cloud lessons,
also ask whether they have submitted, queued, restored or retrieved a task.
Record the first confusing cell and revise it; do not mark this reader check
complete on the basis of automated execution or a mock submission test.

Provision the kernel, optional compiler plugin, personal mounts and service
configuration before teaching. Do not require participants to debug deployment
plumbing in the main experiment. Personal directory names avoid collisions but
do not replace platform ACLs; two-account isolation needs separate verification.

Use [Instructor notes](INSTRUCTOR.md) and [Environment notes](environment/README.md).
CPU examples do not establish GPU support or distributed scalability. The three
simulator example checks CPU correctness and gradient agreement, not performance.
