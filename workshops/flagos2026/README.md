# FlagQuantum Workshop: choose your own experiment

Build quantum programs, train models, explore physics, and understand what happens between your code and a real device.
This is a collection of hands-on labs, not a course you must finish in one sitting.
Choose a question that interests you and follow its prerequisites. Beginners and experienced developers can start in different places.

All teaching material is in English. Each notebook writes the core computation in its cells, explains the expected result,
and ends with an experiment you can change yourself. The default calculations are small enough for a laptop CPU.

## Three experiments to remember

These labs connect a scientific question to a result you can inspect and explain.
Choose one after completing its prerequisites; each also works entirely on a laptop CPU.

| Experiment | What you will take away |
| --- | --- |
| [One loss, two learning systems](notebooks/training/07_hybrid_model.ipynb) | A fitted hybrid model, held-out predictions, and evidence that both classical and quantum parameters receive gradients. |
| [One problem, three simulators](notebooks/simulation/18_one_problem_three_simulators.ipynb) | The same energy-minimization problem solved with statevector, MPS, and tensor-network execution, with explicit value and gradient checks. |
| [From a learned angle to a measurement](notebooks/cloud/13_trained_hardware.ipynb) | A saved trained circuit and a comparison separating learning error from measurement variation, with optional live hardware execution. |

Use the [field notes](FIELD_NOTES.md) to record a prediction, explain a result, and decide what to change next.

## Find your starting point

| You want to… | Suggested route |
| --- | --- |
| Start from zero | 01 → 02 → 03 → 09 |
| Bring your PyTorch experience | 03 → 07 → 14 → 13 |
| Solve a small scientific problem | 03 → 08 → 18 → 11, then 10 |
| Explore optimization on graphs | 03 → 15 → 09 |
| Use cloud resources and real hardware | 04 → 16, then 05 → 06 → 13 |
| Understand the software underneath | 12 → 17, then 14 and 11 |

These arrows suggest useful background, not a shared notebook session.
Each lab creates its own variables and can run in a fresh kernel. Lab 06 can use a recorded hardware result;
its live option reads the result saved by lab 05.

## Lab catalog

Notebooks are grouped by learning topic. Keep the lab numbers when following
the routes above; each notebook still runs independently.

| Folder | Labs | Focus |
| --- | --- | --- |
| [basics](notebooks/basics/) | 01, 02, 09 | Circuits, device selection, and measurement uncertainty |
| [training](notebooks/training/) | 03, 07, 14 | Parameter learning, hybrid models, and gradient checks |
| [simulation](notebooks/simulation/) | 10, 11, 18 | Noise, MPS, and agreement across simulation representations |
| [compilation](notebooks/compilation/) | 12, 17 | Circuit transformations and compiler extensions |
| [cloud](notebooks/cloud/) | 04, 05, 06, 13, 16 | Jiuding, Quafu, and interpretation of remote results |
| [applications](notebooks/applications/) | 08, 15 | VQE and graph optimization |

| Lab | Question you will answer | Level | Additional resources |
| --- | --- | --- | --- |
| [01 · First circuit](notebooks/basics/01_first_circuit.ipynb) | How does each gate change the state? | Beginner | None |
| [02 · CPU and GPU](notebooks/basics/02_cpu_gpu.ipynb) | Can I run the same circuit on another device? | Beginner | Optional CUDA GPU |
| [03 · Training from scratch](notebooks/training/03_quantum_training.ipynb) | How does a quantum parameter learn? | Beginner | None |
| [04 · A cloud job](notebooks/cloud/04_jiuding_jobs.ipynb) | How do I submit my own Python program and retrieve its result? | Intermediate | Jiuding for submission |
| [05 · Quantum hardware](notebooks/cloud/05_quafu_hardware.ipynb) | How do I execute the circuit I wrote on a QPU? | Intermediate | Quafu and QSteed for submission |
| [06 · Result comparison](notebooks/cloud/06_compare_results.ipynb) | What do hardware counts tell me? | Beginner | Recorded data included |
| [07 · Hybrid model](notebooks/training/07_hybrid_model.ipynb) | Can one optimizer train classical and quantum layers? | Intermediate | None |
| [08 · VQE](notebooks/applications/08_vqe.ipynb) | Can a circuit learn a low-energy state? | Intermediate | None |
| [09 · Shot uncertainty](notebooks/basics/09_shots.ipynb) | How many measurements do I need? | Beginner/intermediate | None |
| [10 · Noise](notebooks/simulation/10_noise.ipynb) | How does relaxation change the answer? | Intermediate | None |
| [11 · MPS](notebooks/simulation/11_mps.ipynb) | When can an entangled state be stored compactly? | Advanced | None |
| [12 · Compilation](notebooks/compilation/12_compilation.ipynb) | What can a compiler remove without changing the result? | Intermediate | Optional QSteed and calibration access |
| [13 · Learned parameter on hardware](notebooks/cloud/13_trained_hardware.ipynb) | Does a locally trained circuit give the expected hardware measurement? | Intermediate | Optional Quafu and QSteed |
| [14 · Gradient checks](notebooks/training/14_gradients.ipynb) | How can I verify a gradient before trusting training? | Advanced | None |
| [15 · QAOA and Max-Cut](notebooks/applications/15_maxcut.ipynb) | How does a graph problem become a circuit objective? | Intermediate/advanced | None |
| [16 · Cloud sweep](notebooks/cloud/16_cloud_sweep.ipynb) | How do I manage several reproducible experiments? | Advanced | Jiuding for submission |
| [17 · Compiler extension](notebooks/compilation/17_extension.ipynb) | Can I implement and test my own compiler transformation? | Advanced | None |
| [18 · Three simulators](notebooks/simulation/18_one_problem_three_simulators.ipynb) | Do different simulation representations agree on energies, gradients, and training? | Intermediate/advanced | None |

## Coverage and next topics

These 18 labs introduce the main local learning and remote execution workflows;
they are not a complete tour of every FlagQuantum module or API. In particular,
cloud job submission is not a lesson in partitioning one state across devices,
and saving a trained circuit is not a deployment-package tutorial.

The following topics have reference material but no dedicated workshop lab yet:

| Topic | Continue with |
| --- | --- |
| Circuit drawing and presentation | [Drawer](../../flagquantum/drawer/README.md) |
| Execution planning and runtime controls | [Runtime](../../flagquantum/runtime/README.md) |
| Distributed statevector and MPS training | [Statevector examples](../../examples/distributed_statevector_topologies/README.md), [MPS examples](../../examples/distributed_mps/README.md) |
| Deployment packages | [Capability catalog](../../docs/generated/CAPABILITIES.md) |
| Calibration-based device models and experiment validation | [Digital twin](../../flagquantum/twin/README.md) |
| Repetition-code memory experiments and decoding | [QEC](../../flagquantum/qec/README.md) |
| Optional JAX execution and other framework adapters | [JAX](../../flagquantum/simulation/jax/README.md), [Ecosystem](../../flagquantum/ecosystem/README.md) |

Use the [capability catalog](../../docs/generated/CAPABILITIES.md) for support
boundaries. Future labs should teach a runnable task with an interpretable
result; module coverage alone is not a reason to add another notebook.

## Get ready

Follow [Setup](SETUP.md), select the workshop Python kernel, and run:

```bash
python workshops/flagos2026/scripts/preflight.py
```

A kernel is the Python process that executes notebook cells. Open a lab and run one cell at a time with Shift+Enter.
Read the prediction or question before executing its cell. When changing a model or a circuit-building step, restart from its initialization cell.
The standalone scripts in `scripts/` are reference programs for later use; you do not need to import them to complete the labs.

## How the cloud services fit together

**FlagQuantum** is the Python framework used throughout these experiments.
**Liangzhi Cloud** is the event portal; the instructor supplies its address and navigation.
**Jiuding** runs Python jobs on classical CPUs and GPUs.
**Quafu** executes quantum circuits on hardware, and **QSteed** compiles circuits for a selected device.

Cloud and hardware actions are disabled until you enable the relevant switch in a notebook.
Lab 12's optional hardware compilation can contact the calibration service without submitting an execution task.
A local pass does not establish that the event has available GPU or QPU resources.

Lab 13 trains a parameter locally and optionally submits the frozen circuit to hardware.
It does not run a hardware training loop. Lab 06 labels recorded data as `historical_replay` and never presents it as your live run.

## Build something of your own

See the [challenge board](CHALLENGES.md) for open-ended projects at different levels.
Keep your outputs in `outputs/`, which is excluded from Git. Save enough information for another participant to reproduce your result.
A useful final artifact is a short explanation with a circuit, a chart, and a numerical check—not just a screenshot saying that code ran.

Instructors can use the [teaching guide](INSTRUCTOR.md), [environment notes](environment/README.md),
and [validation record](VALIDATION.md) to prepare the venue and choose demonstrations.
