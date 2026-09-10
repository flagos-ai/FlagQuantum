# Set up your workshop environment

Choose one of the two routes below. If you are attending the event, use the environment your instructor provides.

## Use the event environment

1. Open the Liangzhi Cloud link supplied by your instructor and sign in.
2. Follow the instructor's directions to open your assigned development environment.
3. Open a terminal in the FlagQuantum repository and run:

```bash
python workshops/flagos2026/scripts/preflight.py
```

This check runs a small CPU calculation and reports your Python version, FlagQuantum installation,
and available devices. It checks whether credentials are present without displaying them.
It does not contact a cloud service or submit a task.

Look for a successful `cpu_check`. The `source` field should point to the course's copy of FlagQuantum.
Missing CUDA, a Quafu token, or the QSteed plugin does not prevent you from completing the local exercises.
Ask the instructor to help configure these before trying the corresponding remote exercise.

Open notebook 01 and select the **FlagQuantum Workshop** kernel, or the equivalent kernel supplied by the instructor.
The exact portal address and navigation depend on the event deployment.

## Use your own computer

Use Python 3.12. From the repository root, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r workshops/flagos2026/environment/requirements.txt
python -m ipykernel install --user --name flagquantum-workshop --display-name 'FlagQuantum Workshop'
python workshops/flagos2026/scripts/preflight.py
python -m jupyterlab workshops/flagos2026/notebooks
```

These shell commands are for macOS or Linux. On Windows, create the environment with Python 3.12
and activate it using `.venv\Scripts\Activate.ps1` in PowerShell before running the remaining Python commands.

In Jupyter, select **FlagQuantum Workshop** as the notebook kernel.
If imports fail or show an unexpected version, check the kernel first: it may be using a different Python installation.

## Before submitting a Jiuding job

The submitting environment and the job container must both be able to read your project and write its results.
This is what *shared storage* means in this workshop. Installing FlagQuantum on a laptop alone does not provide this access.

Your instructor will supply a compatible container image and confirm the Python executable inside it.
An image contains the software that a cloud job starts with. Its Python path may differ from the one on your laptop.
Existing Jiuding development environments can read their injected AK/SK credentials automatically.
See the [Jiuding guide](../../docs/guides/JIUDING.md) for configuration details.

## Before submitting a Quafu task

The environment needs a compatible `flagquantum-compiler-qsteed` plugin and a valid `QUAFU_API_TOKEN`.
The instructor should install the plugin and supply credentials through the platform's secret settings or process environment.
Do not type tokens into notebook cells or commit them to Git.
See the [Quafu guide](../../docs/guides/QUAFU_BACKEND.md) for details.

## When you finish

Check your Jiuding jobs and cancel any you no longer need. Confirm that they have actually stopped.
A timeout only means that your notebook stopped waiting; the cloud task may still be running.
Close your development environment according to the instructor's directions.
