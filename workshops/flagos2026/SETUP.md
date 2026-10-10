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
The `source_commit` field should match the revision announced by the instructor. Missing CUDA,
Quafu credentials, or the optional QSteed plugin does not prevent you from completing the local exercises.
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
Notebook 01 prints the imported package path, version, and source commit without installing or replacing the package.

If a kernel says `Disconnected`, select **FlagQuantum Workshop** again and restart it before running the notebook.
Use a clean kernel for each batch validation so imports and environment variables do not leak from an earlier run.

## Deploy only the workshop directory

The notebooks support both a full Git checkout and a deployment containing only the `flagos2026` directory.
For a source-based event image, install the reviewed checkout in editable mode and ensure its root is ahead of
site-packages. A Liangzhi Cloud deployment that checks out FlagQuantum at `/share/project/flagquantum` can use:

```bash
export PYTHONPATH=/share/project/flagquantum${PYTHONPATH:+:$PYTHONPATH}
export FLAGQUANTUM_SOURCE_COMMIT="$(git -C /share/project/flagquantum rev-parse HEAD)"
```

Put these settings in the kernel or environment startup configuration, not in notebook cells. If the event image
uses a wheel instead, record the exact wheel version and commit in `FLAGQUANTUM_SOURCE_COMMIT`. Do not run
`pip install flagquantum` inside a notebook: that can silently replace the version being tested.

## Before submitting a Jiuding job

The submitting environment and the job container must both be able to read your project and write its results.
This is what *shared storage* means in this workshop. Installing FlagQuantum on a laptop alone does not provide this access.

Your instructor will supply a compatible container image and confirm the Python executable inside it.
An image contains the software that a cloud job starts with. Its Python path may differ from the one on your laptop.
Existing Jiuding development environments can read their injected AK/SK credentials automatically.
See the [Jiuding guide](../../docs/guides/JIUDING.md) for configuration details.

## Before submitting a Quafu task

The hardware notebooks use the current Quafu Task API with server-side compilation. They read `QUAFU_API_KEY`
and, when a non-default endpoint is required, `QUAFU_TASK_SERVER_URL` from the process environment. The instructor
should supply these through the platform's secret settings. Do not type keys into notebook cells, print them, or
commit them to Git. The optional QSteed plugin and legacy `QUAFU_API_TOKEN` are not required for notebooks 05 and 13.
See the [Quafu guide](../../docs/guides/QUAFU_BACKEND.md) for details.

## When you finish

Check your Jiuding jobs and cancel any you no longer need. Confirm that they have actually stopped.
A timeout only means that your notebook stopped waiting; the cloud task may still be running.
Close your development environment according to the instructor's directions.
