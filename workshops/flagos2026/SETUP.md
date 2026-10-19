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

### Published examples versus the event deployment

The published 05/06 notebooks keep the visible submission switches `False` so a
downloaded example cannot consume quota accidentally. The instructor's event copy
must set `submit_to_cloud = True` and `submit_to_hardware = True` after verifying
the image, project, queue, credentials and target. Do not enable submission merely
because credentials exist. Each lesson prints its active configuration before
submission; no credential values are printed. Event acceptance requires real
submissions and recovery checks, not just a successful local rehearsal.

The asymmetric lessons use new receipt names. Keep old Bell receipts and results;
they are a different experiment and must not be reused or relabeled. Switching
between English and Chinese editions of the same lesson restores the same saved
task. Updating the source does not update already deployed personal copies or images.

Notebooks 05/06 require a matching SDK that supports `fq.submit(..., receipt=...)`.
Update the SDK, image and teaching templates together; replacing notebook files
alone cannot add this parameter to an older installed SDK. Repeating the same
call restores a matching request without another submission. Keep the receipt
and its SDK-created journal in your personal persistent output directory.
Existing manual receipts remain usable through `fq.restore_job(path)` but are
not silently adopted without request context. Never remove old receipts,
submission records or results to force another task; reconcile uncertain
attempts with the platform first.

The submitting environment and the job container must both be able to read your project and write its results.
This is what *shared storage* means in this workshop. Installing FlagQuantum on a laptop alone does not provide this access.

Your instructor will supply a compatible container image and confirm the Python executable inside it.
An image contains the software that a cloud job starts with. Its Python path may differ from the one on your laptop.
The Jiuding notebook (05) automatically prepares a personal copy using the authenticated platform user ID, never
Linux UID or workspace owner as identity. It prefers verified personal `/home/<name>` storage;
the fallback is `/share/project/flagquantum/<ID>/`. `mountPointPath` can be empty for platform defaults:
directory types 2/4 are matched against kernel bind roots containing the project IDs and user ID/PUBLIC.
Unverified storage stops submission. Open the prepared copy shown by the first cell; no ID/path edits
are needed. Existing personal edits are preserved; outputs, .env and notebook caches are not copied.
Local outputs/receipt pointers live in the personal copy and each task has a fresh `jobs/bell-.../` folder.
This prevents accidental collisions, not unauthorized access: per-user credentials and platform ACLs
or individual mounts remain required. Shared credentials and multiple privileged root containers
cannot be securely isolated by directory names or permissions alone.
The image must contain FlagQuantum; the notebook does not inject its local source tree into the worker.
Private catalog images use `image_region="PRIVATE"` by default. If staging reports no verified writable
mount, the instructor must attach shared storage at the same container path in the development environment
and its jobs. Creating an ordinary local folder does not fix missing shared storage. This preflight checks
the inherited configuration, not live visibility inside a worker; it does not start a probe job.
Existing Jiuding development environments can read their injected AK/SK credentials automatically.
See the [Jiuding guide](../../docs/guides/JIUDING.md) for configuration details.

## Before submitting a Quafu task

The hardware notebooks use the current Quafu Task API with server-side compilation. They read `QUAFU_API_KEY`
and, when a non-default endpoint is required, `QUAFU_TASK_SERVER_URL` from the process environment. The instructor
should supply these through the platform's secret settings. Do not type keys into notebook cells, print them, or
commit them to Git. The optional QSteed plugin and legacy `QUAFU_API_TOKEN` are not required for example 06 or reference notebook 13.
See the [Quafu guide](../../docs/guides/QUAFU_BACKEND.md) for details.

## When you finish

Check your Jiuding jobs and cancel any you no longer need. Confirm that they have actually stopped.
A timeout only means that your notebook stopped waiting; the cloud task may still be running.
Close your development environment according to the instructor's directions.
