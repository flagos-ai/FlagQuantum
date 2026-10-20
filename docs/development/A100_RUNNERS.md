# Shared A100 CPU CI

## Verified runner labels

On 2026-10-20, the refreshed repository runner settings listed
`a100-jiuding-0` and `a100-jiuding-1` as Idle in the organization Default group.
Both expose `self-hosted`, `Linux`, `X64`, `a100`, `jiuding`, and `gpu`.
CI targets that complete label set; either runner can accept a job.

The earlier local migration targeted `flagquantum-a100-cpu` and described
sixteen dedicated CPU registrations. That label and those registrations were
absent during the initial settings inspection. Later API checks found the
dedicated registrations partially online. They remain separate from the shared
Jiuding runners and are not prerequisites of this change.
The existing local GPU workflow retains its original hardware gates.

## First migration batch

The following jobs use the shared runners on main pushes and same-repository
pull requests:

- `quality`
- `cpu-core` (Python 3.10 and 3.11)
- `cpu-core-3-12-tests`
- `cpu-runtime`
- `cpu-core-3-12` (the existing required-check aggregator)
- `distributed-cpu`
- `pre-commit`

Fork pull requests use `ubuntu-latest`. Other jobs retain their existing runners.
Required check names, test commands, and hosted capacity buckets are unchanged.
No `pull_request_target` trigger executes fork code on a shared machine.

Each Python job creates a fresh venv without system site packages or inherited
Python search paths. Native library paths inside inherited Python package
directories are also removed so a host Torch library cannot override the CPU
wheel. Test jobs install CPU PyTorch and the declared setuptools/wheel build
tooling before installing the project. CUDA devices are hidden and
numerical library threads are limited to two. These settings do not impose host
resource isolation or prevent other repositories from competing for the same
machines. The two registrations do not prove two independent hosts or a CPU
quota, and jobs can queue when both runners are busy.

The first cold run spent about sixteen minutes preparing the interpreter,
framework, and editable package before the full unit suite. The full unit,
runtime integration, and distributed CPU jobs therefore have a sixty-minute
budget; shorter first-batch jobs retain forty-five minutes. A queued job does
not consume its execution budget.

Shared jobs disable setup-python's remote pip-cache upload: the persistent host
already retains its downloads, and its global pip cache can include several
GB of unrelated packages. Fork jobs retain the GitHub-hosted pip cache.

## Package mirrors

Self-hosted jobs use the TUNA PyPI mirror. CPU PyTorch wheels are selected
separately from the Aliyun CPU wheel directory with `--no-index --no-deps`.
The mirror download has a 180-second budget and uses a job-owned temporary
directory, so a timeout leaves no partially installed Torch package;
ordinary dependencies are then resolved through the job's PyPI mirror. This
keeps CUDA wheels out of candidate selection. If the mirror attempt fails or times out,
the action retries the official CPU index and reports the fallback in the log.
An import check requires a CPU-only Torch build after installation.

GitHub-hosted fork jobs retain the official PyPI and CPU PyTorch indexes.
The mirror settings are scoped to the job; no host pip configuration is changed.

The SJTU CPU index had matching official wheel hashes during local inspection,
but the shared host's outbound proxy rejected its CONNECT requests with HTTP
500 during CI. The production route therefore uses the Aliyun wheel directory
with the official CPU fallback instead.

Mirror references:

- https://developer.aliyun.com/mirror/pytorch-wheels
- https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

## Validation

Local configuration checks:

```bash
python -m pytest tests/unit/test_cpu_runner_routing.py \
  tests/unit/test_github_hosted_capacity_policy.py \
  tests/unit/test_shared_runner_checkout_policy.py -q
python tools/validate_required_checks.py
```

The isolation test executes the composite action twice, verifies distinct venvs,
and checks that an inherited host-only module cannot be imported.

After publishing this branch, open a same-repository pull request and inspect the
runner assigned to a first-batch job: it must be one of the two Jiuding runners.
The same job on a fork PR must use GitHub-hosted Ubuntu. Core jobs must pass
their existing assertions that NumPy, JAX, and external quantum frameworks are
absent. Configuration checks alone do not prove setup-python downloads, compiler
availability, network access, or host policy. Only a completed Actions run verifies
those conditions.
