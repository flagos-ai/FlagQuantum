# Prepare a reproducible environment

Use Python 3.12 for this course. `requirements.txt` lists the notebook and plotting tools;
FlagQuantum's dependencies come from the chosen repository version.
The dependency ranges are a starting point, not a tested event lock file.

After a successful rehearsal, run these commands in the clean event environment and keep the files with your instructor materials:

```bash
git rev-parse HEAD > workshop-revision.txt
python -m pip freeze > workshop-environment.txt
python workshops/flagos2026/scripts/preflight.py > workshop-preflight.json
```

Also record the container image address and digest, QSteed plugin version or commit, operating system,
GPU driver and CUDA versions, Jiuding queue, Quafu backend, and the time you checked them.
An image digest identifies the exact image contents, even if a tag is later changed.

Inspect `pip freeze` output before sharing it: editable installs can include private paths or package URLs.
Choose a GPU image that matches the platform's driver and CUDA setup, and rehearse each image you intend to use.
A successful laptop CPU test does not validate a GPU container.
