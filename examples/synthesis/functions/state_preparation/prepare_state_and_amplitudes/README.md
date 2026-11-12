# Native probability and amplitude preparation

Generate classical vectors in Python and pass them to the native quantum function
at synthesis time:

```sh
python -m examples.synthesis.functions.state_preparation.prepare_state_and_amplitudes.main
```

The workflow samples the nonuniform eight-outcome distribution, constructs and
normalizes a signed linear amplitude vector, then reads both raw amplitudes and
the Notebook's phase-aligned presentation. No quantum source importer is used.
See [the workflow guide](WORKFLOW.md) for scope and independent acceptance, and
[provenance](provenance.json) for the pinned source and reviewed code cells.

[Third-party license](../../../CLASSIQ_LICENSE.txt)
