# Local compute helpers

This directory owns private helpers for resources controlled directly by the
current process. Its first migrated implementation observes allocator peaks
through the existing Compute platform boundary.

It does not own training algorithms, execution policy, remote submission, or
public contracts. Dependencies are Torch and the existing parent Compute
contracts and registry. Runtime executors consume the private helper; user entry
points remain the existing `flagquantum` APIs. This directory exports no public
API. Other direct-device implementations remain in the parent package until
their complete reference closures can migrate together.

## Ten-minute contributor path

1. Read `_memory.py` and the existing platform contract in `../contracts.py`.
2. Run `python -m pytest tests/unit/test_training_peak_memory.py -q` from the
   repository root. The tests demonstrate CPU behavior and substitute a FlagOS
   platform without changing the consumers, checking synchronize/read/reset
   ordering and the requested device index.
3. Make a focused allocator-observation change and rerun those tests, then run:

   ```sh
   python -m pytest tests/unit/test_statevector_training.py tests/unit/test_mps_training.py tests/integration/test_training_parameter_lifecycle.py -q
   ```

A successful CPU test run provides no evidence of physical accelerator support
or distributed scalability.
