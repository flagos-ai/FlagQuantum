# Local analytical resource dashboard

`physical_resource_dashboard.py` composes the existing FlagQuantum QEC resource
models with an optional Dash/Plotly interface. It does not run a quantum algorithm
or select an execution backend. No reference SDK is needed to serve it.

The verified presentation environment uses Dash 2.18.2 and Plotly 5.24.1, with
pandas and NumPy installed by the caller. These packages are optional example
dependencies, not new core requirements. From the repository root:

```bash
python -m examples.apps.physical_resource_dashboard
```

The app serves the loopback interface at `http://127.0.0.1:8050`. Algorithm counts
must be finite nonnegative integers; integer-valued JSON floats are accepted.
The factory replica count must be positive. The charts retain the original
analytical model conventions and their documented presentation limits.

The complete fourteen-function source map, exact callback/figure comparisons,
source defects and numerical dependency evidence are recorded in
`docs/development/PHYSICAL_RESOURCE_DASHBOARD_AUDIT.md`.
