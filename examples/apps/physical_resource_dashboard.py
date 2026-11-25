#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Interactive physical-resource estimates through native QEC domain models.

Optional example dependencies are Dash 2.18.2, Plotly 5.24.1 and pandas.
Run this file directly to serve the local loopback interface.
"""

from collections.abc import Iterator, Sequence
from importlib import import_module
from math import isfinite
from types import ModuleType
from typing import Any

from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources import beverland, physical_search, rotation_synthesis
from flagquantum.qec.resources.ccz_to_t_factory import CCZToTFactory
from flagquantum.qec.resources.data_layout import SquareDataLayout
from flagquantum.qec.resources.error_model import LogicalErrorModel, SurfaceCodeScheme
from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
from flagquantum.qec.resources.magic_state_factory import MagicStateFactory
from flagquantum.qec.resources.parallel_factory import ParallelFactory
from flagquantum.qec.resources.physical_parameters import PhysicalParameters

# All optional presentation packages remain at this example boundary.
np = import_module("numpy")
pd = import_module("pandas")
px = import_module("plotly.express")
go = import_module("plotly.graph_objects")
dash = import_module("dash")
ALL, Dash, dcc, html, Input, Output = (
    dash.ALL,
    dash.Dash,
    dash.dcc,
    dash.html,
    dash.Input,
    dash.Output,
)
PreventUpdate = import_module("dash.exceptions").PreventUpdate


def get_objects(
    modules: Sequence[ModuleType], object_type: type[Any]
) -> Iterator[tuple[str, Any]]:
    """Get all objects of a given type from a list of modules."""
    for module in modules:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isinstance(obj, object_type):
                yield obj_name, obj


_QEC_SCHEMES = {
    "BeverlandEtAl": SurfaceCodeScheme.beverland(),
    "GidneyFowler": SurfaceCodeScheme.gidney_fowler(),
}
_MAGIC_FACTORIES = {
    "FifteenToOne733": FifteenToOneFactory(7, 3, 3),
    "FifteenToOne933": FifteenToOneFactory(9, 3, 3),
    "GidneyFowlerCCZ": CCZToTFactory(),
}
_ROTATION_MODELS = dict(
    get_objects([rotation_synthesis], rotation_synthesis.RotationCostModel)
)

_GIDNEY_FOWLER_MODEL = "GidneyFowler (arxiv:1812.01238)"
_BEVERLAND_MODEL = "Beverland et al (arxiv:2211.07629)"
_SUPPORTED_ESTIMATION_MODELS = [_GIDNEY_FOWLER_MODEL, _BEVERLAND_MODEL]

_ALGORITHM_INPUTS = [Input({"type": "algorithm", "property": ALL}, "value")]
_QEC_INPUTS = [Input("QEC", "value")]
_MAGIC_FACTORIES_INPUTS = [
    Input("_MAGIC_FACTORIES", "value"),
    Input("_MAGIC_FACTORIES_number", "value"),
]
_ROTATION_MODELS_INPUTS = [Input("rotation-cost-model", "value")]
_ESTIMATION_INPUTS = [Input("estimation_model", "value")]
_ERROR_INPUTS = [Input("physical_error_rate", "value"), Input("error_budget", "value")]

_ALL_INPUTS = [
    *_ERROR_INPUTS,
    *_ESTIMATION_INPUTS,
    *_ALGORITHM_INPUTS,
    *_QEC_INPUTS,
    *_MAGIC_FACTORIES_INPUTS,
    *_ROTATION_MODELS_INPUTS,
]


def algorithm_summary_components() -> list[Any]:
    return [
        html.P("Enter a summary of the quantum algorithm/circuit"),
        html.Table(
            [
                html.Tr(
                    [
                        html.Td("Qubits"),
                        dcc.Input(
                            debounce=True,
                            id={"type": "algorithm", "property": "qubits"},
                            type="number",
                            min=0,
                            style={"marginLight": "10px"},
                            placeholder="Number of Qubits",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Measurements"),
                        dcc.Input(
                            debounce=True,
                            id={"type": "algorithm", "property": "Measurements"},
                            type="number",
                            min=0,
                            style={"marginLight": "10px"},
                            placeholder="Number of Measurements",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Ts"),
                        dcc.Input(
                            debounce=True,
                            id={"type": "algorithm", "property": "Ts"},
                            type="number",
                            min=0,
                            style={"marginLight": "10px"},
                            placeholder="Number of T gates",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Toffolis"),
                        dcc.Input(
                            debounce=True,
                            id={"type": "algorithm", "property": "Toffolis"},
                            type="number",
                            min=0,
                            style={"marginLight": "10px"},
                            placeholder="Number of Toffolis",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Rotations"),
                        dcc.Input(
                            debounce=True,
                            id={"type": "algorithm", "property": "Rotations"},
                            type="number",
                            min=0,
                            style={"marginLight": "10px"},
                            placeholder="Number of Rotations",
                            value=10,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Rotation Circuit Depth"),
                        dcc.Input(
                            debounce=True,
                            id={"type": "algorithm", "property": "Rotation Depth"},
                            type="number",
                            min=0,
                            style={"marginLight": "10px"},
                            placeholder="Rotation Circuit Depth",
                            value=10,
                        ),
                    ]
                ),
            ]
        ),
    ]


def factory_components() -> list[Any]:
    return [
        html.P("Enter number of magic state factories:"),
        dcc.RadioItems(
            id="_MAGIC_FACTORIES",
            options=sorted(_MAGIC_FACTORIES),
            value=next(iter(_MAGIC_FACTORIES.keys())),
        ),
        dcc.Input(
            debounce=True, id="_MAGIC_FACTORIES_number", type="number", min=1, value=1
        ),
        html.Details(
            [
                html.Summary("Magic Factories' information"),
                html.Table(
                    [
                        html.Tr(
                            [html.Td(k), html.Td(str(v), style={"padding": "10px"})]
                        )
                        for k, v in _MAGIC_FACTORIES.items()
                    ]
                ),
            ]
        ),
    ]


def qec_summary_components() -> list[Any]:
    return [
        html.P("Select a QEC algorithm:"),
        dcc.RadioItems(
            id="QEC",
            options=sorted(_QEC_SCHEMES),
            value=next(iter(_QEC_SCHEMES.keys())),
        ),
        html.Details(
            [
                html.Summary("QECs' information"),
                html.Table(
                    [
                        html.Tr(
                            [html.Td(k), html.Td(str(v), style={"padding": "10px"})]
                        )
                        for k, v in _QEC_SCHEMES.items()
                    ]
                ),
            ]
        ),
    ]


def rotation_cost_model_components() -> list[Any]:
    return [
        html.P("Select Rotation Cost Model:"),
        dcc.RadioItems(
            id="rotation-cost-model",
            options=sorted(_ROTATION_MODELS),
            value=next(iter(_ROTATION_MODELS.keys())),
        ),
        html.Details(
            [
                html.Summary("Models' information"),
                html.Table(
                    [
                        html.Tr(
                            [html.Td(r), html.Td(str(v), style={"padding": "10px"})]
                        )
                        for r, v in _ROTATION_MODELS.items()
                    ]
                ),
            ]
        ),
    ]


def input_components() -> list[Any]:
    return [
        html.Table(
            [
                html.Tr(
                    [
                        html.Td("Enter physical error rate"),
                        dcc.Input(
                            debounce=True,
                            id="physical_error_rate",
                            type="number",
                            min=0,
                            max=1,
                            style={"marginLeft": "10px"},
                            value=1e-4,
                        ),
                    ]
                ),
                html.Tr(
                    [
                        html.Td("Enter error budget"),
                        dcc.Input(
                            debounce=True,
                            id="error_budget",
                            type="number",
                            min=0,
                            max=1,
                            style={"marginLeft": "10px"},
                            value=1e-3,
                        ),
                    ]
                ),
            ]
        ),
        html.P("Select Estimation Cost Model:"),
        dcc.RadioItems(
            id="estimation_model",
            options=_SUPPORTED_ESTIMATION_MODELS,
            value=_GIDNEY_FOWLER_MODEL,
        ),
        *algorithm_summary_components(),
        *qec_summary_components(),
        *factory_components(),
        *rotation_cost_model_components(),
    ]


def output_components() -> list[Any]:
    return [
        html.Table(
            [
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    "Total Number of Toffoli gates", id="magic-name"
                                ),
                                html.Td(
                                    dcc.Input(
                                        id="total_magic",
                                        type="text",
                                        value="",
                                        readOnly=True,
                                    )
                                ),
                            ]
                        ),
                        html.Tr(
                            [
                                html.Td("#magic factories for min runtime"),
                                html.Td(
                                    dcc.Input(
                                        id="min_factory_count", value="", readOnly=True
                                    )
                                ),
                            ],
                            id="magic-factory-container",
                        ),
                        html.Tr(
                            [
                                html.Td("Duration"),
                                html.Td(
                                    dcc.Input(
                                        id="duration",
                                        type="text",
                                        value="",
                                        readOnly=True,
                                    )
                                ),
                            ],
                            id="duration-container",
                        ),
                    ]
                )
            ]
        )
    ]


app = Dash(__name__)
app.title = "FlagQuantum Resource Estimation"
app.layout = html.Div(
    [
        html.H4("Interactive QEC overhead estimation"),
        html.Table(
            [
                html.Tr(
                    [
                        html.Td(input_components(), style={"width": "30vw"}),
                        html.Td(output_components(), style={"width": "20vw"}),
                        html.Td(
                            dcc.Loading(
                                dcc.Graph(id="qubit-pie-chart"),
                                type="cube",
                                style={"max-width": "80vw", "float": "right"},
                            )
                        ),
                    ]
                )
            ],
            style={"width": "100vw"},
        ),
        html.Div(
            id="runtime-container",
            children=[
                dcc.Loading(
                    dcc.Graph(id="runtime-plot"),
                    type="cube",
                    style={"max-width": "80vw"},
                )
            ],
        ),
    ]
)


def create_qubit_pie_chart(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: ModeledResources,
    magic_factory: MagicStateFactory,
    magic_count: int,
    n_logical_gates: GateCost,
) -> Any:
    """Create a pie chart of the physical qubit utilization."""
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        res, factory, _ = physical_search.search_gidney_fowler_resources(
            logical=ModeledResources(
                n_logical_gates, int(algorithm.peak_logical_qubits)
            ),
            physical_error=physical_error_rate,
            error_budget=error_budget,
            factories=[
                ParallelFactory(f, int(magic_count))
                for f in physical_search.iter_ccz_factories()
            ],
        )
        memory_footprint = pd.DataFrame(columns=["source", "qubits"])
        memory_footprint["source"] = [
            "logical qubits + routing overhead",
            "Magic State Distillation",
        ]
        memory_footprint["qubits"] = [
            res.physical_qubits - factory.n_physical_qubits(),
            factory.n_physical_qubits(),
        ]
        fig = px.pie(
            memory_footprint,
            values="qubits",
            names="source",
            title="Physical Qubit Utilization",
        )
        fig.update_traces(textinfo="value")
        return fig
    multi_factory = ParallelFactory(magic_factory, int(magic_count))
    memory_footprint = pd.DataFrame(columns=["source", "qubits"])
    memory_footprint["source"] = [
        "logical qubits + routing overhead",
        "Magic State Distillation",
    ]
    memory_footprint["qubits"] = [
        SquareDataLayout.tile_count(int(algorithm.peak_logical_qubits)),
        multi_factory.n_physical_qubits(),
    ]
    fig = px.pie(
        memory_footprint,
        values="qubits",
        names="source",
        title="Physical Qubit Utilization",
    )
    fig.update_traces(textinfo="value")
    return fig


def format_duration(duration: Sequence[float]) -> tuple[str, Sequence[float]]:
    """Returns a tuple of the format (unit, duration)

    Finds the best unit to report `duration` and assumes that `duration` is initially in us.
    """
    unit = "us"
    if duration[0] > 86400_000_000:
        duration = [d / 86400_000_000 for d in duration]
        unit = "days"
    elif duration[0] > 3600_000_000:
        duration = [d / 3600_000_000 for d in duration]
        unit = "hours"
    elif duration[0] > 60_000_000:
        duration = [d / 60_000_000 for d in duration]
        unit = "min"
    elif duration[0] > 1000_000:
        duration = [d / 1000_000 for d in duration]
        unit = "s"
    elif duration[0] > 1000:
        duration = [d / 1000 for d in duration]
        unit = "ms"
    return unit, duration


def create_runtime_plot(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: ModeledResources,
    qec: SurfaceCodeScheme,
    magic_factory: MagicStateFactory,
    magic_count: int,
    rotation_model: rotation_synthesis.RotationCostModel,
    n_logical_gates: GateCost,
    rotation_layers: int | None = None,
) -> tuple[dict[str, str], Any]:
    """Creates the runtime figure and decides whether to display it or not.

    Currently displays the runtime plot for the Beverland model only.
    """
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return {"display": "none"}, go.Figure()
    factory = ParallelFactory(magic_factory, int(magic_count))
    c_min = beverland.minimum_time_steps(
        error_budget=error_budget,
        logical=algorithm,
        rotation_model=rotation_model,
        rotation_layers=rotation_layers,
    )
    err_model = LogicalErrorModel(qec_scheme=qec, physical_error=physical_error_rate)
    factory_cycles = factory.n_cycles(
        n_logical_gates=n_logical_gates, logical_error_model=err_model
    )
    min_num_factories = int(np.ceil(factory_cycles / c_min))
    magic_counts = list(
        1
        + np.random.choice(
            min_num_factories, replace=False, size=min(min_num_factories, 5)
        )
    )
    magic_counts.sort(reverse=True)
    magic_counts = np.array(magic_counts)
    time_steps = np.ceil(factory_cycles / magic_counts)
    magic_counts[0] = min_num_factories
    time_steps[0] = c_min
    cds = [
        beverland.code_distance(
            error_budget=error_budget,
            time_steps=t,
            logical=algorithm,
            qec_scheme=qec,
            physical_error=physical_error_rate,
        )
        for t in time_steps
    ]
    # TODO: plumb through PhysicalParameters
    cycle_time_us = PhysicalParameters.gidney_fowler().cycle_time_us
    duration = [cycle_time_us * t for t, d in zip(time_steps, cds, strict=True)]
    unit, formatted_duration = format_duration(duration)
    duration_name = f"Duration ({unit})"
    num_qubits = (
        SquareDataLayout.tile_count(int(algorithm.peak_logical_qubits))
        + factory.n_physical_qubits() * magic_counts
    )
    df = pd.DataFrame(
        {
            "label": [
                f"code distance: {c}<br>time steps: {t:g}<br>num factories: {f}"
                for t, c, f in zip(
                    time_steps, cds, magic_count * magic_counts, strict=True
                )
            ],
            duration_name: formatted_duration,
            "num qubits": num_qubits,
        }
    )
    fig = px.line(df, x=duration_name, y="num qubits", text="label")
    return {"display": "block"}, fig


def update(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm_data: Sequence[Any],
    qec_name: str,
    magic_name: str,
    magic_count: int,
    rotation_model_name: str,
) -> tuple[Any, ...]:
    """Updates the visualization."""
    if any(
        x is None
        for x in [physical_error_rate, error_budget, *algorithm_data, magic_count]
    ):
        raise PreventUpdate

    if len(algorithm_data) != 6:
        raise ValueError("Expected six algorithm counts in layout order")
    for value in (*algorithm_data, magic_count):
        if (
            type(value) not in (int, float)
            or not isfinite(value)
            or value < 0
            or int(value) != value
        ):
            raise ValueError(
                "Algorithm counts and factory count must be finite integers"
            )
    if magic_count < 1:
        raise ValueError("Factory count must be positive")
    qubits, measurements, ts, toffolis, rotations, n_rotation_layers = algorithm_data
    algorithm = ModeledResources(
        peak_logical_qubits=int(qubits),
        gates=GateCost(
            measurement=int(measurements),
            t=int(ts),
            toffoli=int(toffolis),
            rotation=int(rotations),
        ),
    )
    qec = _QEC_SCHEMES[qec_name]
    magic_factory = _MAGIC_FACTORIES[magic_name]
    rotation_model = _ROTATION_MODELS[rotation_model_name]
    n_logical_gates = beverland.discrete_logical_gates(
        synthesis_budget=error_budget / 3,
        logical=algorithm,
        rotation_model=rotation_model,
    )
    magic_count = int(magic_count)
    logical_err_model = LogicalErrorModel(
        qec_scheme=qec, physical_error=physical_error_rate
    )
    return (
        create_qubit_pie_chart(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            magic_factory,
            magic_count,
            n_logical_gates,
        ),
        *create_runtime_plot(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            qec,
            magic_factory,
            magic_count,
            rotation_model,
            n_logical_gates,
            rotation_layers=int(n_rotation_layers),
        ),
        *total_magic(estimation_model, n_logical_gates),
        *min_num_factories(
            logical_err_model,
            error_budget,
            estimation_model,
            algorithm,
            rotation_model,
            magic_factory,
            n_logical_gates,
            rotation_layers=int(n_rotation_layers),
        ),
        *compute_duration(
            physical_error_rate,
            error_budget,
            estimation_model,
            algorithm,
            rotation_model,
            magic_count,
            n_logical_gates,
        ),
    )


def total_magic(
    estimation_model: str, n_logical_gates: GateCost
) -> tuple[list[str], str]:
    """Compute the number of magic states needed for the algorithm and their type."""
    total_t = n_logical_gates.total_t_count()
    total_ccz = total_t / 4
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return ["Total Number of Toffoli gates"], f"{total_ccz:g}"
    return ["Total Number of T gates"], f"{total_t:g}"


def min_num_factories(
    logical_error_model: LogicalErrorModel,
    error_budget: float,
    estimation_model: str,
    algorithm: ModeledResources,
    rotation_model: rotation_synthesis.RotationCostModel,
    magic_factory: MagicStateFactory,
    n_logical_gates: GateCost,
    rotation_layers: int | None = None,
) -> tuple[dict[str, str], int]:
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        return {"display": "none"}, 1
    c_min = beverland.minimum_time_steps(
        error_budget=error_budget,
        logical=algorithm,
        rotation_model=rotation_model,
        rotation_layers=rotation_layers,
    )
    return {"display": "block"}, int(
        np.ceil(
            magic_factory.n_cycles(
                n_logical_gates=n_logical_gates, logical_error_model=logical_error_model
            )
            / c_min
        )
    )


def compute_duration(
    physical_error_rate: float,
    error_budget: float,
    estimation_model: str,
    algorithm: ModeledResources,
    rotation_model: rotation_synthesis.RotationCostModel,
    magic_count: int,
    n_logical_gates: GateCost,
) -> tuple[dict[str, str], str]:
    """Compute the duration of running the algorithm and whether to display the result or not.

    Currently displays the result only for GidneyFowler (arxiv:1812.01238).
    """
    if estimation_model == _GIDNEY_FOWLER_MODEL:
        res, _, _ = physical_search.search_gidney_fowler_resources(
            logical=ModeledResources(
                n_logical_gates, int(algorithm.peak_logical_qubits)
            ),
            physical_error=physical_error_rate,
            error_budget=error_budget,
            factories=[
                ParallelFactory(f, magic_count)
                for f in physical_search.iter_ccz_factories()
            ],
        )
        unit, duration = format_duration([res.duration_hours * 60 * 60 * 10**6])
        return {"display": "block"}, f"{duration[0]:g} {unit}"
    return {"display": "none"}, ""


app.callback(
    Output("qubit-pie-chart", "figure"),
    Output("runtime-container", "style"),
    Output("runtime-plot", "figure"),
    Output("magic-name", "children"),
    Output("total_magic", "value"),
    Output("magic-factory-container", "style"),
    Output("min_factory_count", "value"),
    Output("duration-container", "style"),
    Output("duration", "value"),
    *_ALL_INPUTS,
)(update)


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False)
