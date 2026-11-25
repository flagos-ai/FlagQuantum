"""Compare the complete native resource panel with frozen callbacks and plots."""

import json
from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from itertools import product
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

np = pytest.importorskip("numpy")
pytest.importorskip("dash")
pytest.importorskip("plotly")
source = pytest.importorskip("qualtran.surface_code.ui")
native = import_module("examples.apps.physical_resource_dashboard")


@pytest.mark.parametrize("model", source._SUPPORTED_ESTIMATION_MODELS)
def test_complete_callback_preserves_random_state_for_the_next_request(
    model: str,
) -> None:
    states = []
    for module, rotation in (
        (source, "SevenDigitsOfPrecisionConstantCost"),
        (native, "SEVEN_BIT_PHASE_GRADIENT_COST"),
    ):
        np.random.seed(17)
        module.update(
            1e-4,
            1e-3,
            model,
            (10**11,) * 6,
            "GidneyFowler",
            "FifteenToOne733",
            1,
            rotation,
        )
        states.append(np.random.get_state())
    assert states[0][0] == states[1][0]
    np.testing.assert_array_equal(states[0][1], states[1][1])
    assert states[0][2:] == states[1][2:]


@pytest.mark.parametrize("model", source._SUPPORTED_ESTIMATION_MODELS)
@pytest.mark.parametrize(
    "rotation", ("BeverlandEtAlRotationCost", "SevenDigitsOfPrecisionConstantCost")
)
@pytest.mark.parametrize("factories", (1, 3))
def test_complete_callback_choices_and_all_figure_data(
    model: str, rotation: str, factories: int
) -> None:
    names = {
        "BeverlandEtAlRotationCost": "BEVERLAND_ROTATION_COST",
        "SevenDigitsOfPrecisionConstantCost": "SEVEN_BIT_PHASE_GRADIENT_COST",
    }
    assert version("dash") == "2.18.2" and version("plotly") == "5.24.1"
    for qec, factory in product(source._QEC_SCHEMES, source._MAGIC_FACTORIES):
        np.random.seed(17)
        expected = source.update(
            1e-4, 1e-3, model, (10**11,) * 6, qec, factory, factories, rotation
        )
        np.random.seed(17)
        actual = native.update(
            1e-4, 1e-3, model, (10**11,) * 6, qec, factory, factories, names[rotation]
        )
        assert actual[1] == expected[1]
        assert actual[3:] == expected[3:]
        for i in (0, 2):
            assert json.loads(actual[i].to_json()) == json.loads(expected[i].to_json())


@pytest.mark.parametrize(
    "duration",
    (
        0.0,
        1000.0,
        1001.0,
        1e6,
        1e6 + 1,
        60e6,
        60e6 + 1,
        3600e6,
        3600e6 + 1,
        86400e6,
        86400e6 + 1,
    ),
)
def test_exact_duration_thresholds_and_vector_units(duration: float) -> None:
    assert native.format_duration([duration, duration * 2]) == source.format_duration(
        [duration, duration * 2]
    )


def test_source_missing_input_and_empty_duration_failures() -> None:
    for module, rotation in (
        (source, "BeverlandEtAlRotationCost"),
        (native, "BEVERLAND_ROTATION_COST"),
    ):
        with pytest.raises(module.PreventUpdate):
            module.update(
                None,
                1e-3,
                module._GIDNEY_FOWLER_MODEL,
                (10**11,) * 6,
                "GidneyFowler",
                "FifteenToOne733",
                1,
                rotation,
            )
        with pytest.raises(IndexError):
            module.format_duration([])


def test_registered_http_callback_and_all_nine_outputs() -> None:
    responses = []
    for module, rotation in (
        (source, "BeverlandEtAlRotationCost"),
        (native, "BEVERLAND_ROTATION_COST"),
    ):
        client = module.app.server.test_client()
        assert client.get("/").status_code == 200
        assert client.get("/_dash-layout").status_code == 200
        assert client.get("/_dash-dependencies").status_code == 200
        assert len(module.app.callback_map) == 1
        key, callback = next(iter(module.app.callback_map.items()))
        values = {
            "physical_error_rate": 1e-4,
            "error_budget": 1e-3,
            "estimation_model": module._GIDNEY_FOWLER_MODEL,
            "QEC": "GidneyFowler",
            "_MAGIC_FACTORIES": "FifteenToOne733",
            "_MAGIC_FACTORIES_number": 1,
            "rotation-cost-model": rotation,
        }
        inputs = [
            dict(
                item,
                value=(
                    [10**11] * 6 if "algorithm" in item["id"] else values[item["id"]]
                ),
            )
            for item in callback["inputs"]
        ]
        outputs = [
            {"id": item.component_id, "property": item.component_property}
            for item in callback["output"]
        ]
        assert len(outputs) == 9
        response = client.post(
            "/_dash-update-component",
            json={
                "output": key,
                "outputs": outputs,
                "inputs": inputs,
                "state": [],
                "changedPropIds": ["physical_error_rate.value"],
            },
        )
        assert response.status_code == 200
        responses.append(response.get_json())
    assert responses[0] == responses[1]
    assert "FlagQuantum" in native.app.title


def test_frozen_complete_dashboard_sources() -> None:
    assert source.__file__ is not None
    for name, blob in (
        ("ui.py", "85f4010c42616909458bdb8490b01a9990595469"),
        ("ui_test.py", "4fa2944a0cb1c4a74df1ff2a27a4796dcd314787"),
    ):
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("rotation_layers", (0, 3, 11))
def test_independent_rotation_depth_reaches_runtime_and_factory_consumers(
    rotation_layers: int,
) -> None:
    data = (17, 19, 23, 29, 11, rotation_layers)
    np.random.seed(17)
    expected = source.update(
        1e-4,
        1e-3,
        source._BEVERLAND_MODEL,
        data,
        "GidneyFowler",
        "GidneyFowlerCCZ",
        3,
        "BeverlandEtAlRotationCost",
    )
    np.random.seed(17)
    actual = native.update(
        1e-4,
        1e-3,
        native._BEVERLAND_MODEL,
        data,
        "GidneyFowler",
        "GidneyFowlerCCZ",
        3,
        "BEVERLAND_ROTATION_COST",
    )
    assert actual[1] == expected[1]
    assert actual[3:] == expected[3:]
    for i in (0, 2):
        assert json.loads(actual[i].to_json()) == json.loads(expected[i].to_json())


def test_layout_algorithm_fields_and_output_builders() -> None:
    # The layout is consumed through Dash serialization, including pattern IDs.
    layouts = []
    for module in (source, native):
        response = module.app.server.test_client().get("/_dash-layout")
        layout = response.get_json()
        assert isinstance(layout, dict)
        layouts.append(layout)

    def collect_ids(value: object) -> list[object]:
        if isinstance(value, list):
            return [item for child in value for item in collect_ids(child)]
        if isinstance(value, dict):
            identifiers = [value["id"]] if "id" in value else []
            return identifiers + [
                item for child in value.values() for item in collect_ids(child)
            ]
        return []

    assert collect_ids(layouts[0]) == collect_ids(layouts[1])
    for original, adapted in (
        (source.algorithm_summary_components, native.algorithm_summary_components),
        (source.create_ouputs, native.output_components),
    ):
        expected = [item.to_plotly_json() for item in original()]
        actual = [item.to_plotly_json() for item in adapted()]
        encode = import_module("plotly.utils").PlotlyJSONEncoder
        assert json.dumps(actual, cls=encode) == json.dumps(expected, cls=encode)


@pytest.mark.parametrize("value", (1.5, -1, float("inf"), float("nan"), True, "2"))
def test_rejects_invalid_original_counts_before_integer_coercion(value: object) -> None:
    for position in range(7):
        arguments = [17, 19, 23, 29, 11, 3, 1]
        data = [value if i == position else count for i, count in enumerate(arguments)]
        with pytest.raises(ValueError, match="finite integers"):
            native.update(
                1e-4,
                1e-3,
                native._BEVERLAND_MODEL,
                data[:6],
                "GidneyFowler",
                "GidneyFowlerCCZ",
                data[6],
                "BEVERLAND_ROTATION_COST",
            )


def test_integral_json_numbers_match_integer_counts() -> None:
    results = []
    for data in ((17, 19, 23, 29, 11, 3), (17.0, 19.0, 23.0, 29.0, 11.0, 3.0)):
        np.random.seed(17)
        result = native.update(
            1e-4,
            1e-3,
            native._BEVERLAND_MODEL,
            data,
            "GidneyFowler",
            "GidneyFowlerCCZ",
            3.0,
            "BEVERLAND_ROTATION_COST",
        )
        results.append(
            (
                json.loads(result[0].to_json()),
                result[1],
                json.loads(result[2].to_json()),
                result[3:],
            )
        )
    assert results[0] == results[1]


def test_rejects_malformed_algorithm_shape_and_zero_factories() -> None:
    for data, factories, message in (
        ((1, 2, 3), 1, "six algorithm"),
        ((17, 19, 23, 29, 11, 3), 0, "positive"),
    ):
        with pytest.raises(ValueError, match=message):
            native.update(
                1e-4,
                1e-3,
                native._BEVERLAND_MODEL,
                data,
                "GidneyFowler",
                "GidneyFowlerCCZ",
                factories,
                "BEVERLAND_ROTATION_COST",
            )
