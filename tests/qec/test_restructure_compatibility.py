"""Pre-migration QEC contracts captured at ff69b2526; never refresh to hide drift."""

from __future__ import annotations

import base64
import dataclasses
import importlib
import inspect
import json
import pickle
from enum import Enum
from pathlib import Path

import pytest
from _retirement import (
    RETIRED,
    resolve,
)

import flagquantum.qec as qec

pytestmark = pytest.mark.unit
_FIXTURE = Path(__file__).with_name("fixtures") / "restructure_001_api.json"


def _describe(value: object) -> dict[str, object]:
    description: dict[str, object] = {}
    if inspect.isclass(value) and issubclass(value, BaseException):
        description["exception_bases"] = [base.__name__ for base in value.__mro__[1:]]
    elif inspect.isclass(value) and issubclass(value, Enum):
        # Enum construction belongs to Python and differs across supported versions.
        pass
    elif callable(value):
        description["signature"] = str(inspect.signature(value))
    else:
        description["value"] = value
    if inspect.isclass(value):
        if dataclasses.is_dataclass(value):
            description["fields"] = [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "init": field.init,
                    "default": (
                        repr(field.default)
                        if field.default is not dataclasses.MISSING
                        else None
                    ),
                    "factory": getattr(field.default_factory, "__qualname__", None),
                }
                for field in dataclasses.fields(value)
            ]
            description["frozen"] = value.__dataclass_params__.frozen
        if issubclass(value, Enum):
            description["members"] = {member.name: member.value for member in value}
        description["methods"] = {
            name: str(inspect.signature(member))
            for name, member in inspect.getmembers(value)
            if not name.startswith("_")
            and (inspect.isfunction(member) or inspect.ismethod(member))
        }
        description["properties"] = sorted(
            name
            for name, member in inspect.getmembers(value)
            if isinstance(member, property)
        )
    return description


def test_top_level_exports_and_signatures_match_pre_migration_baseline() -> None:
    baseline = json.loads(_FIXTURE.read_text())
    # Proposal 087 adds one reviewed method without rewriting historical evidence.
    contract = json.loads(
        (
            Path(__file__).parents[2] / "contracts/qec-batch-prediction-v1.json"
        ).read_text()
    )
    assert list(qec.__all__) == baseline["exports"]
    for name, expected in baseline["api"].items():
        if name in contract["additive_methods"]:
            expected = {
                **expected,
                "methods": {
                    **expected["methods"],
                    **contract["additive_methods"][name],
                },
            }
        assert _describe(getattr(qec, name)) == expected, name


def test_imports_follow_the_approved_retirement_boundary() -> None:
    baseline = json.loads(_FIXTURE.read_text())
    for name, path in baseline["origins"].items():
        assert _resolve_current(path) is getattr(qec, name)
    for module in RETIRED:
        with pytest.raises(ModuleNotFoundError) as caught:
            importlib.import_module(module)
        assert caught.value.name == module
    from flagquantum.qec.codes import CssCode

    assert CssCode is qec.CssCode


def _resolve_current(path: str) -> object:
    return resolve(path)


def test_original_pickle_payloads_keep_surviving_identifiers_and_refuse_retired_paths() -> (
    None
):
    baseline = json.loads(_FIXTURE.read_text())
    for name, payload in baseline["pickles"].items():
        module_name, attribute = name.rsplit(".", 1)
        data = base64.b64decode(payload)
        if module_name in RETIRED:
            with pytest.raises(ModuleNotFoundError) as caught:
                pickle.loads(data)
            assert caught.value.name == module_name
        else:
            restored = pickle.loads(data)
            assert type(restored) is getattr(
                importlib.import_module(module_name), attribute
            )
            assert pickle.dumps(restored, protocol=4) == data


def test_registry_names_and_factory_identity_are_preserved() -> None:
    baseline = json.loads(_FIXTURE.read_text())
    assert list(qec.code_names()) == baseline["code_names"]
    assert list(qec.decoder_names()) == baseline["decoder_names"]
    assert type(qec.get_code("repetition", distance=3)) is qec.RepetitionCode
    model = qec.DetectorErrorModel.from_stim_text(
        "error(0.125) D0 L0\ndetector D0\nlogical_observable L0\n"
    )
    decoder = qec.get_decoder(qec.AUTHORITY_NAME, model)
    assert type(decoder) is qec.MinimumWeightMatchingDecoder
    assert decoder.decode((0,)).observables == (0,)
    assert decoder.decode(()).observables == ()


def test_dem_text_and_exact_boundary_probabilities_are_preserved() -> None:
    text = "error(1.0) D0 L0\nerror(0.0) D1\ndetector D0\ndetector D1\nlogical_observable L0\n"
    model = qec.DetectorErrorModel.from_stim_text(text)
    assert model.to_stim_text() == text
    sample = model.dem_sampling(shots=4, seed=7)
    assert sample.detectors.tolist() == [[1, 0]] * 4
    assert sample.observables.tolist() == [[1]] * 4


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode", ("compiled_lookup", "offline_pauli_frame", "runtime_decoder")
)
def test_existing_user_feedback_workflow(mode: str) -> None:
    result = qec.run_repetition_memory_experiment(
        error_schedule=qec.ErrorSchedule((qec.ErrorEvent(1, 1),)),
        rounds=3,
        shots=2,
        seed=53,
        strategy="trajectory",
        feedback_mode=mode,
    )
    assert result.logical_error_rate == 0.0
    assert (
        tuple(shot.decoded_data_bits for shot in result.shot_records)
        == ((0, 0, 0),) * 2
    )
    expected_raw = (0, 1, 0) if mode == "offline_pauli_frame" else (0, 0, 0)
    assert all(shot.raw_final_data_bits == expected_raw for shot in result.shot_records)


def test_plugins_share_registries_across_old_and_top_level_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.qec import codes
    from flagquantum.qec.decoders import registry
    from flagquantum.qec.decoders import registry as registry_owner

    monkeypatch.setattr(codes, "_CODES", dict(codes._CODES))
    monkeypatch.setattr(registry_owner, "_DECODERS", dict(registry_owner._DECODERS))
    assert qec.register_code is codes.register_code
    assert qec.register_decoder is registry.register_decoder
    assert (
        qec.register_code("compat_test_code")(qec.RepetitionCode) is qec.RepetitionCode
    )
    assert type(codes.get_code("compat_test_code", distance=3)) is qec.RepetitionCode
    with pytest.raises(ValueError):
        codes.register_code("compat_test_code")(qec.SteaneCode)
    codes.register_code("compat_test_code", replace=True)(qec.SteaneCode)
    assert type(qec.get_code("compat_test_code")) is qec.SteaneCode
    qec.register_decoder("compat_test_decoder")(qec.MinimumWeightMatchingDecoder)
    model = qec.DetectorErrorModel(1, 1, (qec.DemError(0.125, (0,), (0,)),))
    assert registry.get_decoder("compat_test_decoder", model).decode(
        (0,)
    ).observables == (0,)
    # Asking for names must not overwrite a plugin's explicit replacement.
    registry.register_decoder(qec.SLIDING_WINDOW_NAME, replace=True)(
        qec.MinimumWeightMatchingDecoder
    )
    assert qec.SLIDING_WINDOW_NAME in qec.decoder_names()
    assert (
        type(qec.get_decoder(qec.SLIDING_WINDOW_NAME, model))
        is qec.MinimumWeightMatchingDecoder
    )


def test_surviving_module_public_apis_match_pre_migration_baseline() -> None:
    baseline = json.loads(_FIXTURE.read_text())
    for module_name, expected in baseline["module_api"].items():
        if module_name in RETIRED:
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(module_name)
            continue
        module = importlib.import_module(module_name)
        assert list(getattr(module, "__all__", ())) == expected["exports"]
        for name, root_name in expected["root_aliases"].items():
            assert getattr(module, name) is getattr(qec, root_name)
        for name, description in expected["symbols"].items():
            assert _describe(getattr(module, name)) == description, (
                f"{module_name}.{name}"
            )


def test_seeded_dem_numerical_baseline() -> None:
    baseline = json.loads(_FIXTURE.read_text())["seeded_dem"]
    model = qec.DetectorErrorModel(
        2, 1, (qec.DemError(0.125, (0,), (0,)), qec.DemError(0.25, (0, 1), ()))
    )
    sample = model.dem_sampling(shots=16, seed=7)
    assert sample.detectors.tolist() == baseline["detectors"]
    assert sample.observables.tolist() == baseline["observables"]
    assert model.detector_rates().tolist() == [0.3125, 0.25]
    assert model.observable_rates().tolist() == [0.125]


def test_surviving_objects_from_old_compound_pickles_keep_their_state() -> None:
    from _retirement import assert_surviving_objects

    assert_surviving_objects()
