"""A hardware lane must reject empty, skipped and fallback-only acceptance."""

from pathlib import Path

import pytest
import yaml

from tools.ci.run_ascend_910c import validate_report

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("tag", ("skipped", "failure", "error"))
def test_incomplete_hardware_cases_cannot_make_the_lane_green(
    tmp_path: Path, tag: str
) -> None:
    report = tmp_path / "native.xml"
    report.write_text(
        f'<testsuite><testcase name="native"><{tag}/></testcase></testsuite>'
    )
    with pytest.raises(RuntimeError, match="Incomplete Ascend regression"):
        validate_report(report, ())


def test_no_collected_hardware_cases_is_not_a_success(tmp_path: Path) -> None:
    report = tmp_path / "native.xml"
    report.write_text("<testsuites><testsuite/></testsuites>")
    with pytest.raises(RuntimeError, match="collected no test cases"):
        validate_report(report, ())


def test_a_case_outside_a_junit_suite_is_not_a_report(tmp_path: Path) -> None:
    report = tmp_path / "native.xml"
    report.write_text('<testcase name="unaccounted"/>')
    with pytest.raises(RuntimeError, match="did not produce a JUnit suite"):
        validate_report(report, ())


def test_native_cases_require_clean_logs_and_preserve_all_suites(
    tmp_path: Path,
) -> None:
    report, log = tmp_path / "native.xml", tmp_path / "native.log"
    report.write_text(
        '<testsuites><testsuite><testcase name="circuit"/></testsuite>'
        '<testsuite><testcase name="dynamics"/></testsuite></testsuites>'
    )
    log.write_text("native gradient and replay passed\n")
    assert validate_report(report, (log,)) == 2
    log.write_text("[flagos cpu_fallback] unexpected storage copy\n")
    with pytest.raises(RuntimeError, match="CPU fallback observed"):
        validate_report(report, (log,))


def test_ascend_dispatch_preserves_cuda_default_and_requires_its_own_runner() -> None:
    document = yaml.safe_load(
        (ROOT / ".github/workflows/flagos-reference.yml").read_text()
    )
    events = document.get("on", document.get(True))
    assert set(events) == {"workflow_dispatch"}
    choice = events["workflow_dispatch"]["inputs"]["platform"]
    assert choice["default"] == "cuda"
    assert set(choice["options"]) == {"cuda", "ascend-910c"}
    cuda = document["jobs"]["statevector-local-p0"]
    assert cuda["if"] == "${{ inputs.platform != 'ascend-910c' }}"
    native = document["jobs"]["ascend-910c-native"]
    assert native["if"] == "${{ inputs.platform == 'ascend-910c' }}"
    assert "ascend-910c" in native["runs-on"]
    assert native["concurrency"]["group"] == "flagquantum-ascend-910c"
    assert any(
        "tools/ci/run_ascend_910c.py" in step.get("run", "") for step in native["steps"]
    )
