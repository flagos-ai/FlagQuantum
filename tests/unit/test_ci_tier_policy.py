from pathlib import Path

import pytest

from tools.ci_tier import CI_TIERS, main

pytestmark = pytest.mark.unit


def test_ci_tiers_define_commands_and_triggers():
    expected = {
        "pr-default",
        "pr-runtime",
        "pr-distributed",
        "nightly",
        "gpu-scheduled",
        "multinode-scheduled",
        "release",
    }
    assert set(CI_TIERS) == expected
    for tier in CI_TIERS.values():
        assert tier.trigger
        assert tier.proves
        assert tier.does_not_prove
        assert tier.commands
        assert all(command for command in tier.commands)


def test_default_pr_stays_fast_and_local():
    lines = CI_TIERS["pr-default"].command_lines()
    assert lines == ("python -m pytest -m smoke or unit -q",)
    joined = "\n".join(lines)
    assert "distributed_cpu" not in joined
    assert "distributed_accel" not in joined
    assert "distributed_multinode" not in joined
    assert "benchmark" not in joined
    assert "scalability" not in joined


def test_runtime_pr_tier_uses_seeded_integration_marker():
    lines = CI_TIERS["pr-runtime"].command_lines()
    assert lines == ("python -m pytest -m integration -q",)
    assert "integration marker" in CI_TIERS["pr-runtime"].proves


def test_distributed_pr_tier_is_contract_not_scalability_evidence():
    tier = CI_TIERS["pr-distributed"]
    lines = tier.command_lines()
    assert "python -m pytest -m distributed_cpu -q" in lines
    assert "python -m pytest -m benchmark_contract or release_gate -q" in lines
    assert "not release evidence" in tier.does_not_prove
    assert "scalability" in tier.does_not_prove


def test_scheduled_hardware_tiers_do_not_block_default_pr():
    assert CI_TIERS["pr-default"].blocks_default_pr is True
    assert CI_TIERS["gpu-scheduled"].blocks_default_pr is False
    assert CI_TIERS["multinode-scheduled"].blocks_default_pr is False
    assert CI_TIERS["gpu-scheduled"].command_lines() == (
        "python -m pytest -m distributed_accel and gpu -q",
    )
    assert CI_TIERS["multinode-scheduled"].command_lines() == (
        "python -m pytest -m distributed_multinode -q",
    )


def test_release_tier_runs_fail_closed_benchmark_audits():
    lines = CI_TIERS["release"].command_lines()
    assert "python -m pytest -m not scalability -q" in lines
    assert "python benchmarks/audit_results.py --input benchmarks/results" in lines
    assert (
        "python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability"
        in lines
    )


def test_ci_tier_commands_preserve_arg_boundaries_and_python_substitution():
    for tier in CI_TIERS.values():
        for command in tier.commands:
            assert command[0] == "{python}", tier.name
            assert all(token for token in command), tier.name
            assert not any(
                token in {"&&", "||", "|", ";"} for token in command
            ), tier.name
            benchmark_scripts = [
                token
                for token in command
                if token.startswith("benchmarks/") and token.endswith(".py")
            ]
            if benchmark_scripts:
                assert benchmark_scripts == ["benchmarks/audit_results.py"], tier.name

        rendered = tier.rendered_commands("PYTHON_EXE")
        assert all(command[0] == "PYTHON_EXE" for command in rendered), tier.name


def test_nightly_tier_excludes_hardware_multinode_scalability_and_benchmark_contracts():
    lines = CI_TIERS["nightly"].command_lines()
    assert lines == (
        "python -m pytest -m not benchmark_contract and not scalability and not distributed_multinode and not distributed_accel and not gpu -q",
    )
    line = lines[0]
    assert "not benchmark_contract" in line
    assert "not scalability" in line
    assert "not distributed_multinode" in line
    assert "not distributed_accel" in line
    assert "not gpu" in line


def test_docs_ci_matrix_matches_script_tiers_and_fail_closed_boundaries():
    text = Path("docs/TESTING.md").read_text(encoding="utf-8")

    for tier_name in CI_TIERS:
        assert f"python tools/ci_tier.py {tier_name}" in text
    assert "explicit accelerator runner" in text
    assert "explicit rank placement" in text
    assert "not scalability evidence" in text
    assert "fail-closed" in text
    assert "Release" in text


def test_ci_tier_cli_lists_and_dry_runs_without_executing(capsys):
    assert main(["--list"]) == 0
    list_output = capsys.readouterr().out
    for tier_name in CI_TIERS:
        assert tier_name in list_output

    assert main(["pr-distributed", "--dry-run", "--python", "PYTHON_EXE"]) == 0
    dry_run_output = capsys.readouterr().out
    assert "+ PYTHON_EXE -m pytest -m distributed_cpu -q" in dry_run_output
    assert (
        "+ PYTHON_EXE -m pytest -m benchmark_contract or release_gate -q"
        in dry_run_output
    )
