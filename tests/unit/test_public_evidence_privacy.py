import pytest

from tools.sanitize_public_evidence import sanitize_text, unsanitized_files

pytestmark = pytest.mark.unit


def test_public_evidence_is_sanitized() -> None:
    assert unsanitized_files() == ()


def test_sanitizer_redacts_infrastructure_identity() -> None:
    payload = (
        '{"hostname": "worker-01", '
        '"address": "10.20.30.40:1234", '
        '"command": "/home/alice/src/FlagQuantum/FlagQuantum/tools/run.py"}'
    )

    sanitized = sanitize_text(payload)

    assert '"hostname": "redacted"' in sanitized
    assert "<redacted-private-address>:1234" in sanitized
    assert '"command": "tools/run.py"' in sanitized
    assert sanitize_text(sanitized) == sanitized
