"""Regenerate the immutable external Python session observations."""

import hashlib
import json
from pathlib import Path

import kickmix as km

from tests.unit.fixtures.physical_basis_session_cases import observe_sessions


def main() -> None:
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    extension = Path(km.__file__)
    assert (
        hashlib.sha256(extension.read_bytes()).hexdigest()
        == "5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e"
    )
    data = observe_sessions(km.Simulator, km.Circuit, km.q, km.b, km.r)
    target = Path(__file__).resolve().parents[1] / "data/physical_basis_session.json"
    target.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
