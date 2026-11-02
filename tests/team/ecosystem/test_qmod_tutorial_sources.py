"""Added tutorial identities stay pinned independently of the old manifests."""

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
CORPUS = Path(__file__).resolve().parents[2] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    "name,blob,digest",
    [
        (
            "classiq_overview_tutorial.qmod",
            "1f476102f49ad56009d5329a8f180c1596be61dc",
            "5e28d56fea67b203a6e0e0b44ef0fa8998e66ae477fc62a4b02d2c5c8bcd30d1",
        ),
        (
            "execution_tutorial.qmod",
            "7e9df09937d8360be71688762c4d22f309faa443",
            "4dbff47c68a449c918d30cdb63d8398de7084f02617e604d82c24191bcbfea8b",
        ),
    ],
)
def test_original_tutorial_and_manifest_pin_same_upstream_bytes(
    name: str, blob: str, digest: str
) -> None:
    m = json.loads((CORPUS / "tutorial_sources.json").read_text())
    assert m["repository"] == "https://github.com/Classiq/classiq-library"
    assert m["commit"] == "2bfd835e73b6882fcc515a725098b04766c238e7"
    assert m["license_file"] == "LICENSE.txt"
    assert (
        m["license_sha256"]
        == "241f68c9f35357f97e8a2a195e447d4f80098c140f0b3b3c770a11eed96f58e0"
    )
    assert (
        hashlib.sha256((CORPUS / "LICENSE.txt").read_bytes()).hexdigest()
        == m["license_sha256"]
    )
    assert next(item for item in m["fixtures"] if item["file"] == name) == {
        "file": name,
        "upstream_path": "tutorials/basic_tutorials/the_classiq_tutorial/" + name,
        "git_blob_sha1": blob,
        "sha256": digest,
    }
    raw = (CORPUS / name).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == blob
    )
    assert hashlib.sha256(raw).hexdigest() == digest
