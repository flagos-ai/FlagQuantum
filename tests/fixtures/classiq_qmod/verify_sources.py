"""Verify retained source fixtures without importing FlagQuantum or Classiq."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["fixtures"]:
        payload = (root / item["file"]).read_bytes()
        blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        if hashlib.sha1(blob).hexdigest() != item["git_blob_sha1"]:
            raise ValueError(f"Git blob mismatch: {item['file']}")
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {item['file']}")
        print(f"Verified {item['file']}")
    license_bytes = (root / manifest["license_file"]).read_bytes()
    if hashlib.sha256(license_bytes).hexdigest() != manifest["license_sha256"]:
        raise ValueError("License checksum mismatch")
    print("Verified original MIT license; 7 digest checks passed")


if __name__ == "__main__":
    main()
