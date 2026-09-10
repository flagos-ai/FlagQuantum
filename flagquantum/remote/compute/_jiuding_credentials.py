"""Credential discovery for Jiuding control-plane requests."""

import base64
import os
from pathlib import Path


def load_jiuding_credentials() -> tuple[str, str]:
    """Load one complete credential pair without mixing sources."""
    ak = os.environ.get("JIUDING_AK")
    sk = os.environ.get("JIUDING_SK")
    if ak is not None or sk is not None:
        if ak is None or sk is None:
            raise RuntimeError(
                "Set both JIUDING_AK and JIUDING_SK; partial environment "
                "credentials are not allowed"
            )
    else:
        try:
            ak = _decode(Path("/etc/accesskey/user-ak"))
            sk = _decode(Path("/etc/accesskey/user-sk"))
        except (OSError, ValueError):
            raise RuntimeError(
                "Set both JIUDING_AK and JIUDING_SK, or use a Jiuding "
                "workspace with both injected credential files"
            ) from None
    ak, sk = ak.strip(), sk.strip()
    if not ak or not sk:
        raise RuntimeError("Jiuding credentials must not be empty")
    return ak, sk


def _decode(path: Path) -> str:
    encoded = b"".join(path.read_bytes().split())
    return base64.b64decode(encoded, validate=True).decode()
