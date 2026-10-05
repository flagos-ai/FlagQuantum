"""Remote control-plane support for QBoson Kaiwu services.

The module is intentionally not re-exported from :mod:`flagquantum.remote`
while its provider contract is under review.
"""

from ._credentials import KaiwuCredentials, resolve_kaiwu_credentials

__all__ = ("KaiwuCredentials", "resolve_kaiwu_credentials")
