# version.py
"""Version information for the distributed quantum device package."""

__version__ = "0.1.0"

# 可选：版本元数据
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release_level": "alpha",  # "alpha", "beta", "rc", "final"
}


def get_version() -> str:
    """Return the current version as a string."""
    return __version__


def check_version_compatibility(min_version: str) -> bool:
    """Check if current version meets minimum requirement."""
    from packaging import version

    return version.parse(__version__) >= version.parse(min_version)
