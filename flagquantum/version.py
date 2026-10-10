# version.py
"""Version information for the distributed quantum device package."""

__version__ = "0.3.0rc2"

VERSION_INFO = {
    "major": 0,
    "minor": 3,
    "patch": 0,
    "release_level": "rc",  # "development", "alpha", "beta", "rc", "final"
}


def get_version() -> str:
    """Return the current version as a string."""
    return __version__
