"""Compatibility alias for :mod:`.frontend.storage`."""

import sys

from .frontend import storage as _implementation
from .frontend.storage import capture_reset as capture_reset
from .frontend.storage import declare_hidden as declare_hidden
from .frontend.storage import drop_hidden as drop_hidden
from .frontend.storage import hidden_declarations as hidden_declarations
from .frontend.storage import require_root_storage as require_root_storage

sys.modules[__name__] = _implementation
