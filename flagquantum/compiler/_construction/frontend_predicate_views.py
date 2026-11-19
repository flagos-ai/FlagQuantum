"""Compatibility alias for :mod:`.frontend.predicate_views`."""

import sys

from .frontend import predicate_views as _implementation
from .frontend.predicate_views import has_predicate_views as has_predicate_views
from .frontend.predicate_views import predicate_views as predicate_views

sys.modules[__name__] = _implementation
