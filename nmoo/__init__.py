"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from pkg_resources import get_distribution, DistributionNotFound

from .wrapped_problem import WrappedProblem
from . import benchmark
from . import callbacks
from . import denoisers
from . import evaluators
from . import noises

try:
    __version__ = get_distribution("nmoo").version
except DistributionNotFound:
    __version__ = "local"
