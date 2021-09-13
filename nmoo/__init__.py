"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""
__docformat__ = "google"

from pkg_resources import get_distribution, DistributionNotFound

from .wrapped_problem import WrappedProblem
from . import benchmark
from . import denoisers
from . import noises
from . import utils

try:
    __version__ = get_distribution('nmoo').version
except DistributionNotFound:
    __version__ = "local"
