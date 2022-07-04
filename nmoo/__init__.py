"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from pkg_resources import get_distribution, DistributionNotFound

from .wrapped_problem import WrappedProblem
from .benchmark import Benchmark
from .callbacks import TimerCallback
from .denoisers import KNNAvg, ResampleAverage
from .evaluators import PenalizedEvaluator
from .noises import GaussianNoise, UniformNoise
from .plotting import generate_delta_F_plots, plot_performance_indicators

try:
    __version__ = get_distribution("nmoo").version
except DistributionNotFound:
    __version__ = "local"
