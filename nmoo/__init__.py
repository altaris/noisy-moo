"""
.. include:: ../README.md
.. include:: ../CHANGELOG.md
"""

from pkg_resources import DistributionNotFound, get_distribution

from .algorithms import ARNSGA2
from .benchmark import Benchmark
from .callbacks import TimerCallback
from .denoisers import KNNAvg, ResampleAverage
from .evaluators import PenalizedEvaluator
from .noises import GaussianNoise, UniformNoise
from .plotting import generate_delta_F_plots, plot_performance_indicators
from .wrapped_problem import WrappedProblem

try:
    __version__ = get_distribution("nmoo").version
except DistributionNotFound:
    __version__ = "local"
