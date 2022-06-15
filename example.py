"""
Example of a benchmark factory. The function `make_benchmark` returns an
`nmoo.Benchmark` object (following the same specifications as in the example
notebook). In a terminal, execute

    nmoo run foobar:make_benchmark

to run it.
"""

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.problems.multi import ZDT1

import nmoo


def make_benchmark() -> nmoo.benchmark.Benchmark:
    """Benchmark factory"""

    zdt1 = ZDT1()
    wrapped_zdt1 = nmoo.WrappedProblem(zdt1)

    mean = np.array([0, 0])
    covariance = np.array([[1.0, -0.5], [-0.5, 1]])
    noisy_zdt1 = nmoo.noises.GaussianNoise(wrapped_zdt1, mean, covariance)

    avg_zdt1 = nmoo.denoisers.ResampleAverage(noisy_zdt1, n_evaluations=10)
    knnavg_zdt1 = nmoo.denoisers.KNNAvg(noisy_zdt1, max_distance=1.0)

    nsga2 = NSGA2()

    pareto_front = zdt1.pareto_front(100)

    return nmoo.benchmark.Benchmark(
        output_dir_path="./out",
        problems={
            "knnavg": {
                "problem": knnavg_zdt1,
                "pareto_front": pareto_front,
            },
            "avg": {
                "problem": avg_zdt1,
                "pareto_front": pareto_front,
                "evaluator": nmoo.evaluators.EvaluationPenaltyEvaluator(10),
            },
        },
        algorithms={
            "nsga2": {
                "algorithm": nsga2,
            },
            "nsga2_100": {
                "algorithm": nsga2,
                "termination": get_termination("n_gen", 100),
            },
        },
        n_runs=3,
    )
