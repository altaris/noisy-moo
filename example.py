from pathlib import Path
import os

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_problem,
    get_sampling,
    get_termination,
)
from pymoo.optimize import minimize

import nmoo

OUT_PATH = Path("./out")

# Problems
zdt1 = get_problem("zdt1", n_var=5)
zdt1_pareto_front = zdt1.pareto_front(100)

# Pipelines
wrapped_zdt1 = nmoo.utils.ProblemWrapper(zdt1)
noisy_zdt1 = nmoo.noises.GaussianNoise(
    wrapped_zdt1,
    {"F": (0.0, 0.25)},
)
avg_zdt1 = nmoo.denoisers.Average(
    noisy_zdt1,
    n_evaluations=5,
)
knnavg_zdt1 = nmoo.denoisers.KNNAvg(
    noisy_zdt1,
    distance_weight_type="squared",
    max_distance=1.0,
    n_neighbors=10,
)

# Algorithm
nsga2 = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", prob=1.0, eta=20),
    eliminate_duplicates=True,
)

# Benchmark
benchmark = nmoo.benchmark.Benchmark(
    problems={
        "knnavg_10": {
            "problem": knnavg_zdt1,
            "pareto_front": zdt1_pareto_front,
        },
        "avg_5": {
            "problem": avg_zdt1,
            "pareto_front": zdt1_pareto_front,
        },
    },
    algorithms={
        "nsga2": {
            "algorithm": nsga2,
            "seed": 42,
            "termination": get_termination("n_gen", 40),
        },
    },
    n_runs=2,
)
benchmark.run()

# Dump history of all parts of the pipeline
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)
benchmark._results.to_csv(OUT_PATH / "benchmark.csv")
