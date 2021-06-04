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

import nmoo

# =============================================================================
# Output directory
# =============================================================================

OUT_PATH = Path("./out")

# =============================================================================
# Problems
# =============================================================================

zdt1 = get_problem("zdt1", n_var=5)
zdt1_pareto_front = zdt1.pareto_front(100)  # Optional

# =============================================================================
# Pipelines
# =============================================================================

# First, wrap the pymoo problem in an nmoo ProblemWrapper
wrapped_zdt1 = nmoo.utils.ProblemWrapper(zdt1)

# Add noise
noisy_zdt1 = nmoo.noises.GaussianNoise(
    wrapped_zdt1,
    {"F": (0.0, 0.25)},
)

# Try the Average denoiser (which naively evaluates the noisy problem 5 times
# and averages the results)
avg_zdt1 = nmoo.denoisers.Average(
    noisy_zdt1,
    n_evaluations=5,
)

# Try the KNNAvg denoiser
knnavg_zdt1 = nmoo.denoisers.KNNAvg(
    noisy_zdt1,
    distance_weight_type="squared",
    max_distance=1.0,
    n_neighbors=10,
)

# =============================================================================
# Algorithms
# =============================================================================

nsga2 = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", prob=1.0, eta=20),
    eliminate_duplicates=True,
)

# =============================================================================
# Benchmark
# =============================================================================

benchmark = nmoo.benchmark.Benchmark(
    # List all problems
    problems={
        "knnavg_10": {
            "problem": knnavg_zdt1,
            "pareto_front": zdt1_pareto_front,  # Optional
        },
        "avg_5": {
            "problem": avg_zdt1,
            "pareto_front": zdt1_pareto_front,  # Optional
        },
    },
    # List all algorithms to run
    algorithms={
        "nsga2": {
            "algorithm": nsga2,
            "seed": 42,  # Optional
            "termination": get_termination("n_gen", 40),  # Optional
        },
    },
    # Number of times to run each problem/algorithm pair
    n_runs=2,
)

benchmark.run()

# Dump all results and histories for later analysis
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)
benchmark.dump_everything(OUT_PATH)

# The following file should appear in ./out:
# * avg_5.1.npz: avg_zdt1 call history;
# * avg_5.2.npz: noisy_zdt1 call history;
# * avg_5.3.npz: wrapped_zdt1 call history;
# * benchmark.csv: Benchmark results;
# * knnavg_10.1.npz: knnavg_zdt1 call history;
# * knnavg_10.2.npz: noisy_zdt1 call history;
# * knnavg_10.3.npz: wrapped_zdt1 call history.

# Note that since wrapped_zdt1 and noisy_zdt1 were reused in both pipelines,
# the contents of avg_5.3.npz and knnavg_10.3.npz will be identical, and
# likewise for avg_5.2.npz and knnavg_10.2.npz.