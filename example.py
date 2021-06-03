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

# Setup problem pipeline
problem = nmoo.utils.ProblemWrapper(get_problem("zdt1"))  # For history, see later
noisy_problem = nmoo.noises.GaussianNoise(
    problem,
    {
        "F": (0.0, 0.01),
    },
)
# denoised_problem = nmoo.denoisers.KNNAvg(
#     noisy_problem,
#     distance_weight_type="squared",
#     max_distance=1.0,
#     n_neighbors=10,
# )
denoised_problem = nmoo.denoisers.Average(
    noisy_problem,
    n_evaluations=10,
)

# Minimize
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", prob=1.0, eta=20),
    eliminate_duplicates=True,
)
results = minimize(
    problem=denoised_problem,
    algorithm=algorithm,
    termination=get_termination("n_gen", 40),
    seed=1,
    save_history=True,
    verbose=True,
)

# Dump history of all parts of the pipeline
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)
problem.dump_history(OUT_PATH / "1_original.npz")
noisy_problem.dump_history(OUT_PATH / "2_noisy.npz")
denoised_problem.dump_history(OUT_PATH / "3_denoised.npz")
