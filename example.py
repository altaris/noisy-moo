from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_problem,
    get_sampling,
    get_termination,
)
from pymoo.optimize import minimize
import numpy as np



import nmoo

problem = get_problem("zdt1")
noisy_problem = nmoo.GaussianNoise(
    problem,
    {
        "F": (0.0, 0),
        # "G": (0.0, 0),
    },
)
denoised_problem = nmoo.KNNAvg(noisy_problem)

# print(problem.evaluate(np.ones((1, 30))))
# print(noisy_problem.evaluate(np.ones((1, 30))))

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
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

print(noisy_problem._history)
