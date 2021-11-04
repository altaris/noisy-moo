import sys
from itertools import product
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from memory_profiler import memory_usage
from pymoo.core.population import Population
from pymoo.util.optimum import filter_optimum

try:
    from nmoo.utils.population import pareto_frontier_mask
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    from nmoo.utils.population import pareto_frontier_mask

Ns = [100, 500, 1000, 5000, 10_000, 50_000, 100_000]
ds = [2, 3, 4, 5, 6, 7]
n_evals = 10

results = []

for N, d, i in product(Ns, ds, range(n_evals)):
    print(f"N = {N}, d = {d}, repetition = {i+1}/{n_evals}")
    points = np.random.RandomState().normal([0.0] * d, 0.3, (N, d))
    population = Population.create(points)
    _ = population.set(
        F=points,
        feasible=np.full((N, 1), True),
    )
    mem = memory_usage((pareto_frontier_mask, (points,)), max_usage=True)
    results.append((N, d, "nmoo", mem))
    mem = memory_usage((filter_optimum, (population,)), max_usage=True)
    results.append((N, d, "pymoo", mem))

df = pd.DataFrame(results, columns=("N", "d", "algo", "mem"))
df.to_csv("memory.csv")

plot = sns.FacetGrid(df, col_wrap=3, col="N", hue="algo", sharey=False)
plot.map(sns.lineplot, "d", "mem")
plot.add_legend()
plot.savefig("memory.jpg")
