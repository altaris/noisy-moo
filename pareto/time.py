from itertools import product
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
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

# Load intermediate results
# df = pd.read_csv("time.csv")
# results = list(map(tuple, df[["N", "d", "algo", "time"]].to_numpy()))

for N, d, i in product(Ns, ds, range(n_evals)):
    print(f"N = {N}, d = {d}, repetition = {i+1}/{n_evals}")
    points = np.random.RandomState().normal([0.0] * d, 0.3, (N, d))
    population = Population.create(points)
    _ = population.set(
        F=points,
        feasible=np.full((N, 1), True),
    )
    start = time()
    filter_optimum(population)
    results.append((N, d, "pymoo", time() - start))
    start = time()
    pareto_frontier_mask(points)
    results.append((N, d, "nmoo", time() - start))

df = pd.DataFrame(results, columns=("N", "d", "algo", "time"))
df.to_csv("time.csv")

plot = sns.FacetGrid(df, col_wrap=3, col="N", hue="algo", sharey=False)
plot.map(sns.lineplot, "d", "time")
plot.add_legend()
plot.savefig("time.jpg")
