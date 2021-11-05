from itertools import product
from time import time
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from memory_profiler import memory_usage
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

try:
    from nmoo.utils.population import pareto_frontier_mask
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    from nmoo.utils.population import pareto_frontier_mask

Ns = [100, 500, 1000, 5000, 10_000, 50_000, 100_000]
ds = [2, 3, 4, 5, 6, 7]
n_evals = 10
pymoo_methods = [
    "efficient_non_dominated_sort",
    "fast_non_dominated_sort",
    # "tree_based_non_dominated_sort",
]


def _time(algorithm: Callable, data: np.ndarray) -> Tuple[float, float]:
    """
    Runs `algorithm` on `data`, returns time.
    """
    start = time()
    algorithm(data)
    return time() - start


if __name__ == "__main__":
    results = []
    for N, d, i in product(Ns, ds, range(n_evals)):
        print(f"N = {N}, d = {d}, repetition = {i+1}/{n_evals}")
        points = np.random.RandomState().normal([0.0] * d, 0.3, (N, d))
        print("    nmoo")
        m, t = memory_usage(
            (_time, (pareto_frontier_mask, points)),
            max_usage=True,
            retval=True,
        )
        results.append((N, d, "nmoo", m, t))
        for pm in pymoo_methods:
            print("    " + pm)
            sorter = NonDominatedSorting(method=pm)
            m, t = memory_usage(
                (_time, (sorter.do, points)),
                max_usage=True,
                retval=True,
            )
            results.append((N, d, pm, m, t))

    df = pd.DataFrame(
        results,
        columns=("N", "d", "algorithm", "memory", "time"),
    )
    df.to_csv("results.csv")

    for t in ["memory", "time"]:
        plot = sns.FacetGrid(
            df,
            col_wrap=3,
            col="N",
            hue="algorithm",
            sharey=False,
        )
        plot.map(sns.lineplot, "d", t)
        plot.add_legend()
        plot.savefig(f"{t}.jpg")
