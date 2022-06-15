# %%
from pathlib import Path
import os

from rich import print
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import (
    get_crossover,
    get_mutation,
    get_problem,
    get_sampling,
    get_termination,
)
from pymoo.core.problem import Problem
import numpy as np

import nmoo

# %% ==========================================================================
# Dynamic KNN
# =============================================================================

# from itertools import product
# from math import ceil, sqrt
# from scipy.spatial.distance import cdist
# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.metrics import silhouette_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler
# from typing import Callable, List


# class ScheduledKNR(nmoo.WrappedProblem):

#     _n_neighbors_schedule: Callable[[int], int]

#     def __init__(
#         self,
#         problem: Problem,
#         n_neighbors_schedule: Callable[[int], int],
#     ):
#         super().__init__(problem)
#         self._n_neighbors_schedule = n_neighbors_schedule

#     def _evaluate(self, x, out, *args, **kwargs):
#         self._problem._evaluate(x, out, *args, **kwargs)
#         hist_x = self._problem._history["X"]
#         hist_F = self._problem._history["F"]
#         n = len(hist_x) - len(x) if len(hist_x) > len(x) else None
#         n_neighbors_schedule = self._n_neighbors_schedule
#         n_neighbors = max(2, n_neighbors_schedule(len(hist_x)))
#         n_neighbors = min(len(hist_x[:n]), n_neighbors)
#         knr = KNeighborsRegressor(
#             metric="seuclidean",
#             metric_params={
#                 "V": np.var(hist_x, axis=0),
#             },
#             n_neighbors=n_neighbors,
#             weights="distance",
#         )
#         knr.fit(hist_x[:n], hist_F[:n])
#         out["F"] = knr.predict(x)
#         self.add_to_history_x_out(x, out)


# class AdaptativeMinibatchKmeansAvg(nmoo.WrappedProblem):

#     _batch_size: int

#     _distance_weight_mode: str

#     _kmeans: List[MiniBatchKMeans]

#     _max_n_clusters: int

#     def __init__(
#         self,
#         problem: Problem,
#         batch_size: int = 10,
#         distance_weight_type: str = "uniform",
#         max_n_clusters: int = 10,
#     ):
#         super().__init__(problem)
#         self._max_n_clusters = max_n_clusters

#         if batch_size < 1:
#             raise ValueError("Batch size should be at least 1.")
#         self._batch_size = batch_size

#         if distance_weight_type not in ["squared", "uniform"]:
#             raise ValueError(
#                 "Parameter distance_weight_type must be either 'squared' or "
#                 "'uniform'."
#             )
#         self._distance_weight_mode = distance_weight_type

#         if max_n_clusters < 2:
#             raise ValueError("Maximum number of clusters must be at least 2.")
#         if max_n_clusters > batch_size:
#             raise ValueError(
#                 "Maximum number of clusters must less or equal to the batch "
#                 f" size ({batch_size})."
#             )
#         self._max_n_clusters = max_n_clusters
#         self._init_kmeans()

#     def _best_kmeans(self) -> KMeans:
#         """
#         Returns:
#             Best KMeans classifier based on its silhouette score on the
#             wrapped problem's history.
#         """
#         x = self._problem._history["X"]
#         ss: List[float] = []
#         for km in self._kmeans:
#             try:
#                 s = silhouette_score(x, km.predict(x))
#             except ValueError:
#                 s = -1
#             ss.append(s)
#         return self._kmeans[np.array(ss).argmax()]

#     def _evaluate(self, x, out, *args, **kwargs):
#         self._problem._evaluate(x, out, *args, **kwargs)
#         hist_x = pd.DataFrame(self._problem._history["X"])
#         self._update_kmeans(x)
#         km = self._best_kmeans()
#         hist_x["_label"] = km.predict(hist_x)
#         for i, (sol, label) in enumerate(zip(x, km.predict(x))):
#             # Compute the standardized Euclidean distances between the current
#             # solution (sol) and all historical solutions.
#             # TODO: Optimize: Variance doesn't need to be recalculated every
#             # time.
#             hist_x["_sed"] = cdist(
#                 self._problem._history["X"],
#                 sol.reshape((1, -1)),
#                 "seuclidean",
#             )
#             # Select sol's cluster.
#             df = hist_x[hist_x["_label"] == label].copy()
#             if len(df.shape) <= 1:
#                 # If sol is alone, skip this current iteration.
#                 continue
#             # Compute the weights.
#             # TODO: Is there a standardized distance-weight method?
#             if self._distance_weight_mode == "squared":
#                 max_sed = df["_sed"].max()
#                 # TODO: Does the +1 even make sense?
#                 df["_w"] = (1 + max_sed - df["_sed"]) ** 2
#             elif self._distance_weight_mode == "uniform":
#                 df["_w"] = 1.0
#             else:
#                 raise RuntimeError(
#                     "Unknown distance weight mode: "
#                     + self._distance_weight_mode
#                 )
#             # Compute the weighted averages of the (numerical) outputs.
#             for k in out:
#                 if not isinstance(out[k], np.ndarray):
#                     continue
#                 hist_out_k = pd.DataFrame(self._problem._history[k])
#                 avg = np.average(
#                     hist_out_k.iloc[df.index],
#                     axis=0,
#                     weights=df["_w"],
#                 )
#                 out[k][i] = avg
#         self.add_to_history_x_out(x, out, _k=np.full((len(x),), km.n_clusters))

#     def _init_kmeans(self) -> None:
#         self._kmeans = [
#             MiniBatchKMeans(n_clusters=i, batch_size=self._batch_size)
#             for i in range(2, self._max_n_clusters + 1)
#         ]

#     def _update_kmeans(self, x: np.ndarray) -> None:
#         """
#         Updates the kmeans, and returns the best one.

#         Args:
#             x (np.ndarray): NEW values on which to train the KMeans
#                 classifiers.
#         """
#         # def _fit_kmeans(km: MiniBatchKMeans, y: np.ndarray):
#         #     for z in np.split(y, self._batch_size):
#         #         km.partial_fit(z)
#         # executor = Parallel(-1)
#         # executor(delayed(_fit_kmeans)(km, x) for km in self._kmeans)
#         # TODO: Parallelize
#         everything = product(
#             self._kmeans, np.split(x, len(x) / self._batch_size)
#         )
#         for km, batch in everything:
#             km.partial_fit(batch)

#     def start_new_run(self):
#         super().start_new_run()
#         self._init_kmeans()


# class AdaptativeKmeansAvg(nmoo.WrappedProblem):

#     _distance_weight_mode: str

#     _max_n_clusters: int

#     def __init__(
#         self,
#         problem: Problem,
#         distance_weight_type: str = "uniform",
#         max_n_clusters: int = 10,
#     ):
#         super().__init__(problem)

#         if distance_weight_type not in ["squared", "uniform"]:
#             raise ValueError(
#                 "Parameter distance_weight_type must be either 'squared' or "
#                 "'uniform'."
#             )
#         self._distance_weight_mode = distance_weight_type

#         if max_n_clusters < 2:
#             raise ValueError("Maximum number of clusters must be at least 2.")
#         self._max_n_clusters = max_n_clusters

#     def _best_kmeans(self, x: np.ndarray) -> KMeans:
#         """
#         Args:
#             x (np.ndarray): Population to cluster. Should be of shape `(N, M)`,
#                 where `N` is geater of equal to `_max_n_clusters`.

#         Returns:
#             Best KMeans classifier based on its silhouette score.
#         """
#         kms: List[KMeans] = []
#         ss: List[float] = []
#         for i in range(2, self._max_n_clusters + 1):
#             km = KMeans(n_clusters=i)
#             km.fit(x)
#             kms.append(km)
#             ss.append(silhouette_score(x, km.labels_) - km.inertia_)
#         return kms[np.array(ss).argmax()]

#     def _evaluate(self, x, out, *args, **kwargs):
#         self._problem._evaluate(x, out, *args, **kwargs)
#         hist_x = pd.DataFrame(self._problem._history["X"])
#         n = len(hist_x) - len(x) if len(hist_x) > len(x) else None
#         # Fit the scaler to the PAST solution set. If there is no PAST
#         # solutions (which happens during the first generation), then fit to
#         # the current solutions instead. Then, scale all solutions.
#         scaler = StandardScaler()
#         scaler.fit(hist_x.iloc[:n])
#         hist_x[:] = scaler.transform(hist_x)
#         km = self._best_kmeans(hist_x)
#         hist_x["_label"] = km.predict(hist_x)
#         for i, (sol, label) in enumerate(zip(x, hist_x["_label"].iloc[n:])):
#             # Compute the distances between the current solution (sol) and all
#             # historical solutions.
#             hist_x["_d"] = cdist(
#                 self._problem._history["X"],
#                 sol.reshape((1, -1)),
#                 # "seuclidean",
#             )
#             # Select sol's cluster.
#             df = hist_x[hist_x["_label"] == label].copy()
#             if len(df.shape) <= 1:
#                 # If sol is alone, skip this current iteration.
#                 continue
#             # Compute the weights.
#             # TODO: Is there a standardized distance-weight method?
#             if self._distance_weight_mode == "squared":
#                 max_d = df["_d"].max()
#                 df["_w"] = (1.0 + max_d - df["_d"]) ** 2
#             elif self._distance_weight_mode == "uniform":
#                 df["_w"] = 1.0
#             else:
#                 raise RuntimeError(
#                     "Unknown distance weight mode: "
#                     + self._distance_weight_mode
#                 )
#             # Compute the weighted averages of the (numerical) outputs.
#             for k in out:
#                 if not isinstance(out[k], np.ndarray):
#                     continue
#                 hist_out_k = pd.DataFrame(self._problem._history[k])
#                 avg = np.average(
#                     hist_out_k.iloc[df.index],
#                     axis=0,
#                     weights=df["_w"],
#                 )
#                 out[k][i] = avg
#         self.add_to_history_x_out(x, out, _k=np.full((len(x),), km.n_clusters))

#     def _update_kmeans(self, x: np.ndarray) -> None:
#         """
#         Updates the kmeans, and returns the best one.

#         Args:
#             x (np.ndarray): NEW values on which to train the KMeans
#                 classifiers.
#         """
#         # def _fit_kmeans(km: MiniBatchKMeans, y: np.ndarray):
#         #     for z in np.split(y, self._batch_size):
#         #         km.partial_fit(z)
#         # executor = Parallel(-1)
#         # executor(delayed(_fit_kmeans)(km, x) for km in self._kmeans)
#         # TODO: Parallelize
#         everything = product(
#             self._kmeans, np.split(x, len(x) / self._batch_size)
#         )
#         for km, batch in everything:
#             km.partial_fit(batch)


# %% ==========================================================================
# Benchmark
# =============================================================================

problem = get_problem("zdt2")
pareto_front = problem.pareto_front()

mean = np.array([0, 0])
# cov = np.array([[1., -.5], [-.5, 1]])
cov = 0.2 * np.eye(2, dtype=float)
noisy_problem = nmoo.noises.GaussianNoise(
    nmoo.WrappedProblem(problem), {"F": (mean, cov)}
)

avg5 = nmoo.denoisers.ResampleAverage(
    noisy_problem,
    n_evaluations=5,
)
avg10 = nmoo.denoisers.ResampleAverage(
    noisy_problem,
    n_evaluations=10,
)
knnavg3 = nmoo.denoisers.KNNAvg(
    noisy_problem,
    distance_weight_type="squared",
    max_distance=0.5,
    n_neighbors=3,
)
knnavg5 = nmoo.denoisers.KNNAvg(
    noisy_problem,
    distance_weight_type="squared",
    max_distance=0.5,
    n_neighbors=5,
)
knnavg10 = nmoo.denoisers.KNNAvg(
    noisy_problem,
    distance_weight_type="squared",
    max_distance=0.5,
    n_neighbors=10,
)
knnavg100 = nmoo.denoisers.KNNAvg(
    noisy_problem,
    distance_weight_type="squared",
    max_distance=0.5,
    n_neighbors=100,
)
# sknr3s = ScheduledKNR(
#     noisy_problem,
#     n_neighbors_schedule=lambda n: 3 * ceil(sqrt(n)),
# )
# sknr5s = ScheduledKNR(
#     noisy_problem,
#     n_neighbors_schedule=lambda n: 5 * ceil(sqrt(n)),
# )
# sknr10s = ScheduledKNR(
#     noisy_problem,
#     n_neighbors_schedule=lambda n: 10 * ceil(sqrt(n)),
# )
# sknr3 = ScheduledKNR(
#     noisy_problem,
#     n_neighbors_schedule=lambda _: 3,
# )
# sknr5 = ScheduledKNR(
#     noisy_problem,
#     n_neighbors_schedule=lambda _: 5,
# )
# sknr10 = ScheduledKNR(
#     noisy_problem,
#     n_neighbors_schedule=lambda _: 10,
# )
# akma3 = AdaptativeKmeansAvg(
#     noisy_problem,
#     distance_weight_type="squared",
#     max_n_clusters=3,
# )
# akma5 = AdaptativeKmeansAvg(
#     noisy_problem,
#     distance_weight_type="squared",
#     max_n_clusters=5,
# )
# akma10 = AdaptativeKmeansAvg(
#     noisy_problem,
#     distance_weight_type="squared",
#     max_n_clusters=10,
# )


class ProblemThatAlwaysFails(Problem):
    def _evaluate(self, x, out, *args, **kwargs):
        raise RuntimeError("Nope")


nsga2 = NSGA2(
    pop_size=20,
    eliminate_duplicates=True,
)

OUT_PATH = Path("./out")
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

benchmark = nmoo.benchmark.Benchmark(
    output_dir_path=OUT_PATH,
    problems={
        # "sknr03s": {
        #     "problem": sknr3s,
        #     "pareto_front": pareto_front,
        # },
        # "sknr05s": {
        #     "problem": sknr5s,
        #     "pareto_front": pareto_front,
        # },
        # "sknr10s": {
        #     "problem": sknr10s,
        #     "pareto_front": pareto_front,
        # },
        # "sknr3": {
        #     "problem": sknr3,
        #     "pareto_front": pareto_front,
        # },
        # "sknr5": {
        #     "problem": sknr5,
        #     "pareto_front": pareto_front,
        # },
        # "sknr10": {
        #     "problem": sknr10,
        #     "pareto_front": pareto_front,
        # },
        # "akma3": {
        #     "problem": akma3,
        #     # "pareto_front": pareto_front,
        #     "hv_ref_point": np.array([5, 5])
        # },
        # "akma5": {
        #     "problem": akma5,
        #     "pareto_front": pareto_front,
        # },
        # "akma10": {
        #     "problem": akma10,
        #     "pareto_front": pareto_front,
        # },
        # "knnavg3": {
        #     "problem": knnavg3,
        #     "pareto_front": pareto_front,
        # },
        # "knnavg5": {
        #     "problem": knnavg5,
        #     "pareto_front": pareto_front,
        # },
        # "knnavg10": {
        #     "problem": knnavg10,
        #     "pareto_front": pareto_front,
        # },
        "knnavg100": {
            "problem": knnavg100,
            # "pareto_front": pareto_front,
        },
        # "avg5": {
        #     "problem": avg5,
        #     "pareto_front": pareto_front,
        # },
        "avg10": {
            "problem": avg10,
            # "pareto_front": pareto_front,
        },
        # "noisy_zdt1": {
        #     "problem": noisy_problem
        # },
        "non_noisy": {
            "problem": nmoo.WrappedProblem(problem),
            # "evaluator": nmoo.evaluators.EvaluationPenaltyEvaluator(),
        },
        "nope": {
            "problem": nmoo.WrappedProblem(ProblemThatAlwaysFails())
        },
    },
    algorithms={
        "nsga2_40": {
            "algorithm": nsga2,
            "termination": get_termination("n_gen", 40),
        },
        # "nsga2_100": {
        #     "algorithm": nsga2,
        #     "termination": get_termination("n_eval", 100),
        # },
        # "nsga2": {
        #     "algorithm": nsga2,
        # },
    },
    n_runs=5,
    max_retry=5,
    dump_histories=True,
    performance_indicators=["ps", "df", "gd", "gd+", "igd", "igd+"],
)

# %%
import logging
logging.basicConfig(level=logging.DEBUG)

benchmark.run(n_jobs=-1, n_post_processing_jobs=-1)

# %% ==========================================================================
# Visualization
# =============================================================================

from nmoo.plotting import plot_performance_indicators

# Uncomment if you want to inject benchmark results, if you didn't run it in
# this session
# benchmark._results = pd.read_csv(benchmark._output_dir_path / "benchmark.csv")
plot_performance_indicators(benchmark, row="problem")

# %%

from nmoo.plotting import generate_delta_F_plots

generate_delta_F_plots(benchmark, 50)

# %% Not great
from itertools import product

def plot_F_predictions_combined(
    benchmark: nmoo.benchmark.Benchmark, n_samples: int
):
    algorithms = benchmark._algorithms.keys()
    problems = benchmark._problems.keys()
    levels = ["1-knn_avg", "2-gaussian_noise", "3-wrapped_problem"]
    for a, p in product(algorithms, problems):
        df = pd.DataFrame()
        for l in levels:
            print(benchmark._output_dir_path / f"{p}.{a}.1.{l}.npz")
            history = np.load(
                benchmark._output_dir_path / f"{p}.{a}.1.{l}.npz"
            )
            tmp = pd.DataFrame()
            tmp["_batch"] = history["_batch"]
            tmp["level"] = l
            for i in range(problem.n_obj):
                tmp[f"F{i}"] = history["F"][:, i]
            df = df.append(tmp, ignore_index=True)
        tmp = pd.DataFrame()
        tmp["_batch"] = history["_batch"]
        tmp["level"] = "true"
        gp = benchmark._problems[p]["problem"].ground_problem()
        out = {}
        gp._evaluate(history["X"], out)
        for i in range(problem.n_obj):
            tmp[f"F{i}"] = out["F"][:, i]
        df = df.append(tmp, ignore_index=True)

        n_samples = 50
        r = np.linspace(1, df._batch.max(), n_samples, dtype=int)
        grid = sns.FacetGrid(
            df[df._batch.isin(r)],
            col="_batch",
            col_wrap=int(sqrt(n_samples)),
        )
        grid.map_dataframe(
            sns.scatterplot, x="F0", y="F1", style="level", hue="level"
        )
        grid.add_legend()
        pareto_front = gp.pareto_front()
        for ax in grid.axes:
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], "--r")
        grid.savefig(benchmark._output_dir_path / f"{p}.{a}.1.jpg")

plot_F_predictions_combined(benchmark, 50)


# %%
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize

zdt1 = get_problem("zdt1")
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True,
)

res = minimize(
    zdt1,
    algorithm,
    get_termination("n_gen", 40),
    save_history=True,
    verbose=True,
)

# %% ==========================================================================
# Dataframe tests
# =============================================================================


# %% ==========================================================================
# Numpy tests
# =============================================================================

import numpy as np
from time import time

# from nmoo.utils.population import pareto_frontier_mask

def pareto_frontier_mask(arr: np.ndarray) -> np.ndarray:
    """
    Computes the Pareto frontier of a set of points. The array must have an
    `ndim` of 2.

    Returns:
        A mask (array of booleans) on `arr` selecting the Pareto points
        belonging to the Pareto frontier. The actual Pareto frontier can be
        computed with

            pfm = pareto_frontier_mask(arr)
            pf = arr[pfm]

    Warning:
        The direction of the optimum is assumed to be south-west.

    Todo:
        * Possibility to specify a direction of optimum;
        * fancy tree-based optimization.

    """
    if not arr.ndim == 2:
        raise ValueError("The input array must be of shape (N, d).")

    d = arr.shape[-1]

    argsort0, mask = np.argsort(arr[:,0]), np.full(len(arr), True)
    for i, j in enumerate(argsort0):
        if not mask[j]:
            continue
        for k in filter(lambda x: mask[x], argsort0[i+1:]):
            # faster than (arr[k] <= arr[j]).any()
            for l in range(1, d):
                if arr[k][l] <= arr[j][l]:
                    mask[k] = True
                    break
                mask[k] = False
    return mask

def pareto_frontier_mask2(arr: np.ndarray) -> np.ndarray:
    """
    Computes the Pareto frontier of a set of points. The array must have an
    `ndim` of 2.

    Returns:
        A mask (array of booleans) on `arr` selecting the Pareto points
        belonging to the Pareto frontier. The actual Pareto frontier can be
        computed with

            pfm = pareto_frontier_mask(arr)
            pf = arr[pfm]

    Warning:
        The direction of the optimum is assumed to be south-west.

    Todo:
        * Possibility to specify a direction of optimum;
        * fancy tree-based optimization.

    """
    if not arr.ndim == 2:
        raise ValueError("The input array must be of shape (N, d).")

    d = arr.shape[-1]

    argsort0, mask = np.argsort(arr[:,0]), np.full(len(arr), True)
    for i, j in enumerate(argsort0):
        if not mask[j]:
            continue
        for k in filter(lambda x: mask[x], argsort0[i+1:]):
            # faster than (arr[k] <= arr[j]).any()
            for l in range(1, d):
                if arr[k][l] <= arr[j][l]:
                    mask[k] = True
                    break
            else:
                mask[k] = False
    return mask


N, d = 5_000, 10
points = np.random.RandomState().normal([0] * d, .3, (N, d))
# points = points[np.argsort(points[:,0])]

start = time()
mask = pareto_frontier_mask(points)
print(time() - start)

print("=====")

start = time()
mask2 = pareto_frontier_mask2(points)
print(time() - start)

print("=====")

print((mask == mask2).all())
# print(mask == mask2)
# print(np.logical_or(np.logical_not(mask), mask2).all())

# df = pd.DataFrame()
# df["x"], df["y"] = points.T
# df["pareto"] = mask2

# sns.scatterplot(data=df, x="x", y="y", hue="pareto", style="pareto")