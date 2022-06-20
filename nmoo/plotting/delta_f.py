"""
ΔF plots
"""
__docformat__ = "google"

from itertools import product
from math import sqrt
from pathlib import Path
from typing import Dict
import logging

from pymoo.core.problem import Problem
import numpy as np
import pandas as pd
import seaborn as sns

from nmoo.benchmark import Benchmark


def _load_problem_data(
    file_path: Path, ground_problem: Problem
) -> pd.DataFrame:
    """
    Loads a problem history and annotates it with true F values.
    """
    history = np.load(file_path)
    d1, d2 = pd.DataFrame(), pd.DataFrame()
    d1["_batch"] = d2["_batch"] = history["_batch"]
    d1["type"], d2["type"] = "approx.", "true"
    out: Dict[str, np.ndarray] = {}
    ground_problem._evaluate(history["X"], out)
    d1["F0"], d1["F1"] = history["F"][:, 0], history["F"][:, 1]
    d2["F0"], d2["F1"] = out["F"][:, 0], out["F"][:, 1]
    history.close()
    return pd.concat([d1, d2], ignore_index=True)


def generate_delta_F_plots(
    benchmark: Benchmark,
    n_generations: int = 10,
) -> None:
    """
    Generate all ΔF plots for a given benchmark, and save them as jpg image in
    the benchmark's output directory. The naming pattern is the same as in
    `nmoo.wrapped_problem.WrappedProblem.dump_all_histories`, except the files
    end with the `.jpg` extension instead of `.npz`.

    <center>
        <img src="https://github.com/altaris/noisy-moo/raw/main/imgs/generate_delta_F_plots.png"
        alt="Example"/>
    </center>

    A ΔF plot show the predicted value of a denoised noisy problem (in blue)
    against the true value of the base problem (in orange). In addition, the
    Pareto front is plotted in red. This kind of plot is only possible in a
    synthetic setting.

    Args:
        n_generations (int): Number of generation to plot.

    Warning:
        The ground pymoo problem must have a `pareto_front()` method that
        returns an actual array.
    """
    if not benchmark._dump_histories:
        raise RuntimeError(
            "The benchmark must have 'dump_histories=True' set when "
            "constructed."
        )
    everything = product(
        benchmark._algorithms.keys(),
        [(k, v["problem"]) for k, v in benchmark._problems.items()],
        range(1, benchmark._n_runs + 1),
    )
    for an, (pn, p), r in everything:
        if p.n_obj != 2:
            logging.warning(
                "Problem %s has %d objectives, but exactly 2 is needed for "
                "plotting",
                pn,
                p.n_obj,
            )
            continue
        ground_problem = p.ground_problem()
        # TODO: What if the ground_problem does not have a Pareto front?
        pareto_front = ground_problem.pareto_front()
        for li, l in enumerate(p.all_layers()):
            file_stem = f"{pn}.{an}.{r}.{li + 1}-{l._name}"
            df = _load_problem_data(
                benchmark._output_dir_path / (file_stem + ".npz"),
                ground_problem,
            )
            grid = sns.FacetGrid(
                df[
                    df._batch.isin(
                        np.linspace(
                            1, df._batch.max(), n_generations, dtype=int
                        )
                    )
                ],
                col="_batch",
                col_wrap=int(sqrt(n_generations)),
            )
            grid.map_dataframe(
                sns.scatterplot, x="F0", y="F1", style="type", hue="type"
            )
            grid.add_legend()
            if pareto_front.shape[0]:
                for ax in grid.axes:
                    ax.plot(pareto_front[:, 0], pareto_front[:, 1], "--r")
            grid.savefig(benchmark._output_dir_path / (file_stem + ".jpg"))
