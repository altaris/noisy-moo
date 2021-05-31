"""
A denoiser tries to cancel noise. (also water is wet)
"""

from typing import List, Optional

from pymoo.model.problem import Problem
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

from .utils import *

Denoiser = ProblemWrapper


class KNNAvg(Denoiser):
    """
    Implementation of the NKK-Avg algorithm of Klikovits and Arcaini.

    See also:
        https://github.com/ERTOMMSD/QUATIC2021-KNN-Averaging
        https://raw.githubusercontent.com/ERATOMMSD/QUATIC2021-KNN-Averaging/main/KlikovitsArcaini-KNNAvgForNoisyNoisyMOO.pdf
    """

    _distance_weight_mode: str
    _max_distance: Optional[float]
    _n_neighbors: int

    def __init__(
        self,
        problem: Problem,
        distance_weight_type: str = "uniform",
        max_distance: Optional[float] = None,
        n_neighbors: int = 5,  # KNN
    ):
        """
        Constructor.

        Args:
            problem (:obj:`Problem`): Noisy pymoo problem.
            distance_weight_type (str): Either "squared" or "uniform".
            max_distance (Optional[float]): Distance cutoff.
            n_neighbors (int): Number of neighbors to consider (KNN).
        """
        super().__init__(problem)

        if distance_weight_type not in ["squared", "uniform"]:
            raise ValueError(
                "Parameter distance_weight_type must be either 'squared' or "
                "'uniform'."
            )
        self._distance_weight_mode = distance_weight_type

        if max_distance is not None and max_distance < 0.0:
            raise ValueError(
                "Parameter max_distance must either be 'None' or >= 0."
            )
        self._max_distance = max_distance

        if n_neighbors <= 0:
            raise ValueError("Parameter n_neighbors must be >= 1.")
        self._n_neighbors = n_neighbors

    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)
        if self._history.shape[0] != 0:
            for i, sol in enumerate(x):
                df = self._history.copy()
                df["_sed"] = cdist(
                    df.loc[:, "x_0":f"x_{self._problem.n_var - 1}"].to_numpy(),
                    sol.reshape((1, -1)),
                    "seuclidean",
                )
                max_distance = (
                    self._max_distance
                    if self._max_distance is not None
                    else df["_sed"].max() + 1.0
                    # TODO: Check how to handle self._max_distance == None
                )
                df = (
                    df[df["_sed"] <= max_distance]
                    .sort_values(by="_sed")
                    .head(self._n_neighbors)
                )
                if df.shape[0] <= 1:
                    continue
                if self._distance_weight_mode == "squared":
                    df["_w"] = (max_distance - df["_sed"]) ** 2
                    # TODO: Discrepency between the reference implementation
                    # and the paper (listing 1.1, l.21)
                elif self._distance_weight_mode == "uniform":
                    df["_w"] = 1.0
                else:
                    raise RuntimeError(
                        "Unknown distance weight mode: "
                        + self._distance_weight_mode
                    )
                # TODO: How to make this more general?
                avg = np.average(
                    df.loc[:, "F_0":"F_1"].to_numpy(),
                    axis=0,
                    weights=df["_w"].to_numpy(),
                )
                out["F"][i] = avg
        self._history = self._history.append(
            x_out_to_df(x, out),
            ignore_index=True,
        )
