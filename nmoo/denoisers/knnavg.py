"""
A denoiser tries to cancel noise. (also water is wet)
"""
__docformat__ = "google"

from typing import Optional

from pymoo.model.problem import Problem
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

from nmoo.utils import *


class KNNAvg(ProblemWrapper):
    """
    Implementation of the KNN-Avg algorithm of Klikovits and Arcaini.

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
        """
        Applies the KNN-Avg algorithm to the wrapped (noisy) problem's output.
        """
        self._problem._evaluate(x, out, *args, **kwargs)
        for i, sol in enumerate(x):
            df = pd.DataFrame(self._problem._history["x"])
            df["_sed"] = cdist(
                self._problem._history["x"],
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
            elif self._distance_weight_mode == "uniform":
                df["_w"] = 1.0
            else:
                raise RuntimeError(
                    "Unknown distance weight mode: "
                    + self._distance_weight_mode
                )
            # TODO: Generalize
            df_F = pd.DataFrame(self._problem._history["F"])
            avg = np.average(
                df_F.iloc[df.index].to_numpy(),
                axis=0,
                weights=df["_w"].to_numpy(),
            )
            out["F"][i] = avg
        self.add_to_history_x_out(x, out)
