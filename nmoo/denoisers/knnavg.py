"""KNN-Averaging"""
__docformat__ = "google"

from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

from nmoo.wrapped_problem import WrappedProblem


class KNNAvg(WrappedProblem):
    """
    Implementation of the KNN-Avg algorithm of Klikovits and
    Arcaini[^quatic21].

    [^quatic21]: Klikovits, S., Arcaini, P. (2021). KNN-Averaging for Noisy
    Multi-objective Optimisation. In: Paiva, A.C.R., Cavalli, A.R., Ventura
    Martins, P., Pérez-Castillo, R. (eds) Quality of Information and
    Communications Technology. QUATIC 2021. Communications in Computer and
    Information Science, vol 1439. Springer, Cham.
    https://doi.org/10.1007/978-3-030-85347-1_36
    """

    _distance_weight_mode: str
    _max_distance: float
    _n_neighbors: int

    def __init__(
        self,
        problem: WrappedProblem,
        max_distance: float,
        n_neighbors: int = 5,  # KNN
        distance_weight_type: str = "uniform",
        *,
        name: str = "knn_avg",
        **kwargs,
    ):
        """
        Constructor.

        Args:
            problem (:obj:`WrappedProblem`): Noisy problem. For memory
                optimization reasons, this should be a `WrappedProblem` as
                opposed to a pymoo `Problem`.
            distance_weight_type (str): Either "squared" or "uniform".
            max_distance (float): Distance cutoff.
            n_neighbors (int): Number of neighbors to consider (KNN).
            name (str): An optional name for this problem. This will be used
                when creating history dump files. Defaults to `knn_avg`.
        """
        super().__init__(problem, name=name, **kwargs)

        if distance_weight_type not in ["squared", "uniform"]:
            raise ValueError(
                "Parameter distance_weight_type must be either 'squared' or "
                "'uniform'."
            )
        self._distance_weight_mode = distance_weight_type

        if max_distance < 0.0:
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
            # Store the solution history into a dataframe (note that we are
            # using the wrapped problem's history to make sure this dataframe
            # is never empty).
            x_hist = pd.DataFrame(self._problem._history["X"])
            # Compute the standardized Euclidean distances between the current
            # solution (sol) and all historical solutions.
            x_hist["_sed"] = cdist(
                self._problem._history["X"],
                sol.reshape((1, -1)),
                "seuclidean",
            )
            # Apply the KNN scheme: select the K closest neighbors among those
            # closer than the maximum allowed distance.
            x_hist = (
                x_hist[x_hist["_sed"] <= self._max_distance]
                .sort_values(by="_sed")
                .head(self._n_neighbors)
            )
            if x_hist.shape[0] <= 1:
                # If only the current solution remains, then skip to the next
                # solution.
                continue
            # Compute the weights.
            if self._distance_weight_mode == "squared":
                x_hist["_w"] = (self._max_distance - x_hist["_sed"]) ** 2
            elif self._distance_weight_mode == "uniform":
                x_hist["_w"] = 1.0
            else:
                raise RuntimeError(
                    "Unknown distance weight mode: "
                    + self._distance_weight_mode
                )
            # Compute the weighted averages of the (numerical) outputs.
            for k in out:
                if not isinstance(out[k], np.ndarray):
                    continue
                out_k_hist = pd.DataFrame(self._problem._history[k])
                avg = np.average(
                    out_k_hist.iloc[x_hist.index],
                    axis=0,
                    weights=x_hist["_w"],
                )
                out[k][i] = avg
        self.add_to_history_x_out(x, out)
