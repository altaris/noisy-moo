from itertools import product
from math import ceil, sqrt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from typing import Callable, List


class ScheduledKNR(nmoo.WrappedProblem):

    _n_neighbors_schedule: Callable[[int], int]

    def __init__(
        self,
        problem: Problem,
        n_neighbors_schedule: Callable[[int], int],
    ):
        super().__init__(problem)
        self._n_neighbors_schedule = n_neighbors_schedule

    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)
        hist_x = self._problem._history["X"]
        hist_F = self._problem._history["F"]
        n = len(hist_x) - len(x) if len(hist_x) > len(x) else None
        n_neighbors_schedule = self._n_neighbors_schedule
        n_neighbors = max(2, n_neighbors_schedule(len(hist_x)))
        n_neighbors = min(len(hist_x[:n]), n_neighbors)
        knr = KNeighborsRegressor(
            metric="seuclidean",
            metric_params={
                "V": np.var(hist_x, axis=0),
            },
            n_neighbors=n_neighbors,
            weights="distance",
        )
        knr.fit(hist_x[:n], hist_F[:n])
        out["F"] = knr.predict(x)
        self.add_to_history_x_out(x, out)


class AdaptativeMinibatchKmeansAvg(nmoo.WrappedProblem):

    _batch_size: int

    _distance_weight_mode: str

    _kmeans: List[MiniBatchKMeans]

    _max_n_clusters: int

    def __init__(
        self,
        problem: Problem,
        batch_size: int = 10,
        distance_weight_type: str = "uniform",
        max_n_clusters: int = 10,
    ):
        super().__init__(problem)
        self._max_n_clusters = max_n_clusters

        if batch_size < 1:
            raise ValueError("Batch size should be at least 1.")
        self._batch_size = batch_size

        if distance_weight_type not in ["squared", "uniform"]:
            raise ValueError(
                "Parameter distance_weight_type must be either 'squared' or "
                "'uniform'."
            )
        self._distance_weight_mode = distance_weight_type

        if max_n_clusters < 2:
            raise ValueError("Maximum number of clusters must be at least 2.")
        if max_n_clusters > batch_size:
            raise ValueError(
                "Maximum number of clusters must less or equal to the batch "
                f" size ({batch_size})."
            )
        self._max_n_clusters = max_n_clusters
        self._init_kmeans()

    def _best_kmeans(self) -> KMeans:
        """
        Returns:
            Best KMeans classifier based on its silhouette score on the
            wrapped problem's history.
        """
        x = self._problem._history["X"]
        ss: List[float] = []
        for km in self._kmeans:
            try:
                s = silhouette_score(x, km.predict(x))
            except ValueError:
                s = -1
            ss.append(s)
        return self._kmeans[np.array(ss).argmax()]

    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)
        hist_x = pd.DataFrame(self._problem._history["X"])
        self._update_kmeans(x)
        km = self._best_kmeans()
        hist_x["_label"] = km.predict(hist_x)
        for i, (sol, label) in enumerate(zip(x, km.predict(x))):
            # Compute the standardized Euclidean distances between the current
            # solution (sol) and all historical solutions.
            # TODO: Optimize: Variance doesn't need to be recalculated every
            # time.
            hist_x["_sed"] = cdist(
                self._problem._history["X"],
                sol.reshape((1, -1)),
                "seuclidean",
            )
            # Select sol's cluster.
            df = hist_x[hist_x["_label"] == label].copy()
            if len(df.shape) <= 1:
                # If sol is alone, skip this current iteration.
                continue
            # Compute the weights.
            # TODO: Is there a standardized distance-weight method?
            if self._distance_weight_mode == "squared":
                max_sed = df["_sed"].max()
                # TODO: Does the +1 even make sense?
                df["_w"] = (1 + max_sed - df["_sed"]) ** 2
            elif self._distance_weight_mode == "uniform":
                df["_w"] = 1.0
            else:
                raise RuntimeError(
                    "Unknown distance weight mode: " + self._distance_weight_mode
                )
            # Compute the weighted averages of the (numerical) outputs.
            for k in out:
                if not isinstance(out[k], np.ndarray):
                    continue
                hist_out_k = pd.DataFrame(self._problem._history[k])
                avg = np.average(
                    hist_out_k.iloc[df.index],
                    axis=0,
                    weights=df["_w"],
                )
                out[k][i] = avg
        self.add_to_history_x_out(x, out, _k=np.full((len(x),), km.n_clusters))

    def _init_kmeans(self) -> None:
        self._kmeans = [
            MiniBatchKMeans(n_clusters=i, batch_size=self._batch_size)
            for i in range(2, self._max_n_clusters + 1)
        ]

    def _update_kmeans(self, x: np.ndarray) -> None:
        """
        Updates the kmeans, and returns the best one.

        Args:
            x (np.ndarray): NEW values on which to train the KMeans
                classifiers.
        """
        # def _fit_kmeans(km: MiniBatchKMeans, y: np.ndarray):
        #     for z in np.split(y, self._batch_size):
        #         km.partial_fit(z)
        # executor = Parallel(-1)
        # executor(delayed(_fit_kmeans)(km, x) for km in self._kmeans)
        # TODO: Parallelize
        everything = product(self._kmeans, np.split(x, len(x) / self._batch_size))
        for km, batch in everything:
            km.partial_fit(batch)

    def start_new_run(self):
        super().start_new_run()
        self._init_kmeans()


class AdaptativeKmeansAvg(nmoo.WrappedProblem):

    _distance_weight_mode: str

    _max_n_clusters: int

    def __init__(
        self,
        problem: Problem,
        distance_weight_type: str = "uniform",
        max_n_clusters: int = 10,
    ):
        super().__init__(problem)

        if distance_weight_type not in ["squared", "uniform"]:
            raise ValueError(
                "Parameter distance_weight_type must be either 'squared' or "
                "'uniform'."
            )
        self._distance_weight_mode = distance_weight_type

        if max_n_clusters < 2:
            raise ValueError("Maximum number of clusters must be at least 2.")
        self._max_n_clusters = max_n_clusters

    def _best_kmeans(self, x: np.ndarray) -> KMeans:
        """
        Args:
            x (np.ndarray): Population to cluster. Should be of shape `(N, M)`,
                where `N` is geater of equal to `_max_n_clusters`.

        Returns:
            Best KMeans classifier based on its silhouette score.
        """
        kms: List[KMeans] = []
        ss: List[float] = []
        for i in range(2, self._max_n_clusters + 1):
            km = KMeans(n_clusters=i)
            km.fit(x)
            kms.append(km)
            ss.append(silhouette_score(x, km.labels_) - km.inertia_)
        return kms[np.array(ss).argmax()]

    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)
        hist_x = pd.DataFrame(self._problem._history["X"])
        n = len(hist_x) - len(x) if len(hist_x) > len(x) else None
        # Fit the scaler to the PAST solution set. If there is no PAST
        # solutions (which happens during the first generation), then fit to
        # the current solutions instead. Then, scale all solutions.
        scaler = StandardScaler()
        scaler.fit(hist_x.iloc[:n])
        hist_x[:] = scaler.transform(hist_x)
        km = self._best_kmeans(hist_x)
        hist_x["_label"] = km.predict(hist_x)
        for i, (sol, label) in enumerate(zip(x, hist_x["_label"].iloc[n:])):
            # Compute the distances between the current solution (sol) and all
            # historical solutions.
            hist_x["_d"] = cdist(
                self._problem._history["X"],
                sol.reshape((1, -1)),
                # "seuclidean",
            )
            # Select sol's cluster.
            df = hist_x[hist_x["_label"] == label].copy()
            if len(df.shape) <= 1:
                # If sol is alone, skip this current iteration.
                continue
            # Compute the weights.
            # TODO: Is there a standardized distance-weight method?
            if self._distance_weight_mode == "squared":
                max_d = df["_d"].max()
                df["_w"] = (1.0 + max_d - df["_d"]) ** 2
            elif self._distance_weight_mode == "uniform":
                df["_w"] = 1.0
            else:
                raise RuntimeError(
                    "Unknown distance weight mode: " + self._distance_weight_mode
                )
            # Compute the weighted averages of the (numerical) outputs.
            for k in out:
                if not isinstance(out[k], np.ndarray):
                    continue
                hist_out_k = pd.DataFrame(self._problem._history[k])
                avg = np.average(
                    hist_out_k.iloc[df.index],
                    axis=0,
                    weights=df["_w"],
                )
                out[k][i] = avg
        self.add_to_history_x_out(x, out, _k=np.full((len(x),), km.n_clusters))

    def _update_kmeans(self, x: np.ndarray) -> None:
        """
        Updates the kmeans, and returns the best one.

        Args:
            x (np.ndarray): NEW values on which to train the KMeans
                classifiers.
        """
        # def _fit_kmeans(km: MiniBatchKMeans, y: np.ndarray):
        #     for z in np.split(y, self._batch_size):
        #         km.partial_fit(z)
        # executor = Parallel(-1)
        # executor(delayed(_fit_kmeans)(km, x) for km in self._kmeans)
        # TODO: Parallelize
        everything = product(self._kmeans, np.split(x, len(x) / self._batch_size))
        for km, batch in everything:
            km.partial_fit(batch)
