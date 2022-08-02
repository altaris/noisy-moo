"""Gaussian process spectral sampling"""
__docformat__ = "google"

from typing import Any, Dict, Optional

import numpy as np
from pymoo.core.problem import Problem
from gradient_free_optimizers import SimulatedAnnealingOptimizer
from scipy.special import erfinv

from nmoo.wrapped_problem import WrappedProblem


# pylint: disable=too-many-instance-attributes
class GPSS(WrappedProblem):
    """
    Implementation of the gaussian process spectral sampling method described
    in [^tsemo]. Reference implementation:
    https://github.com/Eric-Bradford/TS-EMO/blob/master/TSEMO_V4.m .

    [^tsemo] Bradford, E., Schweidtmann, A.M. & Lapkin, A. Efficient
        multiobjective optimization employing Gaussian processes, spectral
        sampling and a genetic algorithm. J Glob Optim 71, 407–438 (2018).
        https://doi.org/10.1007/s10898-018-0609-2
    """

    _generator: np.random.Generator
    """Random number generator."""

    _n_mc_samples: int
    """Number of Monte-Carlo samples"""

    _nu: Optional[int]
    """Smoothness parameter of the Matérn covariance function $k$"""

    _xi: np.ndarray
    """Hyperparameters. See `__init__`."""

    _xi_map_search_n_iter: int
    """Number of iterations for the search of the MAP estimate of $\\xi$"""

    _xi_map_search_exclude_percentile: float
    """Area where to **not** search for MAP estimate of $\\xi$"""

    _xi_prior_mean: np.ndarray
    """Prior means of the components of $\\xi$"""

    _xi_prior_std: np.ndarray
    """Prior variances of the components of $\\xi$"""

    def __init__(
        self,
        problem: Problem,
        xi_prior_mean: np.ndarray,
        xi_prior_std: np.ndarray,
        nu: Optional[int] = None,
        n_mc_samples: int = 4000,
        xi_map_search_n_iter: int = 100,
        xi_map_search_exclude_percentile: float = 0.1,
        seed: Any = None,
        copy_problem: bool = True,
        name: str = "gpss",
    ):
        """
        Args:
            xi_prior_mean: Means of the (univariate) normal distributions that
                components of $\\xi$ are assumed to follow. Recall that $\\xi =
                [\\log \\lambda_1, \\ldots, \\log \\lambda_d, \\log \\sigma_f,
                \\log \\sigma_n]$, where $\\lambda_i$ is the length scale of
                input variable $i$, $d$ is the dimension of the input space
                (a.k.a. the number of variables), $\\sigma_f$ is the standard
                deviation of the output, and $\\sigma_n$ is the standard
                deviation of the noise. In particular, `xi_prior_mean` must
                have shape $(d+2,)$.
            xi_prior_std: Standard deviations of the (univariate) normal
                distributions that components of $\\xi$ are assumed to follow.
            n_mc_samples: Number of Monte-Carlo points for spectral sampling
            nu: $\\nu$ covariance smoothness parameter. Must currently be left
                to `None`.
            xi_map_search_n_iter: Number of iterations for the maximum à
                posteriori search for the $\\xi$ hyperparameter.
            xi_map_search_exclude_percentile: Percentile to **exclude** from
                the maximum à posteriori search for the $\\xi$ hyperparameter.
                For example, if left to $10%$, then the search will be confined
                to the $90%$ region centered around the mean. Should be in $(0,
                0.5)$.
        """
        super().__init__(problem, copy_problem=copy_problem, name=name)
        self.reseed(seed)
        if nu is not None:
            raise NotImplementedError(
                "The smoothness parameter nu must be left to None"
            )
        self._nu = nu
        self._n_mc_samples = n_mc_samples
        if xi_prior_mean.shape != (self.n_var + 2,):
            raise ValueError(
                "Invalid prior mean vector: it must have shape "
                f"(n_var + 2,), which in this case is ({self.n_var + 2},)."
            )
        if xi_prior_std.shape != (self.n_var + 2,):
            raise ValueError(
                "Invalid prior standard deviation vector: it must have shape "
                f"(n_var + 2,), which in this case is ({self.n_var + 2},)."
            )
        if not (xi_prior_std > 0.0).all():
            raise ValueError(
                "Invalid prior standard deviation vector: it can only have "
                "strictly positive components."
            )
        self._xi_prior_mean = xi_prior_mean
        self._xi_prior_std = xi_prior_std
        self._xi_map_search_n_iter = xi_map_search_n_iter
        self._xi_map_search_exclude_percentile = (
            xi_map_search_exclude_percentile
        )
        self._xi = np.array(
            [
                self._generator.normal(xi_prior_mean[i], xi_prior_std[i])
                for i in range(self.n_var + 2)
            ]
        )

    def reseed(self, seed: Any) -> None:
        self._generator = np.random.default_rng(seed)
        if isinstance(self._problem, WrappedProblem):
            self._problem.reseed(seed)
        super().reseed(seed)

    # pylint: disable=too-many-locals
    def _evaluate(self, x, out, *args, **kwargs):
        self._problem._evaluate(x, out, *args, **kwargs)

        self._xi = self._xi_map_search(x, out["F"])
        exp_xi = np.ma.exp(self._xi)
        f_std, n_std = exp_xi[-2:]

        # x: k x n_var, where k is the batch size
        # w: _n_mc_samples x n_var
        # w @ x.T: _n_mc_samples x k
        # b: _n_mc_samples x k
        # zeta_x: _n_mc_samples x k
        # theta: _n_mc_samples x n_obj
        # out["F"]: k x n_obj
        # z = zeta_x.T: k x _n_mc_samples
        # zzi_inv: _n_mc_samples x _n_mc_samples
        # m: _n_mc_samples x n_obj
        # v: _n_mc_samples x _n_mc_samples
        # theta[:,i] ~ N(m[:,i], V): _n_mc_samples x 1
        lambda_mat = np.diag(exp_xi[:-2])
        w = self._generator.multivariate_normal(
            np.zeros(self.n_var), lambda_mat, size=self._n_mc_samples
        )
        b = self._generator.uniform(
            0, 2 * np.pi, size=(self._n_mc_samples, x.shape[0])
        )
        zeta_x = (
            f_std * np.sqrt(2 / self._n_mc_samples) * np.ma.cos(w.dot(x.T) + b)
        )
        z = zeta_x.T
        zzi_inv = np.linalg.inv(
            z.T.dot(z) + (n_std**2) * np.eye(self._n_mc_samples)
        )
        m = zzi_inv.dot(z.T).dot(out["F"])
        v = zzi_inv * (n_std**2)
        # Ensure v is symmetric, as in the reference implementation
        v = 0.5 * (v + v.T)
        theta = np.stack(
            [
                # The mean needs to be 1-dimensional. The Cholesky
                # decomposition is used for performance reasons
                self._generator.multivariate_normal(
                    m[:, i].flatten(), v, method="cholesky"
                )
                for i in range(self.n_obj)
            ],
            axis=-1,  # stacks columns instead of rows
        )
        out["F"] = zeta_x.T.dot(theta)
        self.add_to_history_x_out(x, out)

    def _xi_map_search(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Maximum à posteriori estimate for $\\xi$"""

        def _objective_function(parameters: Dict[str, float]) -> float:
            xi = np.array(
                [parameters[f"xi_{i}"] for i in range(self._xi.shape[0])]
            )
            return _negative_log_likelyhood(
                xi, self._xi_prior_mean, self._xi_prior_std, x, y
            )

        # Use half the percentile in the formula for q
        q = (
            np.sqrt(2.0)
            * erfinv(self._xi_map_search_exclude_percentile - 1)
            * self._xi_prior_std
        )
        search_space = {
            f"xi_{i}": np.linspace(
                self._xi_prior_mean[i] + q[i],  # and not - q[i] !
                self._xi_prior_mean[i] - q[i],
                100,
            )
            for i in range(self._xi.shape[0])
        }
        optimizer = SimulatedAnnealingOptimizer(search_space)
        optimizer.search(
            _objective_function,
            n_iter=self._xi_map_search_n_iter,
            early_stopping={
                "n_iter_no_change": int(self._xi_map_search_n_iter / 5)
            },
            verbosity=[],
        )
        return np.array(
            [optimizer.best_para[f"xi_{i}"] for i in range(self._xi.shape[0])]
        )


def _negative_log_likelyhood(
    xi: np.ndarray,
    xi_prior_mean: np.ndarray,
    xi_prior_std: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Equation 15 but without the terms that don't depend on $\\xi$. Also, there
    is an issue with the $y^T \\Sigma^{-1} y$ term that isn't a scalar if the
    dimension of the output space is not $1$. To remediate this, we consider
    the, $L^\\infty$-norm (i.e. the absolute max) $\\Vert y^T \\Sigma^{-1} y
    \\Vert_\\infty$.
    """
    exp_xi = np.ma.exp(xi)
    f_std, n_std = exp_xi[-2:]
    # x: k x n_var, where k is the batch size
    # y: k x n_obj
    # lambda_mat: n_var x n_var
    # r_squared_mat: k x k
    # sigma_mat: k x k
    lambda_mat = np.diag(exp_xi[:-2])
    r_squared_mat = np.array(
        [[(a - b) @ lambda_mat @ (a - b).T for b in x] for a in x]
    )
    sigma_mat = (f_std**2) * np.ma.exp(
        -0.5 * r_squared_mat
    ) + n_std * np.eye(x.shape[0])
    sigma_mat = 0.5 * (sigma_mat + sigma_mat.T)
    sigma_det = np.linalg.det(sigma_mat)
    sigma_inv = np.linalg.inv(sigma_mat)
    y_sigma_y = y.T.dot(sigma_inv).dot(y)
    return (
        -0.5 * np.log(np.abs(sigma_det))
        - 0.5 * np.linalg.norm(y_sigma_y, np.inf)
        # - 0.5 * x.shape[0] * np.log(2.0 * np.pi)
        + np.sum(
            [
                # -0.5 * np.log(2.0 * np.pi)
                # - 0.5 * np.log(xi_prior_std[i] ** 2)
                -1.0
                / (2.0 * xi_prior_std[i] ** 2)
                * (xi[i] - xi_prior_mean[i]) ** 2
                for i in range(x.shape[1])
            ]
        )
    )
