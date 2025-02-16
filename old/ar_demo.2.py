"""
Accumulative resampling algorithm wrapper for noisy single/multi-objective
problems.
"""
__docformat__ = "google"

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger as logging
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from nmoo.wrapped_problem import WrappedProblem


class TerminationCriterionMet(Exception):
    """
    Raised by `ARDEMO._evaluate_individual` if the termination criterion has
    been met.
    """


class _Individual(Individual):
    """
    An [pymoo
    `Individual`](https://github.com/anyoptimization/pymoo/blob/master/pymoo/core/individual.py)
    but where attributes `F`, `G`, `dF`, `dG`, `ddF`, `ddG`, and `CV` are
    maximum likelyhood estimates of the true values.
    """

    _samples: Dict[str, np.ndarray]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._samples = {}

    def get_estimate(self, key: str) -> np.ndarray:
        """
        Return the maximum likelyhood estimate for for value of `key` by
        averaging all samplings of the objective function that have been made
        using this individual. If the individual has been correctly `update`d,
        the following two are equivalent.

            ind.get_estimate("F")
            ind.F

        """
        return self._samples[key].mean(axis=0)

    def update(self) -> None:
        """
        Adds the current value of `F`, `G` etc. to the sample arrays, and
        reverts the value of `F`, `G` etc. to the maximum likelyhood estimate.

        Here's the thing. When an evaluator `_eval`uates an individual, it
        changes its `F`, `G` etc. directly using `Population.set`. The easiest
        way to adapt this workflow to our needs is to accept that `self.F` will
        have a dual nature: either the latest evaluation (or sampling) of the
        objective function, or the maximum likelyhood estimate.
        """
        for key in ["F", "G", "dF", "dG", "ddF", "ddG", "CV"]:
            value = self.__dict__.get(key)
            if not isinstance(value, np.ndarray):
                break
            if key not in self._samples:
                self._samples[key] = value[np.newaxis]
            else:
                self._samples[key] = np.append(
                    self._samples[key], [value], axis=0
                )
            self.__dict__[key] = self.get_estimate(key)

    def n_eval(self) -> int:
        """
        The number of times this individual has been sampled.
        """
        return len(self._samples.get("F", []))


# pylint: disable=too-many-instance-attributes
class ARDEMO(DE):
    """
    Accumulative resampling algorithm wrapper for noisy single/multi-objective
    problems. The accumulative resampling methods are from [^elite].

    [^elite]: Fieldsend, J.E. (2015). Elite Accumulative Sampling Strategies
        for Noisy Multi-objective Optimisation. In: Gaspar-Cunha, A., Henggeler
        Antunes, C., Coello, C. (eds) Evolutionary Multi-Criterion
        Optimization. EMO 2015. Lecture Notes in Computer Science(), vol 9019.
        Springer, Cham. https://doi.org/10.1007/978-3-319-15892-1_12
    """

    SUPPORTED_RESAMPLING_METHODS = [
        "elite",
        "fixed",
        "min_on_conv",  # Not implemented
        "rate_on_conv",  # Not implemented
    ]

    # pymoo inherited properties
    pop: List[_Individual]
    n_gen: int

    _resampling_elite_cache: Dict[int, Tuple[int, int]] = {}
    """
    At key `t`, contains the average number of resamplings of Pareto
    individuals at generation `t`, and the size of the Pareto population at
    generation `t`. Used as caching for `_resampling_elite`.
    """

    _resampling_method: str
    """Algorithm used for resampling. See `ARDEMO.__init__`"""

    _rng: np.random.Generator

    def __init__(
        self,
        resampling_method: str = "fixed",
        convergence_time_window: int = 5,
        **kwargs,
    ):
        """
        Args:
            algorithm: Single/multi-objective optimization algorithm.
            convergence_time_window (int): Convergence time window for method 2
                and 3, (denoted by $m$ in [^elite])
            resampling_method (str): Resampling method
                * `fixed`: resampling rate is fixed; corresponds to algorithm 1
                  in [^elite];
                * `rate_on_conv` **NOT IMPLEMENTED**: resampling rate may
                  increase based on a convergence assessment that uses the
                  $\\varepsilon +$ indicator; corresponds to algorithm 2 in
                  [^elite];
                * `min_on_conv` **NOT IMPLEMENTED**: resampling rate *of elite
                  *members* may increase based on a convergence assessment that
                  uses the $\\varepsilon +$ indicator; corresponds to algorithm
                  3 in [^elite];
                * `elite`: resample counts of elite members increases over
                  time; corresponds to algorithm 4 in [^elite].

        [^elite]: Fieldsend, J.E. (2015). Elite Accumulative Sampling
            Strategies for Noisy Multi-objective Optimisation. In:
            Gaspar-Cunha, A., Henggeler Antunes, C., Coello, C. (eds)
            Evolutionary Multi-Criterion Optimization. EMO 2015. Lecture Notes
            in Computer Science(), vol 9019. Springer, Cham.
            https://doi.org/10.1007/978-3-319-15892-1_12
        """
        kwargs["n_offsprings"] = 1
        super().__init__(**kwargs)
        if resampling_method not in self.SUPPORTED_RESAMPLING_METHODS:
            raise ValueError(
                "Invalid resampling method. Supported methods are "
                + ", ".join(self.SUPPORTED_RESAMPLING_METHODS)
            )
        self._resampling_method = resampling_method
        self._rng = np.random.default_rng()
        self._convergence_time_window = convergence_time_window

    def _do_resampling(self) -> None:
        """
        Dispatches to `_resampling_elite`, `_resampling_fixed`,
        `_resampling_min_on_conv` or `_resampling_rate_on_conv` depending on
        the value of `_resampling_method`. Also catches
        `TerminationCriterionMet` exceptions.
        """
        method = {
            "fixed": self._resampling_fixed,
            "rate_on_conv": self._resampling_rate_on_conv,
            "min_on_conv": self._resampling_min_on_conv,
            "elite": self._resampling_elite,
        }.get(self._resampling_method)
        if method is None:
            logging.warning(
                "Invalid resampling method {}", self._resampling_method
            )
            return
        try:
            for _ in range(self.n_offsprings):
                method()
        except TerminationCriterionMet:
            return

    def _evaluate_individual(self, individual: _Individual) -> None:
        """Evaluates and updates an individual."""
        if self.n_gen >= 2 and self.termination.has_terminated(self):
            raise TerminationCriterionMet()
        self.evaluator.eval(
            self.problem, individual, skip_already_evaluated=False
        )
        individual.update()
        # Little hack so that WrappedProblem's see this evaluation as part of
        # the same batch as the infills of this generation
        problem = self.problem
        while isinstance(problem, WrappedProblem):
            problem._current_history_batch = self.n_gen
            problem._history["_batch"][-1] = self.n_gen
            problem = problem._problem

    def _pareto_population(self) -> List[_Individual]:
        """Returns the Pareto (aka elite) individuals."""
        if len(self.pop) == 0:
            return []
        sorter = NonDominatedSorting(method="efficient_non_dominated_sort")
        ranks = sorter.do(np.array([p.get_estimate("F") for p in self.pop]))
        # TODO: self.pop[ranks[0]] ?
        return [p for i, p in enumerate(self.pop) if i in ranks[0]]

    def _reevaluate_pareto_individual_with_fewest_evals(self) -> None:
        """
        Randomly choose a Pareto `_Individual` that has the fewest number of
        resamples, and reevaluates it.
        """
        pareto_population = self._pareto_population()
        counts = np.array([p.n_eval() for p in pareto_population])
        index = self._rng.choice(np.where(counts == counts.min())[0])
        self._evaluate_individual(pareto_population[index])

    def _resampling_elite(self) -> None:
        """
        Resample counts of elite members increases over time. Corresponds to
        algorithm 4 in Fieldsend's paper.
        """

        def _mean_n_eval_pareto() -> float:
            """
            Average number of times an individual in the current Pareto
            population has been evaluated. This is called
            `mean_num_resamp(A_t)` in Fieldsend's paper.
            """
            return np.mean([p.n_eval() for p in self._pareto_population()])

        pareto_population = self._pareto_population()
        arr = [p.n_eval() for p in pareto_population]
        self._resampling_elite_cache[self.n_gen] = (
            np.mean(arr),
            len(arr),
        )
        self._reevaluate_pareto_individual_with_fewest_evals()
        alpha = sum(
            [m * s for (m, s) in self._resampling_elite_cache.values()]
        ) / sum([s for (_, s) in self._resampling_elite_cache.values()])
        while _mean_n_eval_pareto() <= alpha:
            self._reevaluate_pareto_individual_with_fewest_evals()

    def _resampling_fixed(self) -> None:
        """
        Resampling rate is fixed. Corresponds to algorithm 1 in Fieldsend's
        paper.
        """
        self._reevaluate_pareto_individual_with_fewest_evals()

    def _resampling_min_on_conv(self) -> None:
        """
        Resampling rate *of elite members* may increase based on a convergence
        assessment that uses the $\\varepsilon +$ indicator. Corresponds to
        algorithm 3 in Fieldsend's paper.
        """
        raise NotImplementedError

    def _resampling_rate_on_conv(self) -> None:
        """
        Resampling rate may increase based on a convergence assessment that
        uses the $\\varepsilon +$ indicator. Corresponds to algorithm 2 in
        Fieldsend's paper.
        """
        raise NotImplementedError

    # pymoo overrides =========================================================

    def _advance(
        self,
        infills: Optional[Union[_Individual, List[_Individual]]] = None,
        **_,
    ) -> None:
        """
        Called after the infills (aka new individuals) have been evaluated.
        """
        if infills is None:
            raise ValueError(
                "ARDEMO's _advance needs the current iteration's infills"
            )
        _update_infills(infills)
        q = infills[0] if not isinstance(infills, Individual) else infills
        pi = q.data["parent"]
        if strictly_dominate(q.F, self.pop[pi].F):
            self.pop[pi] = q
        self._do_resampling()

    def _infill(self) -> Population:
        """
        Generate new individuals for the next generation. Calls `DE._infill`
        and converts the results to `_Individual`s.
        """
        pi, ai, bi, ci = self._rng.choice(len(self.pop), 4, replace=False)
        p, a, b, c = (
            _Individual(self.pop[pi].X),
            self.pop[ai].X,
            self.pop[bi].X,
            self.pop[ci].X,
        )
        p.data["parent"] = pi
        for i in range(len(p.X)):  # pylint: disable=consider-using-enumerate
            if np.random.rand() < self.mating.crossover.CR:
                p.X[i] = a[i] + self.mating.crossover.F * (b[i] - c[i])
        return Population.create(p)

    def _initialize_advance(
        self,
        infills: Optional[Union[_Individual, List[_Individual]]] = None,
        **_,
    ) -> None:
        """Only called after the first generation has been evaluated"""
        if infills is None:
            raise ValueError(
                "ARDEMO's _advance needs the current iteration's infills"
            )
        _update_infills(infills)
        self._do_resampling()

    def _initialize_infill(self) -> Population:
        """
        Only called to get the first generation. Subsequent generations are
        generated by calling `_infill`.
        """
        population = super()._initialize_infill()
        return Population.create(*[_Individual(X=p.X) for p in population])

    def _setup(self, problem, **kwargs) -> None:
        """Called before an algorithm starts running on a problem"""
        super()._setup(problem, **kwargs)
        self._rng = np.random.default_rng(kwargs.get("seed"))
        self._resampling_elite_cache = {}


def _update_infills(infills: Union[_Individual, List[_Individual]]) -> None:
    """
    Takes evaluated infills of type `_Individual` and `_Individual.update`s
    them.
    """
    if isinstance(infills, _Individual):
        infills = [infills]
    for p in infills:
        p.update()


def strictly_dominate(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Wether vector `a` strictly dominates `b`. `a` and `b` must be 1-dimensional
    and of the same length
    """
    # TODO: raise with message
    assert a.ndim == b.ndim == 1
    assert len(a) == len(b)
    return bool((not (a == b).all()) and (a <= b).all())
