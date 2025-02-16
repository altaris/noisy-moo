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
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.replacement import ImprovementReplacement

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

    _sample_generations: List[int]
    """
    Generation number at which samples have been taken, i.e.
    `_sample_generations[i]` is the generation at which `_samples[*][i]` has
    been measured.
    """

    _samples: Dict[str, np.ndarray]

    generation_born: int
    """Generation at which this individual was born"""

    def __init__(self, generation_born: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sample_generations = []
        self._samples = {}
        self.generation_born = generation_born

    def get_estimate(
        self, key: str, at_generation: Optional[int] = None
    ) -> np.ndarray:
        """
        Return the maximum likelyhood estimate for for value of `key`, using
        the samples that were made up to a given generation. If that generation
        limit is left to `None`, all samples are used. In this case, using
        direct attribute access produces the same result, e.g.
            ind.get_estimate("F")
            # is equivalent to
            ind.F
        (assuming `ind` has been correctly `update`d).
        """
        samples = self._samples[key]
        if at_generation is not None:
            mask = np.array(self._sample_generations) <= at_generation
            samples = samples[mask]
        return samples.mean(axis=0)

    def update(self, current_generation: int) -> None:
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
        self._sample_generations.append(current_generation)

    def n_eval(self, up_to_generation: Optional[int] = None) -> int:
        """
        The number of times this individual has been sampled up to a given
        generation. If left to `None`, all generations are considered.
        """
        if up_to_generation is None:
            return len(self._sample_generations)
        return len(
            list(
                filter(
                    lambda x: x <= up_to_generation,  # type: ignore
                    self._sample_generations,
                )
            )
        )


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
        "min_on_conv",
        "rate_on_conv",
    ]

    # pymoo inherited properties
    pop: List[_Individual]
    n_gen: int

    _convergence_time_window: int
    """
    Convergence time window for method 2 and 3, (denoted by $m$ in Feidlsend's
    paper)
    """

    _demo_crossover_probability: float
    """Differential evolution parameter"""

    _demo_scaling_factor: float
    """Differential evolution parameter"""

    _resampling_elite_cache: Dict[int, Tuple[int, int]] = {}
    """
    At key `t`, contains the average number of resamplings of Pareto
    individuals at generation `t`, and the size of the Pareto population at
    generation `t`. Used as caching for `_resampling_elite`.
    """

    _resample_number: int = 1
    """Resample number for methods 2 (denoted by $k$ in Feidlsend's paper)."""

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
                * `rate_on_conv`: resampling rate may increase based on a
                  convergence assessment that uses the $\\varepsilon +$
                  indicator; corresponds to algorithm 2 in [^elite];
                * `min_on_conv`: resampling rate *of elite members* may
                  increase based on a convergence assessment that uses the
                  $\\varepsilon +$ indicator; corresponds to algorithm 3 in
                  [^elite];
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
        """Evaluates an `_Individual` and increments its `n_eval` counter"""
        if self.n_gen >= 2 and self.termination.has_terminated(self):
            raise TerminationCriterionMet()
        self.evaluator.eval(
            self.problem, individual, skip_already_evaluated=False
        )
        individual.n_eval += 1
        # Little hack so that WrappedProblem's see this evaluation as part of
        # the same batch as the infills of this generation
        problem = self.problem
        while isinstance(problem, WrappedProblem):
            problem._current_history_batch = self.n_gen
            problem._history["_batch"][-1] = self.n_gen
            problem = problem._problem

    def _pareto_population(self) -> List[_Individual]:
        """
        Returns the Pareto (aka elite) individuals. Unlike
        `pymoo.util.optimum.filter_optimum`, returns a list of `_Individual`s
        rather than a list of `Individual`s reconstructed from
        `self.pop.get("F")`.
        """
        if len(self.pop) == 0:
            return []
        sorter = NonDominatedSorting(method="efficient_non_dominated_sort")
        ranks = sorter.do(np.array([p.F for p in self.pop if p.feasible]))
        return [p for i, p in enumerate(self.pop) if i in ranks[0]]

    def _reevaluate_pareto_individual_with_fewest_evals(self) -> None:
        """
        Randomly choose a Pareto `_Individual` that has the fewest number of
        resamples, and reevaluates it.
        """
        pareto_population = self._pareto_population()
        counts = np.array([p.n_eval for p in pareto_population])
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
            return np.mean([p.n_eval for p in self._pareto_population()])

        pareto_population = self._pareto_population()
        arr = [p.n_eval for p in pareto_population]
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
        # TODO: Deduplicate code
        # Generation m+1, 2m+1, 3m+1, etc. where
        # m = self._convergence_time_window
        # if self.n_gen > 1 and self.n_gen % self._convergence_time_window == 1:
        #     p1 = self._pareto_population()
        #     p2 = self._pareto_population(
        #         self.n_gen - self._convergence_time_window
        #     )
        #     # It could be that no one from n_gen - _convergence_time_window is
        #     # still alive...
        #     if len(p2) != 0:
        #         a1 = extended_epsilon_plus_indicator(p1, p2)
        #         a2 = extended_epsilon_plus_indicator(p2, p1)
        #         if a1 > a2:
        #             self._resample_number += 1
        # self._reevaluate_pareto_individual_with_fewest_evals(
        # )
        # while True:
        #     population = self._pareto_population()
        #     if min([p.n_eval for p in population]) >= self._resample_number:
        #         break
        #     self._reevaluate_pareto_individual_with_fewest_evals()

    def _resampling_rate_on_conv(self) -> None:
        """
        Resampling rate may increase based on a convergence assessment that
        uses the $\\varepsilon +$ indicator. Corresponds to algorithm 2 in
        Fieldsend's paper.
        """
        raise NotImplementedError
        # Generation m+1, 2m+1, 3m+1, etc. where
        # m = self._convergence_time_window
        # if self.n_gen > 1 and self.n_gen % self._convergence_time_window == 1:
        #     p1 = self._pareto_population()
        #     p2 = self._pareto_population(
        #         self.n_gen - self._convergence_time_window
        #     )
        #     # It could be that no one from n_gen - _convergence_time_window is
        #     # still alive...
        #     if len(p2) != 0:
        #         a1 = extended_epsilon_plus_indicator(p1, p2)
        #         a2 = extended_epsilon_plus_indicator(p2, p1)
        #         if a1 > a2:
        #             self._resample_number += 1
        # for _ in range(self._resample_number):
        #     self._reevaluate_pareto_individual_with_fewest_evals(
        #     )

    # pymoo overrides =========================================================

    def _advance(
        self,
        infills: Optional[Union[_Individual, List[_Individual]]] = None,
        **kwargs,
    ) -> None:
        """
        Called after the infills (aka new individuals) have been evaluated.
        """
        _update_infills(infills)
        q = infills[0]
        pi = q.data["parent"]
        if dominate(q.F, self.pop[pi].F):
            self.pop[pi] = q
        self._do_resampling()

    def _infill(self) -> Population:
        """
        Generate new individuals for the next generation. Calls `DE._infill`
        and converts the results to `_Individual`s.
        """
        pi, ai, bi, ci = np.random.default_rng().choice(
            len(self.pop), 4, replace=False
        )
        p, a, b, c = (
            Individual(self.pop[pi].X),
            self.pop[ai].X,
            self.pop[bi].X,
            self.pop[ci].X,
        )
        p.data["parent"] = pi
        p.__dict__["n_eval"] = 0
        for i in range(len(p.X)):  # pylint: disable=consider-using-enumerate
            if np.random.rand() < self.mating.crossover.CR:
                p.X[i] = a[i] + self.mating.crossover.F * (b[i] - c[i])
        return Population.create(p)

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        """Only called after the first generation has been evaluated"""
        _update_infills(infills)
        self._do_resampling()

    def _initialize_infill(self) -> Population:
        """
        Only called to get the first generation. Subsequent generations are
        generated by calling `_infill`.
        """
        population = super()._initialize_infill()
        for p in population:
            p.__dict__["n_eval"] = 0
        return population
        # return Population.create(*[_Individual(p.X) for p in population])

    def _setup(self, problem, **kwargs) -> None:
        """Called before an algorithm starts running on a problem"""
        super()._setup(problem, **kwargs)
        self._rng = np.random.default_rng(kwargs.get("seed"))
        self._resampling_elite_cache = {}
        self._resample_number = 1


def _update_infills(
    infills: Optional[Union[_Individual, List[_Individual]]]
) -> None:
    """
    Takes evaluated infills of type `_Individual` and increments their
    `n_eval` counter.
    """
    if infills is None:
        raise ValueError(
            "ARDEMO's _advance needs the current iteration's infills"
        )
    if isinstance(infills, Individual):
        infills = [infills]
    for p in infills:
        p.n_eval += 1


# def extended_epsilon_plus_indicator(
#     population_1: List[Individual], population_2: List[Individual]
# ) -> float:
#     """
#     Extended $\\varepsilon+$ indicator:
#     $$
#         I (A, B) = \\max_{a, b, i} F_i (b) - F_i (a)
#     $$
#     where $A$ and $B$ are two populations, and where $F_i$ is the $i$-th
#     objective.
#     """
#     arr = map(
#         lambda t: np.max(t[1].F - t[0].F), product(population_1, population_2)
#     )
#     return max(list(arr))


def dominate(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Wether vector `a` (strictly) dominates `b`. `a` and `b` must be
    1-dimensional and of the same length
    """
    assert a.ndim == b.ndim == 1
    assert len(a) == len(b)
    if (a == b).all():
        return False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
    return True
