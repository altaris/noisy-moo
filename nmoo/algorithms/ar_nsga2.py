"""
Accumulative resampling algorithm wrapper for noisy single/multi-objective
problems.
"""
__docformat__ = "google"

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from nmoo.wrapped_problem import WrappedProblem


class TerminationCriterionMet(Exception):
    """
    Raised by `ARNSGA2._evaluate_individual` if the termination criterion has
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
class ARNSGA2(NSGA2):
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
    ]

    # pymoo inherited properties
    pop: List[_Individual]
    n_gen: int

    _pareto_population_history: List[Tuple[int, int]]
    """
    At key `i`, contains a tuple of two numbers:
    * the average number of times a Pareto individual at generation i (an
        individual that was Pareto at timestep i) has been evaluated (at and
        before timestep i);
    * the number of Pareto individuals at timestep i (e.g. number of
        individuals that were Pareto at timestep i).
    Recall that a timestep is everytime an individual is evaluated. This dict is used in `ARNSGA2._resampling_elite` to compute the value of
    $$\\alpha$$ (see Fieldsend's paper, algorithm 4).
    """

    _resampling_method: str
    """Algorithm used for resampling. See `ARNSGA2.__init__`"""

    _rng: np.random.Generator

    def __init__(
        self,
        resampling_method: str = "fixed",
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

    def _do_resampling(self) -> None:
        """
        Dispatches to `_resampling_elite`, `_resampling_fixed`,
        `_resampling_min_on_conv` or `_resampling_rate_on_conv` depending on
        the value of `_resampling_method`. Also catches
        `TerminationCriterionMet` exceptions.
        """
        method = {
            "fixed": self._resampling_fixed,
            "elite": self._resampling_elite,
        }.get(self._resampling_method)
        if method is None:
            raise ValueError(
                f"Invalid resampling method '{self._resampling_method}'"
            )
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
        individual.update()
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
        # if self.opt:
        #     return self.opt
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

        self._reevaluate_pareto_individual_with_fewest_evals()
        # Recall that if (m, s) = _pareto_population_history[i], then m was the
        # average number of times a Pareto individual at timestep i (an
        # individual that was Pareto at timestep i) has been evaluated (at and
        # before timestep i), while s was the number of Pareto individuals at
        # timestep i (e.g. number of individuals that were Pareto at timestep
        # i). In particular, m * s is the total number of times Pareto
        # individuals of timestep i have been evaluated so far (i.e. at and
        # before timestep i).
        a = sum([m * s for (m, s) in self._pareto_population_history])
        b = sum([s for (_, s) in self._pareto_population_history])
        alpha = a / b
        while _mean_n_eval_pareto() <= alpha:
            self._reevaluate_pareto_individual_with_fewest_evals()
            pareto_population = self._pareto_population()
            arr = [p.n_eval() for p in pareto_population]
            self._pareto_population_history.append(
                (
                    np.mean(arr),
                    len(arr),
                )
            )

    def _resampling_fixed(self) -> None:
        """
        Resampling rate is fixed. Corresponds to algorithm 1 in Fieldsend's
        paper.
        """
        self._reevaluate_pareto_individual_with_fewest_evals()

    # pymoo overrides =========================================================

    def _advance(
        self,
        infills: Optional[Union[_Individual, List[_Individual]]] = None,
        **kwargs,
    ) -> None:
        """
        Called after the infills (aka new individuals) have been evaluated.
        """
        if infills is None:
            raise ValueError(
                "ARNSGA2's _advance needs the current iteration's infills"
            )
        _update_infills(infills)
        super()._advance(infills, **kwargs)
        self._do_resampling()

    def _infill(self) -> Population:
        """
        Generate new individuals for the next generation. Calls `NSGA2._infill`
        and converts the results to `_Individual`s.
        """
        population = super()._infill()
        return Population.create(*[_Individual(p.X) for p in population])

    def _initialize_advance(
        self,
        infills: Optional[Union[_Individual, List[_Individual]]] = None,
        **kwargs,
    ) -> None:
        """Only called after the first generation has been evaluated"""
        if infills is None:
            raise ValueError(
                "ARNSGA2's _advance needs the current iteration's infills"
            )
        _update_infills(infills)
        super()._initialize_advance(infills, **kwargs)
        self._do_resampling()

    def _initialize_infill(self) -> Population:
        """
        Only called to get the first generation. Subsequent generations are
        generated by calling `_infill`.
        """
        population = super()._initialize_infill()
        return Population.create(*[_Individual(p.X) for p in population])

    def _setup(self, problem, **kwargs) -> None:
        """Called before an algorithm starts running on a problem"""
        super()._setup(problem, **kwargs)
        self._rng = np.random.default_rng(kwargs.get("seed"))
        self._pareto_population_history = []


def _update_infills(infills: Union[_Individual, List[_Individual]]) -> None:
    """
    Takes evaluated infills of type `_Individual` and increments their
    `n_eval` counter.
    """
    if isinstance(infills, _Individual):
        infills = [infills]
    for p in infills:
        p.update()
