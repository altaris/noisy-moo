"""
DEMO (Differential Evolution for Multiobjective Optimization, `DE/rand/1/bin`
variant) [^demo] with an accumulative resampling scheme to deal with noise,
following [^elite].

Todo:
    I'm sure pymoo already offers some of the functionalities that are used in
    this module. If only they documented their code...

[^demo]: Robič, T., Filipič, B. (2005). DEMO: Differential Evolution for
    Multiobjective Optimization. In: Coello Coello, C.A., Hernández Aguirre,
    A., Zitzler, E. (eds) Evolutionary Multi-Criterion Optimization. EMO 2005.
    Lecture Notes in Computer Science, vol 3410. Springer, Berlin, Heidelberg.
    https://doi.org/10.1007/978-3-540-31880-4_36
[^elite]: Fieldsend, J.E. (2015). Elite Accumulative Sampling Strategies for
    Noisy Multi-objective Optimisation. In: Gaspar-Cunha, A., Henggeler
    Antunes, C., Coello, C. (eds) Evolutionary Multi-Criterion Optimization.
    EMO 2015. Lecture Notes in Computer Science(), vol 9019. Springer, Cham.
    https://doi.org/10.1007/978-3-319-15892-1_12
"""
__docformat__ = "google"

import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union
from itertools import product

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
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
class ARDEMO(Algorithm):
    """
    DEMO (differential evolution for MOO problems, `DE/rand/1/bin` variant)
    with an accumulative resampling scheme to deal with noise.
    """

    SUPPORTED_RESAMPLING_METHODS = [
        "elite",
        "fixed",
        "min_on_conv",
        "rate_on_conv",
    ]

    n_gen: int
    pop: List[_Individual]  # In reality it'll be a Population

    _convergence_time_window: int
    """
    Convergence time window for method 2 and 3, (denoted by $m$ in Feidlsend's
    paper)
    """

    _demo_crossover_probability: float
    """Differential evolution parameter"""

    _demo_scaling_factor: float
    """Differential evolution parameter"""

    _max_population_size: int

    # _nds_cache: Deque[Tuple[int, int, List[np.ndarray]]]

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
        demo_crossover_probability: float = 0.3,
        demo_scaling_factor: float = 0.5,
        max_population_size: int = 100,
        **kwargs,
    ):
        """
        Args:
            convergence_time_window (int): Convergence time window for method 2
                and 3, (denoted by $m$ in [^elite])
            demo_crossover_probability (float): DEMO parameter, see [^demo].
                Defaults to $0.3$ as recommended by [^demo], section 4. Note
                that [^elite] sets it to $0.9$.
            demo_scaling_factor (float): DEMO parameter, see [^demo]. Defaults
                to $0.5$ as recommended by [^demo]
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

        [^demo]: Robič, T., Filipič, B. (2005). DEMO: Differential Evolution
            for Multiobjective Optimization. In: Coello Coello, C.A., Hernández
            Aguirre, A., Zitzler, E. (eds) Evolutionary Multi-Criterion
            Optimization. EMO 2005. Lecture Notes in Computer Science, vol
            3410. Springer, Berlin, Heidelberg.
            https://doi.org/10.1007/978-3-540-31880-4_36
        [^elite]: Fieldsend, J.E. (2015). Elite Accumulative Sampling
            Strategies for Noisy Multi-objective Optimisation. In:
            Gaspar-Cunha, A., Henggeler Antunes, C., Coello, C. (eds)
            Evolutionary Multi-Criterion Optimization. EMO 2015. Lecture Notes
            in Computer Science(), vol 9019. Springer, Cham.
            https://doi.org/10.1007/978-3-319-15892-1_12
        """
        super().__init__(**kwargs)
        if resampling_method not in self.SUPPORTED_RESAMPLING_METHODS:
            raise ValueError(
                "Invalid resampling method. Supported methods are "
                + ", ".join(self.SUPPORTED_RESAMPLING_METHODS)
            )
        self._resampling_method = resampling_method
        self._rng = np.random.default_rng()
        self._convergence_time_window = convergence_time_window
        self._demo_crossover_probability = demo_crossover_probability
        self._demo_scaling_factor = demo_scaling_factor
        self._max_population_size = max_population_size
        # self._nds_cache = deque(maxlen=5)

    def _evaluate_individual(self, individual: _Individual) -> None:
        """Evaluates and updates an individual."""
        if self.termination.has_terminated(self):
            raise TerminationCriterionMet()
        self.evaluator.eval(
            self.problem, individual, skip_already_evaluated=False
        )
        individual.update(self.n_gen)
        # Little hack so that WrappedProblem's see this evaluation as part of
        # the same batch as the infills of this generation
        problem = self.problem
        while isinstance(problem, WrappedProblem):
            problem._current_history_batch = self.n_gen
            problem._history["_batch"][-1] = self.n_gen
            problem = problem._problem

    def _non_dominated_sort(self, generation: int) -> List[np.ndarray]:
        """Cached non-dominated sort of the current population"""
        population = self._population_at_gen(generation)
        # for (g, s, rs) in self._nds_cache:
        #     if g == generation and s == len(population):
        #         return rs
        sorter = NonDominatedSorting(method="efficient_non_dominated_sort")
        ranks = sorter.do(
            np.array([p.get_estimate("F", generation) for p in population])
        )
        # self._nds_cache.append((generation, len(population), ranks))
        return ranks

    def _pareto_population(
        self, generation: Optional[int] = None
    ) -> List[_Individual]:
        """
        Returns the Pareto (aka elite) individuals among all individual born at
        or before the given generation. By default, the current generation is
        used.
        """
        if generation is None:
            generation = self.n_gen
        population = self._population_at_gen(generation)
        ranks = self._non_dominated_sort(generation)
        return [p for i, p in enumerate(population) if i in ranks[0]]

    def _population_at_gen(self, generation: int) -> List[_Individual]:
        """
        Returns the population of all individual born at or before the given
        timestep.
        """
        return list(
            filter(lambda p: p.generation_born <= generation, self.pop)
        )

    def _reevaluate_individual_with_fewest_resamples(
        self, population: List[_Individual]
    ) -> None:
        """
        Randomly choose an individual in the given population that has the
        fewest number of resamples, and reevaluates it. Returns the individual
        in question.
        """
        counts = np.array([p.n_eval() for p in population])
        index = self._rng.choice(np.where(counts == counts.min())[0])
        self._evaluate_individual(population[index])

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
            return np.mean(
                [p.n_eval(self.n_gen) for p in self._pareto_population()]
            )

        self._reevaluate_individual_with_fewest_resamples(
            self._pareto_population()
        )
        alpha = sum(
            [m * s for (m, s) in self._resampling_elite_cache.values()]
        ) / sum([s for (_, s) in self._resampling_elite_cache.values()])
        while _mean_n_eval_pareto() <= alpha:
            self._reevaluate_individual_with_fewest_resamples(
                self._pareto_population()
            )

    def _resampling_fixed(self) -> None:
        """
        Resampling rate is fixed. Corresponds to algorithm 1 in Fieldsend's
        paper.
        """
        self._reevaluate_individual_with_fewest_resamples(
            self._pareto_population()
        )

    def _resampling_min_on_conv(self) -> None:
        """
        Resampling rate *of elite members* may increase based on a convergence
        assessment that uses the $\\varepsilon +$ indicator. Corresponds to
        algorithm 3 in Fieldsend's paper.
        """
        # TODO: Deduplicate code
        # Generation m+1, 2m+1, 3m+1, etc. where
        # m = self._convergence_time_window
        if self.n_gen > 1 and self.n_gen % self._convergence_time_window == 1:
            p1 = self._pareto_population()
            p2 = self._pareto_population(
                self.n_gen - self._convergence_time_window
            )
            a1 = extended_epsilon_plus_indicator(p1, p2)
            a2 = extended_epsilon_plus_indicator(p2, p1)
            if a1 > a2:
                self._resample_number += 1
        self._reevaluate_individual_with_fewest_resamples(
            self._pareto_population()
        )
        while True:
            population = self._pareto_population()
            if min([p.n_eval() for p in population]) >= self._resample_number:
                break
            self._reevaluate_individual_with_fewest_resamples(population)

    def _resampling_rate_on_conv(self) -> None:
        """
        Resampling rate may increase based on a convergence assessment that
        uses the $\\varepsilon +$ indicator. Corresponds to algorithm 2 in
        Fieldsend's paper.
        """
        # Generation m+1, 2m+1, 3m+1, etc. where
        # m = self._convergence_time_window
        if self.n_gen > 1 and self.n_gen % self._convergence_time_window == 1:
            p1 = self._pareto_population()
            p2 = self._pareto_population(
                self.n_gen - self._convergence_time_window
            )
            a1 = extended_epsilon_plus_indicator(p1, p2)
            a2 = extended_epsilon_plus_indicator(p2, p1)
            if a1 > a2:
                self._resample_number += 1
        for _ in range(self._resample_number):
            self._reevaluate_individual_with_fewest_resamples(
                self._pareto_population(),
            )

    def _truncate_population(self) -> None:
        """
        Keeps the `self._max_population_size` first individuals as sorted by non-dominated
        sorting.
        """
        if len(self.pop) <= self._max_population_size:
            return
        ranks = np.concatenate(self._non_dominated_sort(self.n_gen))
        ranks = ranks[: self._max_population_size]
        self.pop = self.pop[ranks]

    # pymoo overrides =========================================================

    def _advance(
        self,
        infills: Optional[Union[_Individual, Iterable[_Individual]]] = None,
        **kwargs,
    ):
        """
        Called after the infills (aka new individuals) have been evaluated.
        """
        if infills is None:
            raise ValueError(
                "ARDEMO's _advance needs the current iteration's infills"
            )
        if isinstance(infills, _Individual):
            infills = [infills]
        for p in infills:
            p.update(self.n_gen)
        self._truncate_population()

        # Caching for _resampling_elite
        arr = [p.n_eval(self.n_gen) for p in self._pareto_population()]
        self._resampling_elite_cache[self.n_gen] = (
            np.mean(arr),
            len(arr),
        )

        method = {
            "fixed": self._resampling_fixed,
            "rate_on_conv": self._resampling_rate_on_conv,
            "min_on_conv": self._resampling_min_on_conv,
            "elite": self._resampling_elite,
        }.get(self._resampling_method)
        if method is None:
            logging.warning(
                "Invalid resampling method %s", self._resampling_method
            )
            return
        try:
            method()
        except TerminationCriterionMet:
            return

    def _finalize(self):
        """
        Called in `Algorithm.advance` once the termination criterion has been
        met
        """

    def _infill(self) -> _Individual:
        """
        Generate new individuals for the next generation. Uses 3-way mating
        (kinky) and binary crosseover. The new individual is added to the
        algorithm's population but is not evaluated.
        """
        # TODO: Do a, b, c need to be distinct?
        p, a, b, c = self._rng.choice(self.pop, 4, replace=False)
        mask = self._rng.binomial(
            1, self._demo_crossover_probability, a.X.shape[0]
        )
        dX = a.X + self._demo_scaling_factor * (b.X - c.X)
        nX = mask * dX + (1 - mask) * p.X
        # n_gen is only updated in Algorithm.advance, which is called AFTER
        # _infill !
        individual = _Individual(self.n_gen + 1, X=nX)
        self.pop = Population.merge(self.pop, individual)
        return individual

    def _initialize_advance(self, infills=None, **kwargs):
        """Only called after the first generation has been evaluated"""
        self._advance(infills)

    def _initialize_infill(self):
        """
        Only called to get the first generation. Subsequent generations are
        generated by calling `_infill`.
        """
        samples = FloatRandomSampling().do(
            self.problem, n_samples=self._max_population_size
        )
        population = [_Individual(1, X=p.X) for p in samples]
        return Population.create(*population)

    def _setup(self, problem, **kwargs):
        """Called before an algorithm starts running on a problem"""
        self._rng = np.random.default_rng(kwargs.get("seed"))
        self._resampling_elite_cache = {}
        self._resample_number = 1


def extended_epsilon_plus_indicator(
    population_1: Iterable[Individual], population_2: Iterable[Individual]
) -> float:
    """
    Extended $\\varepsilon+$ indicator:
    $$
        I (A, B) = \\max_{a, b, i} F_i (b) - F_i (a)
    $$
    where $A$ and $B$ are two populations, and where $F_i$ is the $i$-th
    objective.
    """
    arr = map(
        lambda t: np.max(t[1].F - t[0].F), product(population_1, population_2)
    )
    return max(list(arr))
