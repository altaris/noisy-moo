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
from typing import Dict, List

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class _Individual(Individual):
    """
    An [pymoo
    `Individual`](https://github.com/anyoptimization/pymoo/blob/master/pymoo/core/individual.py)
    but where attributes `F`, `G`, `dF`, `dG`, `ddF`, `ddG`, and `CV` are
    maximum likelyhood estimates of the true values.
    """

    _samples: Dict[str, np.ndarray]

    n_gen: int
    """Generation at which this individual was born"""

    def __init__(self, n_gen: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._samples = {}
        self.n_gen = n_gen

    def add_sample(self, key: str, value: np.ndarray) -> None:
        """
        Adds a sample and updates the corresponding attribute. For example,

            individual._add_sample("F", np.array([1, 2]))

        adds $(1, 2)$ as a new measurement of `F`. Then, `individual.F` is
        updated to reflect the maximum likelyhood estimate of the true value of
        `F`, which for now is assumed to be the mean. `value` must be a 1D
        array."""
        if key not in self._samples:
            self._samples[key] = value[np.newaxis]
        else:
            self._samples[key] = np.append(self._samples[key], [value], axis=0)
        self.__dict__[key] = self._samples[key].mean(axis=0)

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
            if isinstance(value, np.ndarray):
                self.add_sample(key, value)

    def n_eval(self) -> int:
        """
        The number of times this individual has been sampled. In practice, this
        is the number of samples for the `F` value.
        """
        return self._samples["F"].shape[0] if "F" in self._samples else 0


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
    pop_size: int = 100
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

    _resample_number: int
    """Resample number for methods 2 (denoted by $k$ in Feidlsend's paper)."""

    _resampling_method: str
    """Algorithm used for resampling. See `ARDEMO.__init__`"""

    _rng: np.random.Generator

    _sorter: NonDominatedSorting

    def __init__(
        self,
        resampling_method: str = "fixed",
        convergence_time_window: int = 5,
        demo_crossover_probability: float = 0.3,
        demo_scaling_factor: float = 0.5,
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
        self._sorter = NonDominatedSorting()
        self._resample_number = 1
        self._convergence_time_window = convergence_time_window
        self._demo_crossover_probability = demo_crossover_probability
        self._demo_scaling_factor = demo_scaling_factor

    def _evaluate_individual(self, individual: _Individual) -> None:
        """Evaluates and updates an individual."""
        result = self.evaluator.eval(
            self.problem, individual, skip_already_evaluated=False
        )
        for k in ["F", "G", "dF", "dG", "ddF", "ddG", "CV"]:
            if result.get(k) is not None:
                individual.add_sample(k, result.get(k))

    def _pareto_population_at_gen(self, n_gen: int) -> List[_Individual]:
        """
        Returns the Pareto (aka elite) individuals among all individual born at
        or before the given timestep.
        """
        population = self._population_at_gen(n_gen)
        ranks = self._sorter.do(np.array([p.F for p in population]))
        return [p for i, p in enumerate(population) if i in ranks[0]]

    def _population_at_gen(self, n_gen: int) -> List[_Individual]:
        """
        Returns the population of all individual born at or before the given
        timestep.
        """
        return list(filter(lambda p: p.n_gen <= n_gen, self.pop))

    def _reevaluate_individual_with_fewest_resamples(
        self, population: List[_Individual]
    ) -> None:
        """
        Randomly choose an individual in the given population that has the
        fewest number of resamples, and reevaluates it.
        """
        counts = np.array([p.n_eval() for p in population])
        index = self._rng.choice(np.where(counts == counts.min())[0])
        self._evaluate_individual(population[index])

    def _resampling_elite(self) -> None:
        """
        Resample counts of elite members increases over time. Corresponds to
        algorithm 4 in Fieldsend's paper.
        """
        raise NotImplementedError()

    def _resampling_fixed(self) -> None:
        """
        Resampling rate is fixed. Corresponds to algorithm 1 in Fieldsend's
        paper.
        """
        self._reevaluate_individual_with_fewest_resamples(
            self._pareto_population_at_gen(self.n_gen)
        )

    def _resampling_min_on_conv(self) -> None:
        """
        Resampling rate *of elite members* may increase based on a convergence
        assessment that uses the $\\varepsilon +$ indicator. Corresponds to
        algorithm 3 in Fieldsend's paper.
        """
        raise NotImplementedError()

    def _resampling_rate_on_conv(self) -> None:
        """
        Resampling rate may increase based on a convergence assessment that
        uses the $\\varepsilon +$ indicator. Corresponds to algorithm 2 in
        Fieldsend's paper.
        """
        raise NotImplementedError()

    def _truncate_population(self) -> None:
        """
        Keeps the `self.pop_size` first individuals as sorted by non-dominated
        sorting.
        """
        if len(self.pop) <= self.pop_size:
            return
        ranks = self._sorter.do(np.array([p.F for p in self.pop]))
        ranks = np.concatenate(ranks)
        ranks = ranks[: self.pop_size]
        self.pop = self.pop[ranks]

    # pymoo overrides =========================================================

    def _advance(self, infills=None, **kwargs):
        """
        Called after the infills (aka new individuals) have been evaluated.
        """
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
        else:
            method()
        for p in self.pop:
            p.update()  # Update maximum-likelyhood estimates
        self._truncate_population()

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
        individual = _Individual(n_gen=self.n_gen, X=nX)
        # self._evaluate_individual(individual)
        self.pop = Population.merge(self.pop, individual)
        return individual

    def _initialize_advance(self, infills=None, **kwargs):
        """Only called after the first generation has been evaluated"""

    def _initialize_infill(self):
        """
        Only called to get the first generation. Subsequent generations are
        generated by calling `_infill`.
        """
        samples = FloatRandomSampling().do(
            self.problem, n_samples=self.pop_size
        )
        population = [_Individual(n_gen=self.n_gen, X=p.X) for p in samples]
        return Population.create(*population)

    def _setup(self, problem, **kwargs):
        """Called before an algorithm starts running on a problem"""
        self._rng = np.random.default_rng(kwargs.get("seed"))
