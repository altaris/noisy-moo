"""
Population related methods.
"""
__docformat__ = "google"

from typing import Dict, List, Union

import numpy as np
from pymoo.core.population import Population


def pareto_frontier_mask(arr: np.ndarray) -> np.ndarray:
    """
    Computes the Pareto frontier of a set of points. The array must have an
    `ndim` of 2.

    Returns:
        A mask (array of booleans) on `arr` selecting the points belonging to
        the Pareto frontier. The actual Pareto frontier can be computed with

            pfm = pareto_frontier_mask(arr)
            pf = arr[pfm]

    Warning:
        The direction of the optimum is assumed to towards the "all negative"
        quadrant, i.e. `a` dominates `b` if `a[i] <= b[i]` for all
        `0 <= i < d`.

    Todo:
        * Possibility to specify a direction of optimum;
        * fancy tree-based optimization.
    """
    if not arr.ndim == 2:
        raise ValueError("The input array must be of shape (N, d).")

    d = arr.shape[-1]

    if d == 1:
        mask = np.full(len(arr), False)
        mask[arr.argmin()] = True
        return mask
    if d == 2:
        return pareto_frontier_mask_2d(arr)

    argsort0, mask = np.argsort(arr[:, 0]), np.full(len(arr), True)
    for i, j in enumerate(argsort0):
        if not mask[j]:
            continue
        for k in filter(lambda x: mask[x], argsort0[i + 1 :]):
            # faster than (arr[k] <= arr[j]).any()
            for l in range(1, d):
                if arr[k][l] <= arr[j][l]:
                    mask[k] = True
                    break
            else:
                mask[k] = False
    return mask


def pareto_frontier_mask_2d(arr: np.ndarray) -> np.ndarray:
    """
    Computes the Pareto frontier of a set of 2D points. Faster than
    `pareto_frontier_mask`. Note that `pareto_frontier_mask` will automatically
    call this method whenever possible.

    Returns:
        A mask (array of booleans) on `arr` selecting the points belonging to
        the Pareto frontier. The actual Pareto frontier can be computed with

            pfm = pareto_frontier_mask(arr)
            pf = arr[pfm]

    Warning:
        The direction of the optimum is assumed to be south-west.
    """
    if not arr.ndim == 2 and arr.shape[-1] != 2:
        raise ValueError("The input array must be of shape (N, 2).")
    argsort0, mask = np.argsort(arr[:, 0]), np.full(len(arr), False)
    i = argsort0[0]  # Index of the last Pareto point
    mask[i] = True
    for j in argsort0:
        # Current point is in the south-east quadrant of the last Pareto point
        mask[j] = arr[j][1] <= arr[i][1]
        if mask[j]:
            i = j
    return mask


def population_list_to_dict(
    populations: Union[Population, List[Population]]
) -> Dict[str, np.ndarray]:
    """
    Transforms a list of pymoo Population (or a single Population) into a dict
    containing the following

    * `X`: an `np.array` containing all `X` fields of all individuals across
      all populations;
    * `F`: an `np.array` containing all `F` fields of all individuals across
      all populations;
    * `G`: an `np.array` containing all `G` fields of all individuals across
      all populations;
    * `dF`: an `np.array` containing all `dF` fields of all individuals across
      all populations;
    * `dG`: an `np.array` containing all `dG` fields of all individuals across
      all populations;
    * `ddF`: an `np.array` containing all `ddF` fields of all individuals
      across all populations;
    * `ddG`: an `np.array` containing all `ddG` fields of all individuals
      across all populations;
    * `CV`: an `np.array` containing all `CV` fields of all individuals across
      all populations;
    * `feasible`: an `np.array` containing all `feasible` fields of all
      individuals across all populations;
    * `_batch`: the index of the population the individual belongs to.

    So all `np.arrays` have the same length, which is the total number of
    individual across all populations. Each "row" corresponds to the data
    associated to this individual (`X`, `F`, `G`, `dF`, `dG`, `ddF`, `ddG`,
    `CV`, `feasible`), as well as the population index it belongs to
    (`_batch`).
    """
    if isinstance(populations, Population):
        populations = [populations]
    fields = ["X", "F", "G", "dF", "dG", "ddF", "ddG", "CV", "feasible"]
    data: Dict[str, List[np.ndarray]] = {f: [] for f in fields + ["_batch"]}
    if not populations:
        return {k: np.array([]) for k in data}
    for i, pop in enumerate(populations):
        for f in fields:
            data[f].append(pop.get(f))
        data["_batch"].append(np.full(len(pop), i + 1))
    return {k: np.concatenate(v) for k, v in data.items()}
