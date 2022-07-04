"""
A custom evaluator that modifies the perceived number of evaluations
(`n_eval`).
"""
__docformat__ = "google"

from pymoo.core.evaluator import Evaluator


class PenalizedEvaluator(Evaluator):
    """
    A custom evaluator that multiplies the perceived number of evaluations
    (`n_eval`).

    For example, an pymoo algorithm using the following evaluator

        PenalizedEvaluator(5)

    will perceive every evaluation of the underlying problem as `5`
    evaluations. In other words, if the algorithm's maximum evaluation count is
    `N`, then the problem's `_evaluate` method will be called on at most `N /
    5` values.
    """

    _multiplier: int

    def __init__(self, multiplier: int = 5):
        """
        Args:
            multiplier (int): multiplier pertaining to the penalty being
                applied.
        """
        super().__init__()
        self._multiplier = multiplier

    def eval(self, *args, **kwargs):  # pylint: disable=signature-differs
        """
        Calls `Evaluator.eval` and modifies `n_eval` according to the penalty
        parameters.
        """
        result = super().eval(*args, **kwargs)
        self.n_eval += (self._multiplier - 1) * result.shape[0]
        return result
