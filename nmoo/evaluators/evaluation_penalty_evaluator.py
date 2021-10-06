"""
A custom evaluator that modifies the percieved number of evaluations
(`n_eval`).
"""
__docformat__ = "google"

from pymoo.core.evaluator import Evaluator


class EvaluationPenaltyEvaluator(Evaluator):
    """
    A custom evaluator that modifies the percieved number of evaluations
    (`n_eval`).

    For example, an algorithm using the following evaluator

        EvaluationPenaltyEvaluator("times", 5)

    will percieve every evaluation of the underlying problem as `5`
    evaluations. In other words, if the algorithm's maximum evaluation count is
    `N`, then the problem's `_evaluate` method will be called on at most `N /
    5` values.
    """

    PENALTY_TYPES = ["times"]

    _coefficient: int
    _penalty_type: str

    def __init__(self, penalty_type: str = "times", coefficient: int = 5):
        """
        Args:
            penalty_type (str): the type of evaluation penalty to be applied.
                Currently, must be `times`.
            coefficient (int): coefficient pertaining to the penalty being
                applied.
        """
        super().__init__()
        self._coefficient = coefficient

        if penalty_type not in EvaluationPenaltyEvaluator.PENALTY_TYPES:
            raise ValueError(f"Unknown penalty type '{penalty_type}'")
        self._penalty_type = penalty_type

    def eval(self, *args, **kwargs):
        """
        Calls `Evaluator.eval` and modifies `n_eval` according to the penalty
        parameters.
        """
        result = super().eval(*args, **kwargs)
        if self._penalty_type == "times":
            self.n_eval += (self._coefficient - 1) * result.shape[0]
        return result
