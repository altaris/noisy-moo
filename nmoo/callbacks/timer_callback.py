"""
Timer callback
"""
__docformat__ = "google"

from typing import List

from pymoo.core.callback import Callback

import pandas as pd


class TimerCallback(Callback):
    """
    A simple callback that register a timedelta (from the time it was
    instanciated) everytime it is notified.
    """

    _deltas: List[pd.Timedelta]
    """Timedeltas"""

    _initial_time: pd.Timestamp
    """Timestamp of when this instance has been created"""

    def __init__(self):
        super().__init__()
        self._deltas = []
        self._initial_time = pd.Timestamp.now()

    # pylint: disable=unused-argument
    def notify(self, algorithm, **kwargs):
        """
        See the `pymoo documentation
        <https://pymoo.org/interface/callback.html>`_
        """
        self._deltas.append(pd.Timestamp.now() - self._initial_time)
