"""A module used to represent stochastic process.

TO DO: longer description?

Attributes
----------
State : TypeVar
    Type of state

Classes
-------
StochasticProcess
"""

from collections import deque
from utils import modify_past
from typing import TypeVar, Generic
import numpy as np

State = TypeVar('State')


class StochasticProcess(Generic[State]):
    """A class used to represent a stochastic process.

    TO DO: longer description?

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, current, next_state, past_min_len=0, past=None):
        """
        Parameters
        ----------
        name : str
            The name of the animal
        current: Any
            Current state of the process
        """

        self._past_min_len = past_min_len
        if not past:
            if not isinstance(past, (list, np.ndarray)):
                past = [past]

            if len(past) >= self._past_min_len:
                self._past = deque(past)
            else:
                raise ValueError(f"Past cannot be less than {past_min_len}")

        self._current = current
        self._next_state = modify_past(next_state, past_min_len)

    @property
    def past(self):
        return self._past

    @past.setter
    def past(self, past):
        if type(past) is not list:
            past = [past]

        if len(past) >= self._past_min_len:
            self._past = past
        else:
            raise ValueError(f"Past cannot be less than {self._past_min_len}")

    @property
    def current(self):
        return self._current

    @property
    def next_state(self):
        return self._next_state

    def __iter__(self):
        return self

    def __next__(self):
        self._current = self._next_state(self._current, self._past)
        return self._current

    def sample(self, n):
        X = np.zeros(n)
        X[0] = self._current
        for i in range(1, n):
            X[i] = self.__next__()
        return X


if __name__ == "__main__":
    sp = StochasticProcess(1, lambda x,y: x+1, past_min_len=0, past=[0,0,0])
    print(sp.past)
    print(next(sp))
    print(sp.past)
