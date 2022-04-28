"""A module used to represent stochastic process.

TO DO: longer description?

Attributes
----------
State : TypeVar
    A generic type of state, can be int, List[int], np.array, ...

Classes
-------
StochasticProcess
"""

from collections import deque
from utils import modify_past
from typing import TypeVar, Generic, Sequence, Callable, Optional, Iterable
import numpy as np

State = TypeVar('State')


class StochasticProcess(Generic[State]):
    """A class used to represent a stochastic process.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the process
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
        Function of current (and optionally past) that returns next state:
        (current: State, past: Sequence[State] = None) -> State
    _past_min_len: int, optional
        Minimal length of the past sequence that is needed for obtaining next state. If equal to 0, then no past is
        needed to generate next state.
    _past_max_len: int, optional
        Maximal length of the past sequence that one wants to track. If equal to 0, then no past is tracked.
    _past: Iterable[State], optional
        Sequence of past states

    Methods
    -------
    sample(n: int) -> Sequence[State]:
        Get sample of length n of the stochastic process
    move(n_steps: int) -> State
        Move the stochastic process by n steps
    """

    def __init__(self, current: State, next_state: Callable[[State, Optional[Sequence[State]]], State],
                 past_min_len: Optional[int] = 0, past_max_len: Optional[int] = 0, past: Sequence[State] = None) -> None:
        """
        Parameters
        ----------
        current: State
            Current state of the process
        next_state: Callable[[State, Optional[Sequence[State]]], State]
            Function of current (and optionally past) that returns next state:
            (current: State, past: Sequence[State] = None) -> State
        past_min_len: int, optional
            Minimal length of the past sequence that is needed for obtaining next state. If equal to 0, then no past is
            needed to generate next state.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track. If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states
        """

        self._past_min_len = past_min_len
        if past is not None:
            if len(past) >= self._past_min_len:
                self._past = deque(past)
            else:
                raise ValueError(f"Past cannot be less than {past_min_len}.")

        if past_min_len <= past_max_len:
            self._past_max_len = past_max_len
        else:
            raise ValueError('Past min length has to be lower than maximum.')

        self._step = 0
        self._current = current
        self._next_state = modify_past(next_state, past_min_len, past_max_len)

    @property
    def current(self) -> State:
        """Get current"""
        return self._current

    @property
    def next_state(self) -> Callable[[State, Optional[Sequence[State]]], State]:
        """Get next_state"""
        return self._next_state

    @property
    def past_min_len(self) -> int:
        """Get past_min_len"""
        return self._past_min_len

    @property
    def past_max_len(self) -> int:
        """Get past_max_len"""
        return self._past_max_len

    @property
    def past(self) -> Sequence[State]:
        """Get past TO DO NULL"""
        return self._past

    @past.setter
    def past(self, past: Sequence[State]) -> None:
        """Set past"""
        if len(past) >= self._past_min_len:
            self._past = past
        else:
            raise ValueError(f"Past cannot be less than {self._past_min_len}")

    def __iter__(self) -> Iterable:
        """Get iterator"""
        return self

    def __next__(self) -> State:
        """Move the process and return new current state"""
        self._current = self._next_state(self._current, self._past)
        return self._current

    def sample(self, n: int) -> Sequence[State]:
        """Get sample of length n of the stochastic process

        Parameters
        ----------
        n: int
            Length of the sample

        Returns
        -------
        X: Sequence[State]
            Sample of the stochastic process
        """

        X = [None] * n
        X[0] = self._current
        for i in range(1, n):
            X[i] = self.__next__()
        return X

    def move(self, n_steps: int) -> State:
        """Move the stochastic process by n steps

        Parameters
        ----------
        n_steps: int
            Number of steps to move stochastic process

        Returns
        -------
        _current: State
            Current state of the stochastic process
        """

        for _ in range(n_steps):
            self.__next__()
        return self._current


if __name__ == "__main__":
    sp = StochasticProcess(np.array([0,0]), lambda x, y: x+y[0], past=np.array([[1, 1], [2, 2]]), past_max_len=3, past_min_len=1)
    print(sp.sample(100))
    print(sp.past)
