"""A module used to represent markov chain.

TO DO: longer description?

Attributes
----------
State : TypeVar
    A generic type of state, can be int, List[int], np.array, ...

Classes
-------
MarkovChain
HomogeneousMarkovChain
"""

from stochastic_process import StochasticProcess, State
import numpy as np
from typing import Sequence, Callable, Optional


class MarkovChain(StochasticProcess[State]):
    """A class used to represent a markov chain.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the process
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
        Function of current (and optionally past) that returns next state:
        (current: State, past: Sequence[State] = None) -> State
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

    def __init__(self, current: State, next_state: Callable[[State], State],
                 past_max_len: Optional[int] = 0, past: Sequence[State] = None) -> None:
        """
        Parameters
        ----------
        current: State
            Current state of the process
        next_state: Callable[[State], State]
            Function of current (and optionally past) that returns next state:
            (current: State, past: Sequence[State] = None) -> State
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track. If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states
        """

        def mc_next_state(current_: State, _: Sequence[State] = None) -> State:
            return next_state(current_)

        super().__init__(current, next_state=mc_next_state, past_min_len=0, past_max_len=past_max_len, past=past)

class HomogeneousMarkovChain(MarkovChain):
    def __init__(self, mu, P, past=[None]):
        self._mu = mu
        self._P = P
        self._states = np.arange(len(mu))

        def next_state(curr):
            return np.random.choice(self._states, size=1, p=P[curr, :])

        super().__init__(current=np.random.choice(self._states, p=mu), next_state=next_state, past=past)

    @property
    def states(self):
        return self._states

    @property
    def mu(self):
        return self._mu

    @property
    def P(self):
        return self._P


def Ehrenfest(N): # TO DO rethink  N
    P = np.zeros((N-1, N-1))
    P[0, 1] = 1
    P[-1, -2] = 1
    for i in range(1, N-2):
        P[i, i-1] = i/N
        P[i, i+1] = (N-i)/N
    return P

def SymmetricWalk(N):
    P = np.zeros((N-1, N-1))
    P[0, 1] = 1
    P[-1, -2] = 1
    for i in range(1, N - 2):
        P[i, i - 1] = 0.5
        P[i, i + 1] = 0.5
    return P