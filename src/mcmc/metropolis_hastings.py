"""A module used to sample using Metropolis-Hastings algorithm.

TO DO: longer description?

Attributes
----------
State : TypeVar
    A generic type of state, can be int, List[int], np.array, ...

Classes
-------
MetropolisHastings
"""

from typing import Sequence, Callable, Any
from nptyping import NDArray, Shape
from markov_chain import MarkovChain, State
from utils import candidate_to_fun
import numpy as np


class MetropolisHastings(MarkovChain):
    """A class used to represent a markov chain generated by M-H algorithm.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
     next_state: Callable[[State, Optional[Sequence[State]]], State]
            Function of current state (and optionally past) that returns next
            state:
            (current: State, past: Sequence[State] = None) -> State
    _next_candidate: Callable[[State], State]
        Function of current state that returns next candidate:
        (current: State) -> State
    _stationary: NDArray[Shape['*'], Any]
        Target stationary distribution.
    _past_max_len: int, default=0
        Maximal length of the past sequence that one wants to track.
        If equal to 0, then no past is tracked.
    _past: Sequence[State], optional
        Sequence of past states.

    Methods
    -------
    sample(n: int) -> Sequence[State]:
        Get sample of length n of the markov chain.
    move(n_steps: int) -> State
        Move the markov chain by n steps.
    """

    def __init__(self,
                 current: State,
                 next_candidate: NDArray[Shape['*, *'], Any]
                 | Callable[[State], State],
                 stationary: NDArray[Shape['*'], Any],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize MetropolisHastings class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        next_candidate: NDArray[Shape['*, *'], Any] | Callable[[State], State]
            Function of current state that returns next candidate:
            (current: State) -> State
        stationary: NDArray[Shape['*'], Any]
            Target stationary distribution.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._stationary = stationary
        self._next_candidate = candidate_to_fun(next_candidate)

        def next_state(current_: State) -> State:
            Z = self._next_candidate(current_)
            V = np.random.uniform()
            if V <= min(1, stationary[Z] / stationary[current_]):
                return Z
            else:
                return current_

        super().__init__(current=current,
                         next_state=next_state,
                         past_max_len=past_max_len,
                         past=past)

    @property
    def stationary(self) -> NDArray[Shape['*'], Any]:
        """Get the stationary distribution of the markov chain."""
        return self._stationary

    @stationary.setter
    def stationary(self, stationary: NDArray[Shape['*'], Any]) -> None:
        """Set the stationary distribution of the markov chain."""
        if sum(stationary) == 1:
            self._stationary = stationary
        else:
            raise ValueError("Stationary distribution does not sum to 1.")

    @property
    def next_candidate(self) -> Callable[[State], State]:
        """Get the next_candidate function or matrix of the markov chain."""
        return self._next_candidate


if __name__ == "__main__":
    from markov_chain import symmetric_walk_transition
    P = symmetric_walk_transition(6)
    print(P)
    metro = MetropolisHastings(0, P, stationary=[0.2, 0.2, 0.2, 0.2, 0.2])
    x = metro.sample(10000)
    unique, counts = np.unique(x, return_counts=True)
    print(counts/10000)

    import matplotlib.pyplot as plt
    plt.hist(x, density=True)
    plt.show()

