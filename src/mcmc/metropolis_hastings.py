"""A module used to sample using Metropolis-Hastings algorithm.

TO DO: longer description?

Classes
-------
MonteCarloMarkovChain
MetropolisHastings
"""
from typing import Sequence, Callable, Any
from nptyping import NDArray, Shape
from markov_chain import MarkovChain, State
from utils import matrix_to_next_candidate
import numpy as np


# TODO docstrings for all new classes
# TODO states not as integers, especially when stationary[state_i]


class MonteCarloMarkovChain(MarkovChain[State]):
    """A class used to represent a markov chain generated using M-H algorithm.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
        Function of current state that returns next state:
        (current: State) -> State
    _next_candidate: Callable[[State], State]
        Function of current state that returns next candidate:
        (current: State) -> State
    _log_ratio: Callable[[State, State], float]
        Function of two states that returns log ratio needed for M-H step:
        (state_i: State, state_j: State) -> [-oo, 0]
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
    metropolis_hastings_general_step(next_candidate: Callable[State], State],
                                     log_ratio: Callable[[State, State], float]
                                     ) -> Callable[[State], State]
        Produce next_state function according to M-H algorithm.
    """
    def __init__(self,
                 current: State,
                 next_candidate: Callable[[State], State],
                 log_ratio: Callable[[State, State], float],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize MonteCarloMarkovChain class.

        Attributes
        ----------
        current: State
            Current state of the markov chain.
        next_candidate: Callable[[State], State]
            Function of current state that returns next candidate:
            (current: State) -> State
        log_ratio: Callable[[State, State], float]
            Function of two states that returns log ratio needed for M-H step:
            (state_i: State, state_j: State) -> [-oo, 0]
        past_max_len: int, default=0
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._log_ratio = log_ratio
        self._next_candidate = next_candidate

        super().__init__(current=current,
                         next_state=self.metropolis_hastings_general_step(
                             next_candidate=next_candidate,
                             log_ratio=log_ratio
                         ),
                         past_max_len=past_max_len,
                         past=past)

    def metropolis_hastings_general_step(self,
                                         next_candidate: Callable[[State], State],
                                         log_ratio: Callable[[State, State], float]
                                         ) -> Callable[[State], State]:
        """Produce next_state function according to M-H algorithm.

        Parameters
        ----------
        next_candidate: Callable[[State], State]
            Function of current state that returns next candidate:
            (current: State) -> State
        log_ratio: Callable[[State, State], float]
            Function of two states that returns log ratio needed for M-H step:
            (state_i: State, state_j: State) -> [-oo, 0]

        Returns
        -------
        next_step: Callable[[State], State]
        Function of current state that returns next state:
        (current: State) -> State
        """
        def next_step(current: State) -> State:
            Z = next_candidate(current)
            V = np.random.uniform()
            if np.log(V) <= min(0, log_ratio(Z, current)):
                return Z
            else:
                return current

        return next_step


class MetropolisHastings(MonteCarloMarkovChain):
    """A class used to represent a markov chain generated using M-H algorithm.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
        Function of current state that returns next state:
        (current: State) -> State
    _next_candidate: Callable[[State], State]
        Function of current state that returns next candidate:
        (current: State) -> State
    _candidate: NDArray[Shape['*,*'], Any]
        Matrix of transition between candidates.
    _stationary: NDArray[Shape['*'], Any]
        Target stationary distribution.
    _log_ratio: Callable[[State, State], float]
        Function of two states that returns log ratio needed for M-H step:
        (state_i: State, state_j: State) -> [-oo, 0]
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
    metropolis_hastings_general_step(next_candidate: Callable[State], State],
                                     log_ratio: Callable[[State, State], float]
                                     ) -> Callable[[State], State]
        Produce next_state function according to M-H algorithm.
    metropolis_hastings_log_ratio(state_i: State, state_j: State) -> float:
        Calculate classic metropolis-hastings log ratio.
    metropolis_log_ratio(state_i: State, state_j: State) -> float:
        Calculate classic metropolis log ratio.
    """

    def __init__(self,
                 current: State,
                 candidate: NDArray[Shape['*,*'], Any],
                 stationary: NDArray[Shape['*'], Any],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize MetropolisHastings class.

        Attributes
        ----------
        current: State
            Current state of the markov chain.
        _candidate: NDArray[Shape['*,*'], Any]
            Matrix of transition between candidates.
        _stationary: NDArray[Shape['*'], Any]
            Target stationary distribution.
        past_max_len: int, default=0
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._stationary = stationary
        self._candidate = candidate

        if np.all(candidate == candidate.T):  # if candidate matrix is
            # symmetric
            ratio = self.metropolis_log_ratio
            print('symm')
        else:
            ratio = self.metropolis_hastings_log_ratio
            print('no symm')

        super().__init__(current=current,
                         next_candidate=matrix_to_next_candidate(candidate),
                         log_ratio=ratio,
                         past_max_len=past_max_len,
                         past=past)

    def metropolis_hastings_log_ratio(self,
                                      state_i: State,
                                      state_j: State
                                      ) -> float:
        """Calculate classic metropolis-hastings log ratio"""
        return np.log(self._stationary[state_i]) \
               + np.log(self._candidate[state_i, state_j]) \
               - np.log(self._stationary[state_j]) \
               - np.log(self._candidate[state_j, state_i])

    def metropolis_log_ratio(self,
                             state_i: State,
                             state_j: State
                             ) -> float:
        """Calculate classic metropolis log ratio"""
        return np.log(self._stationary[state_i]) \
               - np.log(self._stationary[state_j])


if __name__ == "__main__":
    n = 5
    metro = MetropolisHastings(current=0,
                               candidate=np.ones((n, n))/n,
                               stationary=[0.1, 0.1, 0.4, 0.1, 0.3]
                               )

    x = metro.sample(1000)
    import matplotlib.pyplot as plt
    plt.hist(x, density=True, ec='black', bins=np.arange(n+1))
    plt.show()

