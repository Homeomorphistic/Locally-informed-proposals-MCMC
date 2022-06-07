"""A module used to sample using Metropolis-Hastings algorithm.

TO DO: longer description?

Classes
-------
MonteCarloMarkovChain
MetropolisHastings
"""
from abc import ABC, abstractmethod
from typing import Sequence, Callable, Any, Dict
from nptyping import NDArray, Shape
from markov_chain import MarkovChain, State
from utils import matrix_to_next_candidate
from time import perf_counter
import numpy as np


# TODO docstrings update, after ABC and more
# TODO states not as integers, especially when stationary[state_i]
# TODO add getters and setters


class MonteCarloMarkovChain(ABC, MarkovChain[State]):
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
                 past_max_len: int = 0,
                 past: Sequence[State] = None,
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
        self._stay_counter = 0

        super().__init__(current=current,
                         next_state=self.metropolis_hastings_general_step(),
                         past_max_len=past_max_len,
                         past=past)

    @abstractmethod
    def next_candidate(self) -> State:
        """TODO docstring"""
        pass

    @abstractmethod
    def log_ratio(self, candidate: State) -> float:
        """TODO docstring"""
        pass

    @abstractmethod
    def stop_condition(self, previous: State,
                       current: State,
                       tolerance: float
                       ) -> bool:
        pass

    def metropolis_hastings_general_step(self) -> Callable[[State], State]:
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
            candidate = self.next_candidate()
            unif = np.random.uniform()

            if np.log(unif) <= min(0, self.log_ratio(candidate)):
                self._stay_counter = 0
                return candidate
            else:
                self._stay_counter += 1
                return current

        return next_step

    @abstractmethod
    def save_optimum(self,
                     time: float,
                     max_iter: int,
                     tolerance: float,
                     ) -> Dict:
        pass

    def find_optimum(self,
                     tolerance: float = 0.01,
                     max_iter: int = 1000,
                     stay_count: int = 100,
                     save: bool = False
                     ) -> State:
        """TODO docstring"""
        self.save_optimum(time=0.0,
                          max_iter=0,
                          tolerance=tolerance)
        save_set = {100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500,
                    5000, 7500, 10000, 12500, 15000, 17500, 20000, 50000,
                    100000, 200000, 500000}
        # Stop when chain stays at the same state for too long or some stop
        # condition is achieved or too many iterations.
        start = perf_counter()
        while (self._stay_counter < stay_count
               and self.stop_condition(self._current, self.__next__(), tolerance)
               and self._step_num < max_iter):
            if save and (self.step_num in save_set):
                stop = perf_counter()
                self.save_optimum(time=stop - start,
                                  max_iter=self.step_num,
                                  tolerance=tolerance)
                print(f'tsp_solver at {self.step_num} step.')
        stop = perf_counter()

        print(f'Time elapsed: {stop-start:0.2f}')
        print(f'Number of steps: {self.step_num}')
        print(f'Number of stays: {self.stay_counter}')

        if save:
            self.save_optimum(time=stop-start,
                              max_iter=max_iter,
                              tolerance=tolerance)

        return self._current

    @property
    def stay_counter(self) -> int:
        return self._stay_counter


class MetropolisHastings(MonteCarloMarkovChain[int]):
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
        self._candidate_fun = matrix_to_next_candidate(candidate)

        if np.all(candidate == candidate.T):  # if candidate matrix is
            # symmetric
            self._ratio = self.metropolis_log_ratio
        else:
            self._ratio = self.metropolis_hastings_log_ratio

        super().__init__(current=current,
                         past_max_len=past_max_len,
                         past=past)

    def next_candidate(self) -> State:
        """TODO docstring"""
        return self._candidate_fun(self._current)

    def log_ratio(self, candidate: State) -> float:
        """TODO docstring"""
        return self._ratio(candidate)

    def stop_condition(self, previous: State,
                       current: State,
                       tolerance: float = 0.01
                       ) -> bool:
        return True

    def metropolis_hastings_log_ratio(self, candidate: State) -> float:
        """Calculate classic metropolis-hastings log ratio"""
        return (np.log(self._stationary[candidate])
                + np.log(self._candidate[self._current, candidate])
                - np.log(self._stationary[self._current])
                - np.log(self._candidate[candidate, self._current]))

    def metropolis_log_ratio(self, candidate: State) -> float:
        """Calculate classic metropolis log ratio"""
        return (np.log(self._stationary[candidate])
                - np.log(self._stationary[self._current]))


if __name__ == "__main__":
    n = 5
    metro = MetropolisHastings(current=0,
                               candidate=np.ones((n, n))/n,
                               stationary=np.array([0.1, 0.1, 0.4, 0.1, 0.3])
                               )

    x = metro.sample(1000)
    import matplotlib.pyplot as plt
    plt.hist(x, density=True, ec='black', bins=np.arange(n+1))
    plt.show()

    #print(metro.find_optimum(max_iter=100))
    #print(metro._stay_counter)

