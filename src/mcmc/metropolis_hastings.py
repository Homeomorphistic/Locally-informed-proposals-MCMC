"""A module used to sample using Metropolis-Hastings algorithm.

TO DO: longer description?

Attributes
----------
State : TypeVar
    A generic type of state, can be int, List[int], np.array, ...

Classes
-------
Metropolis
MetropolisHastings
"""

from typing import Sequence, Callable, Any
from nptyping import NDArray, Shape
from markov_chain import MarkovChain, State
from utils import matrix_to_next_candidate, matrix_to_candidate_transition, \
    candidate_transition_to_next_candidate
import numpy as np


def metropolis_classic_step(next_candidate: Callable[[State], State],
                             stationary: NDArray[Shape['*'], Any]
                             ) -> Callable[[State], State]:

    def next_state(current: State) -> State:
        Z = next_candidate(current)
        V = np.random.uniform()
        if V <= min(1, stationary[Z] / stationary[current]):
            return Z
        else:
            return current

    return next_state


def metropolis_hastings_classic_step(next_candidate: Callable[[State], State],
                                     stationary: NDArray[Shape['*'], Any],
                                     candidate_transition: Callable[[State, State], float]
                                     ) -> Callable[[State], State]:
    def next_step(current: State) -> State:
        Z = next_candidate(current)
        V = np.random.uniform()
        if V <= min(1,
                    stationary[Z]
                    * candidate_transition(current, Z)
                    / stationary[current]
                    / candidate_transition(Z, current)
                    ):
            return Z
        else:
            return current

    return next_step


# TODO docstrings for all new classes
# TODO states not as integers, especially when stationary[state_i]
# TODO MetropolisHastings as one class?


class MonteCarloMarkovChain(MarkovChain):
    """ TODO A class used to represent a markov chain generated by M-H
    algorithm.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
    _num_states: int
        Number of states.
     next_state: Callable[[State, Optional[Sequence[State]]], State]
            Function of current state (and optionally past) that returns next
            state:
            (current: State, past: Sequence[State] = None) -> State
    _next_candidate: Callable[[State], State]
        Function of current state that returns next candidate:
        (current: State) -> State
    _candidate_transition: Callable[[State, State], float]
        Function of 2 states that returns probabilities of transitions
        between them:
        (state_1: State, state_2: State) -> [0,1]
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

    def metropolis_hastings_general_step(self,
                                         next_candidate: Callable[[State], State],
                                         log_ratio: Callable[[State, State], float]
                                         ) -> Callable[[State], State]:

        def next_step(current: State) -> State:
            Z = next_candidate(current)
            V = np.random.uniform()
            if np.log(V) <= min(0, log_ratio(Z, current)):
                return Z
            else:
                return current

        return next_step

    def __init__(self,
                 current: State,
                 next_candidate: Callable[[State], State],
                 log_ratio: Callable[[State, State], float],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize Metropolis class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        num_states: int
            Number of states.
        candidate_transition: NDArray[Shape['*, *'], Any]
                            | Callable[[State, State], float]
            Function or matrix of  probabilities of between transitions:
            (state_1: State, state_2: State) -> [0,1]
        stationary: NDArray[Shape['*'], Any]
            Target stationary distribution.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._next_candidate = next_candidate

        super().__init__(current=current,
                         next_state=self.metropolis_hastings_general_step(
                             next_candidate=next_candidate,
                             log_ratio=log_ratio
                         ),
                         past_max_len=past_max_len,
                         past=past)


class Metropolis(MonteCarloMarkovChain):

    def metropolis_log_ratio(self,
                             state_i: State,
                             state_j: State,
                             ) -> float:
        return np.log(self._stationary[state_i]) \
               - np.log(self._stationary[state_j])

    def __init__(self,
                 current: State,
                 candidate: NDArray[Shape['*,*'], Any],
                 stationary: NDArray[Shape['*'], Any],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize Metropolis class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        num_states: int
            Number of states.
        candidate_transition: NDArray[Shape['*, *'], Any]
                            | Callable[[State, State], float]
            Function or matrix of  probabilities of between transitions:
            (state_1: State, state_2: State) -> [0,1]
        stationary: NDArray[Shape['*'], Any]
            Target stationary distribution.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._stationary = stationary
        self._candidate = candidate

        super().__init__(current=current,
                         next_candidate=matrix_to_next_candidate(candidate),
                         log_ratio=self.metropolis_log_ratio,
                         past_max_len=past_max_len,
                         past=past)


class MetropolisHastings(MonteCarloMarkovChain):

    def metropolis_hastings_log_ratio(self,
                                      state_i: State,
                                      state_j: State,
                                      ) -> float:
        return np.log(self._stationary[state_i]) \
               + np.log(self._candidate[state_i, state_j]) \
               - np.log(self._stationary[state_j]) \
               - np.log(self._candidate[state_j, state_i])

    def __init__(self,
                 current: State,
                 candidate: NDArray[Shape['*,*'], Any],
                 stationary: NDArray[Shape['*'], Any],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize Metropolis class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        num_states: int
            Number of states.
        candidate_transition: NDArray[Shape['*, *'], Any]
                            | Callable[[State, State], float]
            Function or matrix of  probabilities of between transitions:
            (state_1: State, state_2: State) -> [0,1]
        stationary: NDArray[Shape['*'], Any]
            Target stationary distribution.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._stationary = stationary
        self._candidate = candidate

        super().__init__(current=current,
                         next_candidate=matrix_to_next_candidate(candidate),
                         log_ratio=self.metropolis_hastings_log_ratio,
                         past_max_len=past_max_len,
                         past=past)


'''
class Metropolis(MarkovChain):
    """A class used to represent a markov chain generated by Metro algorithm.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
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
        """Initialize Metropolis class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        next_candidate: NDArray[Shape['*, *'], Any] | Callable[[State], State]
            Function of current state or matrix that returns next candidate:
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
        self._next_candidate = matrix_to_next_candidate(next_candidate)

        super().__init__(current=current,
                         next_state=metropolis_classic_step(
                             self._next_candidate,
                             self._stationary),
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


class MetropolisHastings(MarkovChain):
    """A class used to represent a markov chain generated by M-H algorithm.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
    _num_states: int
        Number of states.
     next_state: Callable[[State, Optional[Sequence[State]]], State]
            Function of current state (and optionally past) that returns next
            state:
            (current: State, past: Sequence[State] = None) -> State
    _next_candidate: Callable[[State], State]
        Function of current state that returns next candidate:
        (current: State) -> State
    _candidate_transition: Callable[[State, State], float]
        Function of 2 states that returns probabilities of transitions
        between them:
        (state_1: State, state_2: State) -> [0,1]
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
                 num_states: int,
                 candidate_transition: NDArray[Shape['*, *'], Any]
                 | Callable[[State, State], float],
                 stationary: NDArray[Shape['*'], Any],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize Metropolis class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        num_states: int
            Number of states.
        candidate_transition: NDArray[Shape['*, *'], Any]
                            | Callable[[State, State], float]
            Function or matrix of  probabilities of between transitions:
            (state_1: State, state_2: State) -> [0,1]
        stationary: NDArray[Shape['*'], Any]
            Target stationary distribution.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._num_states = num_states
        self._stationary = stationary
        self._candidate_transition = matrix_to_candidate_transition(
            candidate_transition=candidate_transition
        )
        self._next_candidate = candidate_transition_to_next_candidate(
            candidate_transition=candidate_transition,
            num_states=num_states
        )

        super().__init__(current=current,
                         next_state=metropolis_hastings_classic_step(
                             next_candidate=self._next_candidate,
                             stationary=self._stationary,
                             candidate_transition=self._candidate_transition
                              ),
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
    def candidate_transition(self) -> Callable[[State, State], float]:
        """Get the candidate transition probabilities function."""
        return self._candidate_transition

'''



if __name__ == "__main__":
    n = 5
    # metro = Metropolis(current=0,
    #                    candidate=np.ones((n, n))/n,
    #                    stationary=[0.1, 0.1, 0.4, 0.1, 0.3])
    #
    # x = metro.sample(1000)
    # from utils import random_stochastic_matrix
    # metro = MetropolisHastings(current=0,
    #                            candidate_transition=random_stochastic_matrix(n),
    #                            num_states=n,
    #                            stationary=[0.35, 0.1, 0.1, 0.1, 0.35])
    #
    from utils import random_stochastic_matrix
    metro = MetropolisHastings(current=0,
                               candidate=random_stochastic_matrix(n),
                               stationary=[0.35, 0.1, 0.1, 0.1, 0.35])

    x = metro.sample(10000)

    import matplotlib.pyplot as plt
    plt.hist(x, density=True, ec='black', bins=np.arange(n+1))
    plt.show()

