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

Functions
---------
symmetric_walk_transition(n: int) -> NDArray[Shape['n, n'], Any]:
    Get sample of length n of the markov chain.
ehrenfest_transition(n: int) -> NDArray[Shape['n, n'], Any]:
    Move the markov chain by n steps.
"""

from stochastic_process import StochasticProcess, State
from typing import Sequence, Callable, Any
from nptyping import NDArray, Shape
import numpy as np


class MarkovChain(StochasticProcess[State]):
    """A class used to represent a markov chain.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the markov chain.
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
        Function of current state that returns next state:
        (current: State) -> State
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
                 next_state: Callable[[State], State],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize MarkovChain class.

        Parameters
        ----------
        current: State
            Current state of the markov chain.
        next_state: Callable[[State], State]
            Function of current (and optionally past) that returns next state:
            (current: State, past: Sequence[State] = None) -> State
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """

        def mc_next_state(current_: State, _: Sequence[State] = None) -> State:
            return next_state(current_)

        super().__init__(current=current,
                         next_state=mc_next_state,
                         past_min_len=0,
                         past_max_len=past_max_len,
                         past=past)


class HomogeneousMarkovChain(MarkovChain):
    """A class used to represent a homogeneous markov chain.

    TO DO: longer description?

    Attributes
    ----------
    _current: State
        Current state of the homogeneous markov chain.
    _next_state: Callable[[State, Optional[Sequence[State]]], State]
        Function of current state that returns next state:
        (current: State) -> State
    _initial: NDArray[Shape['*'], Any]
        Distribution of initial state of the homogeneous markov chain.
    _transition: NDArray[Shape['*, *'], Any]
        Transition matrix of the homogeneous markov chain.
    _past_max_len: int, default=0
        Maximal length of the past sequence that one wants to track.
        If equal to 0, then no past is tracked.
    _past: Sequence[State], optional
        Sequence of past states.

    Methods
    -------
    sample(n: int) -> Sequence[State]:
        Get sample of length n of the homogeneous markov chain.
    move(n_steps: int) -> State
        Move the homogeneous markov chain by n steps.
    """

    def __init__(self,
                 initial: NDArray[Shape['*'], Any],
                 transition: NDArray[Shape['*, *'], Any],
                 past_max_len: int = 0,
                 past: Sequence[State] = None
                 ) -> None:
        """Initialize HomogeneousMarkovChain class.

        Parameters
        ----------
        initial: NDArray[Shape['*'], Any]
            Distribution of initial state of the homogeneous markov chain.
        transition: NDArray[Shape['*, *'], Any]
            Transition matrix of the homogeneous markov chain.
        past_max_len: int, optional
            Maximal length of the past sequence that one wants to track.
            If equal to 0, then no past is tracked.
        past: Sequence[State], optional
            Sequence of past states.
        """
        self._initial = initial
        self._transition = transition
        self._states = np.arange(len(initial))

        def next_state(current):
            return np.random.choice(self._states, p=transition[current, :])

        super().__init__(current=np.random.choice(self._states, p=initial),
                         next_state=next_state,
                         past_max_len=past_max_len,
                         past=past)

    @property
    def states(self):
        """Get the states of the homogeneous markov chain."""
        return self._states

    @property
    def initial(self):
        """Get the initial distribution."""
        return self._initial

    @property
    def transition(self):
        """Get the transition matrix."""
        return self._transition


def ehrenfest_transition(n: int) -> NDArray[Shape['*, *'], Any]:
    """Get a ehrenfest model transition matrix."""
    transition = np.zeros((n, n))
    transition[0, 1] = 1
    transition[-1, -2] = 1
    for i in range(1, n-1):
        transition[i, i-1] = i/n
        transition[i, i+1] = (n-i)/n
    return transition


def symmetric_walk_transition(n: int) -> NDArray[Shape['*, *'], Any]:
    """Get a symmetric walk transition matrix."""
    transition = np.zeros((n, n))
    transition[0, 1] = 1
    transition[-1, -2] = 1
    for i in range(1, n-1):
        transition[i, i-1] = 0.5
        transition[i, i+1] = 0.5
    return transition


if __name__ == '__main__':
    hmc = HomogeneousMarkovChain(initial=np.ones((20,))/20,
                                 transition=ehrenfest_transition(20),
                                 past=[2],
                                 past_max_len=2)
    import matplotlib.pyplot as plt
    from scipy.stats import binom
    x = np.arange(21)
    plt.hist(hmc.sample(10000), density=True, ec='black', bins=x)
    plt.plot(x, binom.pmf(x, 20, 0.5))
    plt.show()
