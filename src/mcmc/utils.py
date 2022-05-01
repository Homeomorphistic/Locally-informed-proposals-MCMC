"""Utilities for MCMC methods."""

from collections import deque
from typing import TypeVar, Callable, Optional, Deque, Any
from nptyping import NDArray, Shape
import numpy as np


State = TypeVar('State')


def modify_past(next_state: Callable[[State, Optional[Deque[State]]], State],
                past_min_len: int,
                past_max_len: int
                ) -> Callable[[State, Optional[Deque[State]]], State]:
    """Wrap the next_state, so that it also modifies the past.

    Parameters
    ----------
    next_state: Callable[[State, Optional[Deque[State]]], State]
        Function of current (and optionally past) that returns next state:
        (current: State, past: Sequence[State] = None) -> State
    past_min_len: int
        Minimal length of the past sequence that is needed for obtaining
        next state. If equal to 0, then no past is needed to generate next
        state.
    past_max_len: int
        Maximal length of the past sequence that one wants to track.
        If equal to 0, then no past is tracked.

    Returns
    -------
    modified_next_step: Callable[[State, Optional[Deque[State]]], State]
        Wrapped(?) next_state function, so that it modifies the past.
    """
    if past_min_len == 0 and past_max_len == 0:
        return next_state
    else:

        def modified_next_state(current: State,
                                past: Optional[Deque[State]]) \
                                -> State:
            if past is None:
                past = deque()

            next_ = next_state(current, past)

            if past_max_len <= len(past):
                past.pop()
            past.appendleft(current)

            return next_

        return modified_next_state


def candidate_to_fun(next_candidate: NDArray[Shape['*, *'], Any]
                     | Callable[[State], State]
                     ) -> Callable[[State], State]:
    """Convert candidate matrix to function.

    Parameters
    ----------
    next_candidate: NDArray[Shape['*, *'], Any] | Callable[[State], State]
        Candidate matrix or function of current state that returns next
        candidate:
        (current: State) -> State

    Returns
    -------
    next_candidate: Callable[[State], State]
        Function of current state that returns next candidate:
        (current: State) -> State
    """
    if not isinstance(next_candidate, np.ndarray):
        return next_candidate
    else:
        Q = next_candidate
        states = np.arange(Q.shape[0])

        def next_candidate_(current_: State) -> State:
            return np.random.choice(states, p=Q[current_, :])

        return next_candidate_
