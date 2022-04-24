"""Utilities for MCMC methods"""
from collections import deque
from typing import TypeVar, Callable, Optional, Deque, Any

State = TypeVar('State')


def modify_past(next_state: Callable[[State, Optional[Deque[State]]], State],
                past_min_len: int, past_max_len: int) \
        -> Callable[[State, Optional[Deque[State]]], State]:
    """A wrapper of next_state, so that it also modifies the past

    Parameters
    ----------
    next_state: Callable[[State, Optional[Deque[State]]], State]
        Function of current (and optionally past) that returns next state:
        (current: State, past: Sequence[State] = None) -> State
    past_min_len: int
        Minimal length of the past sequence that is needed for obtaining next state. If equal to 0, then no past is
        needed to generate next state.
    past_max_len: int
        Maximal length of the past sequence that one wants to track. If equal to 0, then no past is tracked.

    Returns
    -------
    modified_next_step: Callable[[State, Optional[Deque[State]]], State]
        Wrapped(?) next_state function, so that it modifies the past.
    """

    if past_min_len == 0 and past_max_len == 0:
        return next_state
    else:
        def modified_next_state(current: State, past: Deque[State]) -> State:
            next = next_state(current, past)
            if past_max_len <= len(past):
                past.pop()
            past.appendleft(current)
            return next

        return modified_next_state
