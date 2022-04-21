"""Utilities for MCMC methods"""
from collections import deque


def modify_past(next_step, past_min_len):
    if past_min_len == 0:
        return next_step
    else:
        def modified_next_step(current, past):
            next_state = next_step(current, past)
            past.pop()
            past.appendleft(current)
            return next_state
        return modified_next_step
