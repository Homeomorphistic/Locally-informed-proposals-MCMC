import numpy as np


class StochasticProcess:
    def __init__(self, current, next_state, past_min_len=0, past=[None]):
        self._past_min_len = past_min_len
        if isinstance(past, list):
            past = [past]

        if len(past) >= self._past_min_len:
            self._past = past
        else:
            raise ValueError(f"Past cannot be less than {past_min_len}")

        self._current = current
        self._next_state = next_state

    @property
    def past(self):
        return self._past

    @past.setter
    def past(self, past):
        if type(past) is not list:
            past = [past]

        if len(past) >= self._past_min_len:
            self._past = past
        else:
            raise ValueError(f"Past cannot be less than {self._past_min_len}")

    @property
    def current(self):
        return self._current

    @property
    def next_state(self):
        return self._next_state

    def __iter__(self):
        return self

    def __next__(self):
        self._current = self._next_state(self._current, self._past)
        return self._current

    def sample(self, n):
        X = np.zeros(n)
        X[0] = self._current
        for i in range(1, n):
            X[i] = self.__next__()
        return X
