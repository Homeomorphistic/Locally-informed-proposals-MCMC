"""Base classes"""
from abc import ABC
from abc import abstractmethod

class StochasticProcess(ABC):
    def __init__(self, current, next_fun, past=[None]):
        self._past = past
        self._past_len = len(past)
        self._current = current
        self._next_fun = next_fun

    @property
    def past(self):
        return self._past

    @past.setter
    def past(self, past):
        if len(past) < self._past_len:
            self._past = past
        else:
            raise ValueError("Past cannot be less than ")

    @property
    def current(self):
        return self._current

    @property
    def next_fun(self):
        return self._next_fun

    @abstractmethod
    def sample(self, n):
        pass

class DiscreteStochasticSeqence(StochasticProcess, ABC):
    pass