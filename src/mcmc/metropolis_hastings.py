from markov_chain import MarkovChain
from markov_chain import SymmetricWalk
import numpy as np


class MetropolisHastings(MarkovChain):
    def __init__(self, current, next_candidate, pi, past=[None]):
        if isinstance(pi, list):
            tmp = pi
            pi = lambda n: tmp[n]
        self._pi = pi

        if isinstance(next_candidate, np.ndarray):
            Q = next_candidate
            states = np.arange(Q.shape[0])
            next_candidate = lambda curr: np.random.choice(states, p=Q[curr, :])
        self._next_candidate = next_candidate

        def next_state(curr):
            Z = next_candidate(curr)
            V = np.random.uniform()
            if V <= min(1, pi(Z) / pi(curr)):
                return Z
            else:
                return curr

        super().__init__(current, next_state, past)

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, pi):
        if sum(pi) == 1:
            self._pi = pi
        else:
            raise ValueError(f"Stationary distribution does not sum to 1.")  # TO DO not if pi is a function

    @property
    def next_candidate(self):
        return self._next_candidate

P = SymmetricWalk(6)
print(P)
metro = MetropolisHastings(0, P, pi=[0.2, 0.2, 0.2, 0.2, 0.2])
x= metro.sample(10000)
unique, counts = np.unique(x, return_counts=True)
print(counts/10000)

import matplotlib.pyplot as plt
plt.hist(x, density=True)
plt.show()

