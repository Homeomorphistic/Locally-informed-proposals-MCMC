from stochastic_process import StochasticProcess
import numpy as np
import matplotlib.pyplot as plt

class MarkovChain(StochasticProcess):
    def __init__(self, current, next_state, past=[None]):
        def mc_next_state(curr, pst=[None]):
            return next_state(curr)

        super().__init__(current, next_state=mc_next_state, past_min_len=0, past=past)

class HomogeneousMarkovChain(MarkovChain):
    def __init__(self, mu, P, past=[None]):
        self._mu = mu
        self._P = P
        self._states = np.arange(len(mu))

        def next_state(curr):
            return np.random.choice(self._states, size=1, p=P[curr, :])

        super().__init__(current=np.random.choice(self._states, p=mu), next_state=next_state, past=past)

    @property
    def states(self):
        return self._states

    @property
    def mu(self):
        return self._mu

    @property
    def P(self):
        return self._P


def Ehrenfest(N): # TO DO rethink  N
    P = np.zeros((N-1, N-1))
    P[0, 1] = 1
    P[-1, -2] = 1
    for i in range(1, N-2):
        P[i, i-1] = i/N
        P[i, i+1] = (N-i)/N
    return P

def SymmetricWalk(N):
    P = np.zeros((N-1, N-1))
    P[0, 1] = 1
    P[-1, -2] = 1
    for i in range(1, N - 2):
        P[i, i - 1] = 0.5
        P[i, i + 1] = 0.5
    return P