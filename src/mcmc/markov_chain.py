from stochastic_process import StochasticProcess
import numpy as np
import matplotlib.pyplot as plt

class MarkovChain(StochasticProcess):
    def __init__(self, current, next_fun, past=[None]):
        super().__init__(current, next_fun=lambda c, p: next_fun(c), past_min_len=0, past=past)


class HomogeneousMarkovChain(MarkovChain):
    def __init__(self, mu, P, past=[None]):
        self._mu = mu
        self._P = P
        self._states = np.arange(len(mu))

        def next_fun(current):
            return np.random.choice(self._states, size=1, p=P[current[0], :]) #FIND BETTER SOLUTION FOR [0]

        super().__init__(current=np.random.choice(self._states, size=1, p=mu), next_fun=next_fun, past=past)

    @property
    def states(self):
        return self._states

    @property
    def mu(self):
        return self._mu

    @property
    def P(self):
        return self._P


def Ehrenfest(N):
    P = np.zeros((N+1, N+1))
    P[0, 1] = 1
    P[N, N-1] = 1
    for i in range(1, N):
        P[i, i-1] = i/N
        P[i, i+1] = (N-i)/N
    return P

P = Ehrenfest(20)#np.array([[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])
mu = np.zeros(21)
mu[0] = 1
hmc = HomogeneousMarkovChain(mu, P)
plt.hist(hmc.sample(10000))
plt.show()