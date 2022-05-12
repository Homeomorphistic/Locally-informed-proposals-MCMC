import tsplib95
import numpy as np
from scipy.special import softmax
from itertools import count
from typing import List
from mcmc.metropolis_hastings import MonteCarloMarkovChain
from tsp_path import TravelingSalesmenPath as TSPath


class TravelingSalesmenSolver(MonteCarloMarkovChain[TSPath]):
    """TODO docstrings"""

    def __init__(self,
                 name: str = 'berlin52',
                 locally: bool = False,
                 past_max_len: int = 0,
                 ) -> None:
        """TODO docstrings"""
        self._problem = tsplib95.load('data/' + name + '.tsp')
        self._nodes = list(self._problem.get_nodes())
        self._num_nodes = len(self._nodes)
        self._locally = locally

        super().__init__(current=TSPath(problem=self._problem,
                                        path=self._nodes,
                                        locally=locally),
                         past_max_len=past_max_len)

    def next_candidate(self) -> TSPath:
        """TODO docstrings, deep copy for path"""
        # Random indices to swap in path.
        i, j = np.random.choice(self._num_nodes, size=2, replace=False)
        neighbour_weight = self._current.compute_neighbour_weight(i, j)
        neighbour_path = self.current._path
        # Swap random vertices.
        neighbour_path[i], neighbour_path[j] = neighbour_path[i], neighbour_path[j]
        return TSPath(path=neighbour_path,
                      problem=self._problem,
                      weight=neighbour_weight,
                      locally=self._locally)

    def log_ratio(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        return self._current._weight - candidate._weight

    def stop_condition(self, previous: TSPath, current: TSPath) -> bool:
        return current._weight <= previous._weight


if __name__ == "__main__":
    berlin = TravelingSalesmenSolver()
    #print(berlin.find_optimum(max_iter=10000, stay_count=1000))
    print(berlin.move(10))
    print(berlin._current._weight)


