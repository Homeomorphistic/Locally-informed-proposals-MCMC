import tsplib95
import numpy as np
from typing import List
from mcmc.metropolis_hastings import MonteCarloMarkovChain


class TravelingSalesmenProblemSolver(MonteCarloMarkovChain[List]):
    """TODO docstrings"""

    def __init__(self,
                 name: str = 'berlin52',
                 max_iter: int = 500
                 ) -> None:
        """TODO docstrings"""
        self._problem = tsplib95.load('data/' + name + '.tsp')
        self._nodes = list(self._problem.get_nodes())
        self._weight = self._problem.trace_canonical_tour()
        self._max_iter = max_iter

        super().__init__(current=self._nodes,
                         next_candidate=self.tsp_next_candidate,
                         log_ratio=self.tsp_log_ratio,
                         max_iter=self._max_iter)

    def tsp_next_candidate(self, path: List) -> List:
        """TODO docstrings"""
        # indices to swap in path
        ind_1, ind_2 = np.random.choice(self._nodes, size=2, replace=False)
        # swap random vertices
        path[ind_1], path[ind_2] = path[ind_2], path[ind_1]
        return path

    def tsp_log_ratio(self, path_i: List,  path_j: List) -> float:
        """TODO docstrings"""
        pass

    def tsp_stop_condition(self, state_i: List, state_j: List) -> bool:
        """TODO docstrings"""
        pass


if __name__ == "__main__":
    berlin = TravelingSalesmenProblemSolver()

