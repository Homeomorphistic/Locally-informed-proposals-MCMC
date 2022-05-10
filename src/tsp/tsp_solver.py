import tsplib95
import numpy as np
from typing import List
from mcmc.metropolis_hastings import MonteCarloMarkovChain


class TravelingSalesmenProblemSolver(MonteCarloMarkovChain[List]):
    """TODO docstrings"""

    def __init__(self,
                 name: str = 'berlin52',
                 past_max_len: int = 0,
                 ) -> None:
        """TODO docstrings"""
        self._problem = tsplib95.load('data/' + name + '.tsp')
        self._nodes = list(self._problem.get_nodes())
        self._swapped_indices = None
        self._num_nodes = len(self._nodes)

        super().__init__(current=self._nodes,
                         weight=self._problem.trace_canonical_tour(),
                         next_candidate=self.tsp_next_candidate,
                         compute_weight=self.tsp_compute_weight,
                         log_ratio=self.tsp_log_ratio,
                         past_max_len=past_max_len)

    def tsp_next_candidate(self, path: List) -> List:
        """TODO docstrings, deep copy for path"""
        # indices to swap in path
        ind_1, ind_2 = np.random.choice(self._num_nodes, size=2, replace=False)
        self._swapped_indices = ind_1, ind_2
        next_path = list(path)
        # swap random vertices
        next_path[ind_1], next_path[ind_2] = next_path[ind_2], next_path[ind_1]
        return next_path

    def tsp_compute_weight(self, path_i: List, path_j: List) -> float:
        """TODO docstrings"""
        num_nodes = self._num_nodes
        next_weight = self._weight
        k, l = self._swapped_indices
        k_l, k_r = (k-1) % num_nodes, (k+1) % num_nodes
        l_l, l_r = (l-1) % num_nodes, (l+1) % num_nodes
        # edges from previous state
        edge_1 = path_i[k_l], path_i[k]
        edge_2 = path_i[k], path_i[k_r]
        edge_3 = path_i[l_l], path_i[l]
        edge_4 = path_i[l], path_i[l_r]
        # remove their distances
        next_weight -= (self._problem.get_weight(*edge_1)
                        + self._problem.get_weight(*edge_2)
                        + self._problem.get_weight(*edge_3)
                        + self._problem.get_weight(*edge_4)
                        )
        # edges from current state
        edge_1 = path_j[k_l], path_j[k]
        edge_2 = path_j[k], path_j[k_r]
        edge_3 = path_j[l_l], path_j[l]
        edge_4 = path_j[l], path_j[l_r]
        # add their distances
        next_weight += (self._problem.get_weight(*edge_1)
                        + self._problem.get_weight(*edge_2)
                        + self._problem.get_weight(*edge_3)
                        + self._problem.get_weight(*edge_4)
                        )
        return next_weight

    def tsp_log_ratio(self,
                      path_i: List,
                      path_j: List,
                      next_weight: float
                      ) -> float:
        """TODO docstrings"""
        return self._weight - next_weight


if __name__ == "__main__":
    berlin = TravelingSalesmenProblemSolver(past_max_len=3)
    print(berlin.find_optimum(max_iter=10000, stay_count=20))
    print(berlin.past)
    print(berlin._weight)
    print(berlin._stay_counter)

