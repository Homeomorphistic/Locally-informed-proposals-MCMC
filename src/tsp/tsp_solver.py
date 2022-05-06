import tsplib95
import numpy as np
from typing import List
from mcmc.metropolis_hastings import MonteCarloMarkovChain


class TravelingSalesmenProblemSolver(MonteCarloMarkovChain[List]):

    def __init__(self,
                 path: str = 'data/berlin52.tsp'
                 ) -> None:
        self._problem = tsplib95.load(path)
        self._nodes = list(self._problem.get_nodes())

    def tsp_next_candidate(self, path: List) -> List:
        # indices to swap in path
        ind_1, ind_2 = np.random.choice(self._nodes, size=2, replace=False)
        # swap random vertices
        path[ind_1], path[ind_2] = path[ind_2], path[ind_1]
        return path

    def tsp_log_ratio(self, path_i: List,  path_j: List) -> float:
        pass


if __name__ == "__main__":
    berlin = TravelingSalesmenProblemSolver()

