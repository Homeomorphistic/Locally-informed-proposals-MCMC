"""TODO module description"""

import tsplib95
import numpy as np
from mcmc.metropolis_hastings import MonteCarloMarkovChain
from tsp_path import TravelingSalesmenPath as TSPath

log2 = np.log(2)

class TravelingSalesmenMCMC(MonteCarloMarkovChain[TSPath]):
    """TODO docstrings"""

    def __init__(self,
                 name: str = 'berlin52',
                 locally: bool = False,
                 past_max_len: int = 0,
                 ) -> None:
        """TODO docstrings"""
        self._problem = tsplib95.load('data/' + name + '.tsp')
        self._nodes = np.array(list(self._problem.get_nodes()))
        self._num_nodes = len(self._nodes)
        if locally:
            self._next_candidate = self.next_candidate_locally
            self._log_ratio = self.log_ratio_locally
        else:
            self._next_candidate = self.next_candidate_uniform
            self._log_ratio = self.log_ratio_uniform

        super().__init__(current=TSPath(problem=self._problem,
                                        path=self._nodes,
                                        locally=locally),
                         locally=locally,
                         past_max_len=past_max_len)

    def next_candidate_uniform(self) -> TSPath:
        """TODO docstrings, deep copy for path"""
        # Random indices to swap in path.
        i, j = np.random.choice(self._num_nodes, size=2, replace=False)
        neighbour_weight = self._current.compute_neighbour_weight(i, j)
        neighbour_path = self.current._path.copy()
        # Swap random vertices.
        neighbour_path[i], neighbour_path[j] = neighbour_path[j], neighbour_path[i]
        return TSPath(path=neighbour_path,
                      problem=self._problem,
                      weight=neighbour_weight)

    def next_candidate_locally(self) -> TSPath:
        """TODO docstrings, deep copy for path"""
        local_dist = self.current._local_dist
        norm_const = self.current._norm_const
        num_neighbours = len(local_dist)
        neighbours_dict = self.current._neighbours_dict
        neighbour_path = self.current._path.copy()
        # Choose neighbour from local distribution.
        neighbour = np.random.choice(num_neighbours,
                                     p=self.current._local_dist)
        # Get indices to swap.
        i, j = list(neighbours_dict.keys())[neighbour]
        TSPath._last_swap = i, j
        neighbour_weight = self._current.compute_neighbour_weight(i, j)
        # Swap vertices.
        neighbour_path[i], neighbour_path[j] = neighbour_path[j], neighbour_path[i]
        return TSPath(path=neighbour_path,
                      problem=self._problem,
                      weight=neighbour_weight,
                      locally=True)

    def log_ratio_uniform(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        return self._current._weight - candidate._weight

    def log_ratio_locally(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        i, j = TSPath._last_swap
        current_id = TSPath._neighbours_dict.get((i,j))
        neighour_id = current_id
        return (self._current._weight
                - candidate._weight
                + np.log(self.current._local_dist[neighour_id])
                - np.log(candidate._local_dist[current_id]))

    def next_candidate(self) -> TSPath:
        return self._next_candidate()

    def log_ratio(self, candidate: TSPath) -> float:
        return self._log_ratio(candidate)

    def stop_condition(self,
                       previous: TSPath,
                       current: TSPath,
                       tolerance: float = 0.01
                       ) -> bool:
        return current._weight <= previous._weight * (1 + tolerance)

    def __repr__(self):
        return self._problem.name


if __name__ == "__main__":
    berlin_uni = TravelingSalesmenMCMC()
    berlin_loc = TravelingSalesmenMCMC(locally=True)
    print(berlin_uni.find_optimum(max_iter=1000, stay_count=100,
                                  tolerance=0.01))
    print(berlin_loc.find_optimum(max_iter=1000, stay_count=100,
                                  tolerance=0.01))



