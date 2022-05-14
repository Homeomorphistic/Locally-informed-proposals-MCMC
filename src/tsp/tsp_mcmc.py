import tsplib95
import numpy as np
from mcmc.metropolis_hastings import MonteCarloMarkovChain
from tsp_path import TravelingSalesmenPath as TSPath


class TravelingSalesmenMCMC(MonteCarloMarkovChain[TSPath]):
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
        if locally:
            self._next_candidate = self.next_candidate_locally
        else:
            self._next_candidate = self.next_candidate_uniform

        super().__init__(current=TSPath(problem=self._problem,
                                        path=self._nodes,
                                        locally=locally),
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
        num_neighbours = len(local_dist)
        neighbours_dict = self.current._neighbours_dict
        neighbour_path = self.current._path.copy()
        # Choose neighbour from local distribution.
        neighbour = np.random.choice(num_neighbours, p=local_dist)
        # Get indices to swap.
        i, j = list(neighbours_dict.keys())[neighbour] # TODO list instad of dict?
        neighbour_weight = self._current.compute_neighbour_weight(i, j)
        # Swap vertices.
        neighbour_path[i], neighbour_path[j] = neighbour_path[j], neighbour_path[i]
        return TSPath(path=neighbour_path,
                      problem=self._problem,
                      weight=neighbour_weight,
                      locally=True,
                      neighbours_dict=neighbours_dict)

    def next_candidate(self) -> TSPath:
        return self._next_candidate()

    def log_ratio(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        return self._current._weight - candidate._weight

    def stop_condition(self,
                       previous: TSPath,
                       current: TSPath,
                       tolerance: float = 0.01
                       ) -> bool:
        return current._weight <= previous._weight * (1 + tolerance)


if __name__ == "__main__":
    berlin_uni = TravelingSalesmenMCMC(name='kroA150')
    berlin_loc = TravelingSalesmenMCMC(name='kroA150', locally=True)
    print(berlin_uni.find_optimum(max_iter=10000, stay_count=10000,
                                  tolerance=0.01))
    print(berlin_uni.current._weight)
    print(berlin_loc.find_optimum(max_iter=10000, stay_count=1000,
                              tolerance=0.01))
    print(berlin_loc.current._weight)



