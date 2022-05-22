"""TODO module description"""

import tsplib95
import numpy as np
from typing import Dict, Optional
from mcmc.metropolis_hastings import MonteCarloMarkovChain
from tsp_path import TSPath


class TravelingSalesmenMCMC(MonteCarloMarkovChain[TSPath]):
    """TODO docstrings"""

    def __init__(self,
                 name: str = 'berlin52',
                 locally: bool = False,
                 cooling: float = None,
                 past_max_len: int = 0,
                 ) -> None:
        """TODO docstrings"""
        self._problem = tsplib95.load('data/' + name + '.tsp')
        self._nodes = np.array(list(self._problem.get_nodes()))
        self._num_nodes = len(self._nodes)

        init_weight = self._problem.trace_tours([self._nodes])[0]
        self._cooling = cooling or init_weight * 0.01

        if locally:
            self._next_candidate = self.next_candidate_locally
            self._log_ratio = self.log_ratio_locally
        else:
            self._next_candidate = self.next_candidate_uniform
            self._log_ratio = self.log_ratio_uniform

        super().__init__(current=TSPath(path=self._nodes,
                                        weight=init_weight,
                                        problem=self._problem,
                                        locally=locally,
                                        cooling=cooling),
                         past_max_len=past_max_len)

    def next_candidate_uniform(self) -> TSPath:
        """TODO docstrings, deep copy for path"""
        # Random indices to swap in path.
        i, j = np.random.choice(self._num_nodes, size=2, replace=False)
        neighbour_weight = (self._current.get_neighbour_weight(i, j)
                            + self.current._weight)
        neighbour_path = self.current._path.copy()
        # Swap random vertices.
        neighbour_path[i], neighbour_path[j] = neighbour_path[j], neighbour_path[i]
        return TSPath(path=neighbour_path,
                      weight=neighbour_weight)

    def next_candidate_locally(self) -> TSPath:
        """TODO docstrings, deep copy for path"""
        local_dist = self.current._local_dist
        num_neighbours = len(local_dist)
        neighbours_dict = TSPath._neighbours_dict
        neighbour_path = self.current._path.copy()
        # Choose neighbour from local distribution.
        neighbour = np.random.choice(num_neighbours,
                                     p=self.current._local_dist)
        # Get indices to swap.
        i, j = list(neighbours_dict.keys())[neighbour]
        TSPath._last_swap = i, j
        neighbour_weight = (self._current.get_neighbour_weight(i, j)
                            + self.current._weight)
        next_neighbour_weights = self.current.next_neighbours_weights(i, j)
        # Swap vertices.
        neighbour_path[i], neighbour_path[j] = neighbour_path[j], neighbour_path[i]
        return TSPath(path=neighbour_path,
                      weight=neighbour_weight,
                      neighbour_weights=next_neighbour_weights,
                      locally=True)

    def log_ratio_uniform(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        return (self._current._weight - candidate._weight) / self.cooling

    def log_ratio_locally(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        i, j = TSPath._last_swap
        current_id = TSPath._neighbours_dict.get((i, j))
        neighour_id = current_id
        return (0.01*(self._current._weight
                - candidate._weight) / self.cooling
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
        return True#return current._weight <= previous._weight * (1 + tolerance)

    def save_optimum(self,
                     time: float,
                     max_iter: int,
                     tolerance: float,
                     ) -> Dict:
        """TODO desripttion"""
        # TODO save to pickle, so there is a way to obtain attributes of curr.
        from json import dump
        optimum_dict = {'num_steps': self.step_num,
                        'num_stays': self.stay_counter,
                        'time': time,
                        'iter': max_iter,
                        'tol': tolerance,
                        'locally': self.current._locally,
                        'distance': self.current._weight,
                        'path': self.current._path.tolist()}

        file = open(f'results/{self.current._problem.name}_iter='
                    f'{max_iter}'f'_tol={tolerance}_loc='
                    f'{self.current._locally}.json',
                    "w")
        dump(optimum_dict, file)
        file.close()

        return optimum_dict

    @property
    def cooling(self) -> Optional[float]:
        return self._cooling

    def __repr__(self):
        return self._problem.name


if __name__ == "__main__":
    berlin_uni = TravelingSalesmenMCMC(name='berlin52', cooling=1)
    opt_uni = (berlin_uni.find_optimum(max_iter=100, stay_count=10000,
                                       tolerance=0.00))
    print(opt_uni)
    berlin_loc = TravelingSalesmenMCMC(name='berlin52', locally=True, cooling=1)
    opt_loc = (berlin_loc.find_optimum(max_iter=100, stay_count=10000,
                                       tolerance=0.00))
    print(opt_loc)
    print(max(opt_loc.local_dist))



