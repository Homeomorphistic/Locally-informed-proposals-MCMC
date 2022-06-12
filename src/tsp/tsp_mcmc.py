"""TODO module description"""

import tsplib95
import numpy as np
from typing import Dict, Optional, Callable
from json import dump
import os

from mcmc.metropolis_hastings import MonteCarloMarkovChain
from tsp_path import TSPath


class TravelingSalesmenMCMC(MonteCarloMarkovChain[TSPath]):
    """TODO docstrings"""

    def __init__(self,
                 name: str = 'berlin52',
                 locally: bool = False,
                 temperature: Callable[[int], float] = lambda n: 2,
                 cooling: Callable[[int], float] = lambda n: 3/np.log(n+2),
                 past_max_len: int = 0,
                 ) -> None:
        """TODO docstrings"""
        self._problem = tsplib95.load('data/' + name + '.tsp')
        self._nodes = np.array(list(self._problem.get_nodes()))
        self._num_nodes = len(self._nodes)

        init_path = np.random.permutation(self._num_nodes) + 1
        init_weight = self._problem.trace_tours([init_path])[0]# self._problem.trace_tours([self._nodes])[0]
        self._temperature = temperature
        self._cooling = cooling

        if locally:
            self._next_candidate = self.next_candidate_locally
            self._log_ratio = self.log_ratio_locally
        else:
            self._next_candidate = self.next_candidate_uniform
            self._log_ratio = self.log_ratio_uniform

        super().__init__(current=TSPath(path=init_path,#self._nodes,
                                        weight=init_weight,
                                        problem=self._problem,
                                        locally=locally,
                                        temperature=self._temperature(1)),
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
                      weight=neighbour_weight,
                      temperature=self.temperature)

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
                      locally=True,
                      temperature=self.temperature)

    def log_ratio_uniform(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        return (self._current._weight - candidate._weight) / self.cooling

    def log_ratio_locally(self, candidate: TSPath) -> float:
        """TODO docstrings"""
        i, j = TSPath._last_swap
        current_id = TSPath._neighbours_dict.get((i, j))
        neighour_id = current_id
        return ((self._current._weight - candidate._weight)
                / self.cooling
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
        return True #return current._weight <= previous._weight * (1 + tolerance)

    def save_optimum(self,
                     time: float,
                     max_iter: int,
                     tolerance: float
                     ) -> Dict:
        """TODO desripttion"""
        # TODO save to pickle, so there is a way to obtain attributes of curr.

        optimum_dict = {'num_steps': self.step_num,
                        'num_stays': self.stay_counter,
                        'time': time,
                        'iter': max_iter,
                        'locally': self.current._locally,
                        'temperature': self.temperature,
                        'cooling': self.cooling,
                        'distance': self.current._weight,
                        'path': self.current._path.tolist()}

        filename = f'results/{self.current._problem.name}' \
                   f'/locally={self.current._locally}' \
                   f'/temp={self._temperature(1):0.2f}' \
                   f'/cool={self._cooling(1):0.2f}' \
                   f'/{self.current._problem.name}_iter={max_iter}.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        file = open(filename, "w")
        dump(optimum_dict, file)
        file.close()

        return optimum_dict

    @property
    def temperature(self) -> float:
        return self._temperature(self.step_num)

    @property
    def cooling(self):
        return self._cooling(self.step_num)

    def __repr__(self):
        return self._problem.name


if __name__ == "__main__":
    berlin_uni = TravelingSalesmenMCMC(name='berlin52')
    opt_uni = (berlin_uni.find_optimum(max_iter=1000, stay_count=1000))
    print(opt_uni)
    berlin_loc = TravelingSalesmenMCMC(name='berlin52', locally=True)
    opt_loc = berlin_loc.find_optimum(max_iter=1000, stay_count=1000)
    print(opt_loc)



