"""TODO module description"""

from typing import Dict, Tuple, Any, Callable
from nptyping import NDArray, Shape
from tsplib95.models import Problem
from scipy.special import softmax
from mcmc.utils import normalize
import numpy as np


class TravelingSalesmenPath:
    """TODO class description"""

    _last_swap: Tuple[int, int] = ()
    _neighbours_dict: Dict[Tuple[int, int], int] = {}

    def __init__(self,
                 problem: Problem,
                 path: NDArray[Shape['*'], Any],
                 weight: float = None,
                 locally: bool = False
                 ) -> None:
        """Initialize TravelingSalesmenPath class
        
        TODO description
        """
        self._problem = problem
        self._locally = locally
        self._path = path
        self._weight = weight or problem.trace_tours([path])[0]
        self._neighbours_weights = locally and self.get_neighbours_weights()
        self._local_dist = locally and self.get_local_distribution()
        self._norm_const = None
        self._neighbours_dict = (TravelingSalesmenPath._neighbours_dict or
                                 self.get_neighbours_dict())
        # TODO list instead of dict?

    @staticmethod
    def path_neighbour_weight(path: NDArray[Shape['*'], Any],
                              weight: float,
                              get_weight: Callable[[int, int], float],
                              i: int,
                              j: int) -> float:
        """TODO docstrings indices to swap"""
        num_nodes = len(path)
        neighbour = path.copy()
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        neighbour_weight = weight

        # Indices to the left and right of i and j.
        # Assure that first and last one are correctly connected.
        i_l, i_r = (i - 1) % num_nodes, (i + 1) % num_nodes
        j_l, j_r = (j - 1) % num_nodes, (j + 1) % num_nodes

        # Edges at i and j of this path.
        edge_1 = path[i_l], path[i]
        edge_2 = path[i], path[i_r]
        edge_3 = path[j_l], path[j]
        edge_4 = path[j], path[j_r]
        # Remove their distances.
        neighbour_weight -= (get_weight(*edge_1)
                             + get_weight(*edge_2)
                             + get_weight(*edge_3)
                             + get_weight(*edge_4)
                             )

        # Edges at i and j of neighbour path.
        edge_1 = neighbour[i_l], neighbour[i]
        edge_2 = neighbour[i], neighbour[i_r]
        edge_3 = neighbour[j_l], neighbour[j]
        edge_4 = neighbour[j], neighbour[j_r]
        # Add their distances.
        neighbour_weight += (get_weight(*edge_1)
                             + get_weight(*edge_2)
                             + get_weight(*edge_3)
                             + get_weight(*edge_4)
                             )

        return neighbour_weight

    def get_neighbour_weight(self, i: int, j: int) -> float:
        """TODO docstrings"""
        return self.path_neighbour_weight(path=self._path,
                                          weight=self._weight,
                                          get_weight=self._problem.get_weight,
                                          i=i, j=j)

    def get_neighbours_dict(self) -> Dict[Tuple[int, int], int]:
        n = len(self._path)
        neighbour_id = 0
        neighbours_dict = {}

        for i in range(n):
            for j in range(i + 1, n):
                neighbours_dict[(i, j)] = neighbour_id
                neighbour_id += 1

        TravelingSalesmenPath._neighbours_dict = neighbours_dict
        return neighbours_dict

    def get_neighbours_weights(self) -> NDArray[Shape['*'], Any]:
        """TODO docstrings"""
        n = len(self._path)
        weights = np.zeros(n * (n - 1) // 2)
        neighbour_id = 0

        for i in range(n):
            for j in range(i + 1, n):
                weights[neighbour_id] = self.get_neighbour_weight(i, j)
                neighbour_id += 1

        return weights

    def get_local_distribution(self) -> NDArray[Shape['*'], Any]:
        return softmax(-normalize(self._neighbours_weights))

    def next_neighbours_weights(self,
                                i: int,
                                j: int
                                ) -> NDArray[Shape['*'], Any]:
        """TODO docstrings"""
        n = len(self._path)
        weights = self._weight.copy()
        neighbour_id = 0

        for i in range(n):
            for j in range(i + 1, n):
                weights[neighbour_id] = self.get_neighbour_weight(i, j)
                neighbour_id += 1

        return weights

    def __str__(self):
        return f'Path:\n{str(self._path)}\nDistance: {self._weight}'

if __name__ == "__main__":
    from tsp_mcmc import TravelingSalesmenMCMC
    berlin = TravelingSalesmenMCMC(locally=True)
    berlin_path = TravelingSalesmenPath(problem=berlin._problem,
                                               path=berlin._current._path,
                                               locally=True)

