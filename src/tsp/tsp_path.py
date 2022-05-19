"""TODO module description"""

from typing import Dict, Tuple, Any, Callable
from nptyping import NDArray, Shape
from tsplib95.models import Problem
from scipy.special import softmax
from mcmc.utils import normalize
import numpy as np


class TSPath:
    """TODO class description, name acronym"""

    _last_swap: Tuple[int, int] = ()
    _neighbours_dict: Dict[Tuple[int, int], int] = {}
    _num_nodes: int = None
    _problem: Problem = None

    def __init__(self,
                 path: NDArray[Shape['*'], Any],
                 problem: Problem = None,
                 weight: float = None,
                 locally: bool = False,
                 neighbour_weights: NDArray[Shape['*'], Any] = None
                 ) -> None:
        """Initialize TravelingSalesmenPath class
        
        TODO description
        """
        self._locally = locally
        self._path = path
        self._weight = weight or problem.trace_tours([path])[0]

        # Static attributes:
        if (problem is not None) and (TSPath._problem is None):
            TSPath._problem = problem
            TSPath._nodes = np.array(list(TSPath._problem.get_nodes()))
            TSPath._num_nodes = len(TSPath._nodes)

        # Additional attributes for local methods.
        if locally:
            TSPath._neighbours_dict = self.get_neighbours_dict()

        if locally and neighbour_weights is None:
            self._neighbours_weights = self.get_neighbours_weights()
        else:
            self._neighbours_weights = neighbour_weights

        self._local_dist = locally and self.get_local_distribution()

    @staticmethod
    def path_adjacent_edges(path: NDArray[Shape['*'], Any], i: int) -> Tuple:
        """TODO docstrings indices to swap"""
        n = TSPath._num_nodes

        # Indices to the left and right of i.
        # Assure that first and last one are correctly connected.
        i_l, i_r = (i - 1) % n, (i + 1) % n

        # Edges at i and j of this path.
        left_edge = path[i_l], path[i]
        right_edge = path[i], path[i_r]

        return left_edge, right_edge

    def get_adjacent_edges(self, i: int) -> Tuple:
        return self.path_adjacent_edges(path=self._path, i=i)

    @staticmethod
    def path_adjacent_weights(path: NDArray[Shape['*'], Any],
                              i: int
                              ) -> Tuple[float, float]:
        adj_edges = TSPath.path_adjacent_edges(path=path, i=i)
        adj_weights = (TSPath._problem.get_weight(*adj_edges[0]),
                       TSPath._problem.get_weight(*adj_edges[1]))
        return adj_weights

    def get_adjacent_weights(self, i: int) -> Tuple[float, float]:
        return self.path_adjacent_weights(path=self._path, i=i)

    @staticmethod
    def path_neighbour_weight(path: NDArray[Shape['*'], Any],
                              weight: float,
                              i: int,
                              j: int) -> float:
        """TODO docstrings indices to swap"""
        neighbour = path.copy()
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        neighbour_weight = weight

        weights_to_rmv = (sum(TSPath.path_adjacent_weights(path, i))
                          + sum(TSPath.path_adjacent_weights(path, j)))
        weights_to_add = (sum(TSPath.path_adjacent_weights(neighbour, i))
                          + sum(TSPath.path_adjacent_weights(neighbour, j)))

        return neighbour_weight - weights_to_rmv + weights_to_add

    def get_neighbour_weight(self, i: int, j: int) -> float:
        """TODO docstrings"""
        return self.path_neighbour_weight(path=self._path,
                                          weight=self._weight,
                                          i=i, j=j)

    @staticmethod
    def get_neighbours_dict() -> Dict[Tuple[int, int], int]:
        n = TSPath._num_nodes
        neighbour_id = 0
        neighbours_dict = {}

        for i in range(n):
            for j in range(i + 1, n):
                neighbours_dict[(i, j)] = neighbour_id
                neighbour_id += 1

        TSPath._neighbours_dict = neighbours_dict
        return neighbours_dict

    def get_neighbours_weights(self) -> NDArray[Shape['*'], Any]:
        """TODO docstrings"""
        n = self._num_nodes
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
        """TODO docstrings explain meaning of i and j"""
        n = self._num_nodes
        neighbour = self._path.copy()
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        weights = self._neighbours_weights.copy()

        # Most of the vertices need to be updated by the same amount.
        weights_to_rmv = (sum(self.get_adjacent_weights(i))
                          + sum(self.get_adjacent_weights(j)))
        weights_to_add = (sum(self.path_adjacent_weights(neighbour, i))
                          + sum(self.path_adjacent_weights(neighbour, j)))
        common_update = weights_to_add - weights_to_rmv
        weights += common_update

        # EXCEPTIONS
        i_l, i_r = (i - 1) % n, (i + 1) % n
        j_l, j_r = (j - 1) % n, (j + 1) % n
        exception_set = {i_l, i, i_r, j_l, j, j_r}
        neighbour_id = 0

        for k in range(n):
            for l in range(k + 1, n):
                if (k in exception_set) or (l in exception_set):
                    weights_to_rmv = (sum(self.get_adjacent_weights(k))
                                      + sum(self.get_adjacent_weights(l)))
                    weights_to_add = (
                                sum(self.path_adjacent_weights(neighbour, k))
                                + sum(self.path_adjacent_weights(neighbour, l)))
                    weights[neighbour_id] += (weights_to_add - weights_to_rmv
                                              - common_update)

                neighbour_id += 1

        return weights

    def __str__(self):
        return f'Path:\n{str(self._path)}\nDistance: {self._weight}'

if __name__ == "__main__":
    from tsp_mcmc import TravelingSalesmenMCMC
    berlin = TravelingSalesmenMCMC(locally=True)
    berlin_path = TSPath(problem=berlin._problem,
                         path=berlin._current._path,
                         locally=True)
    print(TSPath._neighbours_dict)

