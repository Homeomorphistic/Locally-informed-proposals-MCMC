"""TODO module description"""

from typing import List
from tsplib95.models import Problem
from scipy.special import softmax
import numpy as np


class TravelingSalesmenPath:
    """TODO class description"""

    def __init__(self,
                problem: Problem,
                path: List,
                weight: float = None,
                locally: bool = False
                ) -> None:
        """Initialize TravelingSalesmenPath class
        
        TODO description
        """
        self._problem = problem
        self._path = path
        self._weight = weight or problem.trace_tours([path])[0]
        self._local_dist = locally and self.compute_local_distribution()

    def compute_neighbour_weight(self, i: int, j: int) -> float:
        """TODO docstrings indices to swap"""
        num_nodes = len(self._path)
        neighbour = self._path.copy()
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        neighbour_weight = self._weight

        # Indices to the left and right of i and j.
        # Assure that first and last one are correctly connected.
        i_l, i_r = (i - 1) % num_nodes, (i + 1) % num_nodes
        j_l, j_r = (j - 1) % num_nodes, (j + 1) % num_nodes

        # Edges at i and j of this path.
        edge_1 = self._path[i_l], self._path[i]
        edge_2 = self._path[i], self._path[i_r]
        edge_3 = self._path[j_l], self._path[j]
        edge_4 = self._path[j], self._path[j_r]
        # Remove their distances.
        neighbour_weight -= (self._problem.get_weight(*edge_1)
                             + self._problem.get_weight(*edge_2)
                             + self._problem.get_weight(*edge_3)
                             + self._problem.get_weight(*edge_4)
                             )

        # Edges at i and j of neighbour path.
        edge_1 = neighbour[i_l], neighbour[i]
        edge_2 = neighbour[i], neighbour[i_r]
        edge_3 = neighbour[j_l], neighbour[j]
        edge_4 = neighbour[j], neighbour[j_r]
        # Add their distances.
        neighbour_weight += (self._problem.get_weight(*edge_1)
                             + self._problem.get_weight(*edge_2)
                             + self._problem.get_weight(*edge_3)
                             + self._problem.get_weight(*edge_4)
                             )

        return neighbour_weight

    def compute_local_distribution(self):
        """TODO docstrings"""
        n = len(self._path)
        local_dist = np.zeros(n * (n - 1) // 2)
        neighbour_id = 0

        for i in range(n):
            for j in range(i + 1, n):
                local_dist[neighbour_id] = self.compute_neighbour_weight(i, j)
                neighbour_id += 1

        # Compute local distribution using softmax.
        return softmax(-local_dist)


if __name__ == "__main__":
    from tsp_solver import TravelingSalesmenSolver
    berlin = TravelingSalesmenSolver(past_max_len=3)
    berlin_path = TravelingSalesmenPath(problem=berlin._problem,
                                               path=berlin._current,
                                               locally=True)

    path2 = berlin._current.copy()
    path2[0], path2[1] = path2[1], path2[0]
    berlin_path2 = TravelingSalesmenPath(problem=berlin._problem,
                                                path=path2,
                                                locally=True)
    dist = berlin_path._local_dist
    print(np.random.choice(len(dist), p=dist))
    dist2 = berlin_path2._local_dist
    print(np.random.choice(len(dist), p=dist2))
