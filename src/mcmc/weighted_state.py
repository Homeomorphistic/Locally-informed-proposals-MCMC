"""TODO module description"""

from abc import ABC, abstractmethod

from stochastic_process import State

from typing import Dict, Tuple, Any, Optional
from nptyping import NDArray, Shape
from scipy.special import softmax
import numpy as np


class WeightedState(ABC):
    """TODO class description, name acronym"""

    _last_swap: Tuple[int, int] = ()
    _neighbours_dict: Dict[Tuple[int, int], int] = {}
    _num_nodes: int = None
    _locally = False

    def __init__(self):
        """Initialize WeightedState class.

        TODO description
        """
        pass

    @abstractmethod
    def get_adjacent_edges(self, i: int) -> Tuple:
        """TODO docstrings"""
        pass

    @abstractmethod
    def get_adjacent_weights(self, *args, **kwargs) -> Tuple:
        """TODO docstrings"""
        pass

    @abstractmethod
    def get_neighbour_weight(self, *args, **kwargs) -> float:
        """TODO docstrings"""
        pass

    @staticmethod
    def get_neighbours_dict() -> Dict[Tuple[int, int], int]:
        n = WeightedState._num_nodes
        neighbour_id = 0
        neighbours_dict = {}

        for i in range(n):
            for j in range(i + 1, n):
                neighbours_dict[(i, j)] = neighbour_id
                neighbour_id += 1

        WeightedState._neighbours_dict = neighbours_dict
        return neighbours_dict

    @abstractmethod
    def get_neighbours_weights(self, *args, **kwargs) -> NDArray[Shape['*'], Any]:
        """TODO docstrings"""
        pass

    @abstractmethod
    def get_local_distribution(self, *args, **kwargs) -> NDArray[Shape['*'], Any]:
        """TODO docstrings"""
        pass

    @abstractmethod
    def next_neighbours_weights(self, *args, **kwargs) -> NDArray[Shape['*'], Any]:
        """TODO docstrings"""
        pass

