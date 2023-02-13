from typing import List, Tuple, Any


class ParetoFrontUtils:

    @staticmethod
    def dominates(point_1: List[float], point_2: List[float]) -> bool:
        if len(point_1) != len(point_2):
            raise AttributeError(f"Lengths must be equal.")
        if point_1 == point_2:
            return False
        n: List[bool] = [point_1[i] <= point_2[i] for i in range(len(point_1))]
        return all(n)

    @staticmethod
    def is_dominated(point: List[float], l: List[Tuple[Any, List[float]]]) -> bool:
        for _, point_i in l:
            if ParetoFrontUtils.dominates(point_i, point):
                return True
        return False

    @staticmethod
    def filter_non_dominated_points(l: List[Tuple[Any, List[float]]]) -> List[Tuple[Any, List[float]]]:
        r: List[Tuple[Any, List[float]]] = []
        for o, point_i in l:
            if not ParetoFrontUtils.is_dominated(point_i, l):
                r.append((o, point_i))
        return r
