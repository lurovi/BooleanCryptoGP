from functools import partial
from typing import Callable, List, Dict, Tuple

import numpy as np
from pymoo.core.problem import Problem
import multiprocessing as mp
from itertools import chain

from boolcryptogp.nsgp.stat.StatsCollector import StatsCollector
from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.util.ParetoFrontUtils import ParetoFrontUtils


class BooleanFunctionProblem(Problem):
    def __init__(self, n_bits: int, multiprocess: bool = False, binary_balancing: bool = False) -> None:
        super().__init__(n_var=1, n_obj=4, n_ieq_constr=0, n_eq_constr=0)
        self.__walsh: WalshTransform = WalshTransform(n_bits)
        self.__stats_collector: StatsCollector = StatsCollector(["Degree", "Balancing", "NonLinearity", "Resiliency"],
                                                                [True, False, True, True])
        self.__multiprocess: bool = multiprocess
        self.__n_gen: int = -1
        self.__binary_balancing: bool = binary_balancing
        self.__pareto_per_gen: Dict[int, List[Tuple[PseudoBooleanFunction, List[float]]]] = {}

    def walsh(self) -> WalshTransform:
        return self.__walsh

    def stats_collector(self) -> StatsCollector:
        return self.__stats_collector

    def pareto_per_gen(self) -> Dict[int, List[Tuple[PseudoBooleanFunction, List[float]]]]:
        return self.__pareto_per_gen

    def _evaluate(self, x, out, *args, **kwargs):
        self._eval(x, out, *args, **kwargs)

    def _eval(self, x, out, *args, **kwargs):
        self.__n_gen += 1

        map_function: Callable = map
        pool: mp.Pool = None
        if self.__multiprocess:
            pool = mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1)
            map_function = pool.map

        pp: Callable = partial(perform_individual_eval, walsh=self.__walsh, binary_balancing=self.__binary_balancing)
        curr_pop: List[PseudoBooleanFunction] = []
        curr_truth_tables: List[np.ndarray] = []
        curr_perc_uncert_pos: List[float] = []
        for i in range(len(x)):
            curr_ind: PseudoBooleanFunction = x[i, 0]
            curr_pop.append(curr_ind)
            curr_truth_tables.append(curr_ind.output())
            curr_perc_uncert_pos.append(curr_ind.perc_uncert_pos())

        res: List[List[float]] = list(map_function(pp, curr_truth_tables))
        out["F"] = np.array(res)

        if self.__multiprocess:
            pool.close()
            pool.join()

        self.__stats_collector.update_fitness_stat_dict(self.__n_gen, out["F"], np.array(curr_perc_uncert_pos))
        all_fronts: List[Tuple[PseudoBooleanFunction, List[float]]] = list(chain(*self.__pareto_per_gen.values())) if self.__pareto_per_gen != {} else []
        self.__pareto_per_gen[self.__n_gen] = ParetoFrontUtils.filter_non_dominated_points([(curr_pop[i], res[i]) for i in range(len(curr_pop))] + all_fronts)


def perform_individual_eval(individual: np.ndarray, walsh: WalshTransform, binary_balancing: bool) -> List[float]:
    output: np.ndarray = individual
    spectrum: np.ndarray = walsh.apply(output)
    deg: float = -1.0 * walsh.domain().degree(output)
    bal: float = walsh.domain().balancing(output)
    if binary_balancing:
        n_bal: float = 0 if bal == 0 else 1
    else:
        n_bal: float = bal
    nl: float = -1.0 * walsh.non_linearity(spectrum)
    r: float = -1.0 * walsh.resiliency(spectrum)
    return [deg, n_bal, nl, r]
