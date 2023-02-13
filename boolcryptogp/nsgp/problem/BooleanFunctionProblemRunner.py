from functools import partial
import random

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.result import Result
import pandas as pd
import numpy as np
import time
from numpy.random import Generator
import multiprocessing as mp

from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.util.ParetoFrontUtils import ParetoFrontUtils
from genepro.node_impl import Plus, Minus, Times, Mod, Even, Odd
from pymoo.optimize import minimize
from typing import Tuple, List, Callable, Dict
from boolcryptogp.nsgp.operator.PseudoBooleanFunctionSetting import PseudoBooleanFunctionSetting
from boolcryptogp.nsgp.problem.BooleanFunctionProblem import BooleanFunctionProblem
from boolcryptogp.nsgp.stat.StatsCollector import StatsCollector
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from boolcryptogp.util.ResultUtils import ResultUtils


class BooleanFunctionProblemRunner:
    def __init__(self,
                 n_bits: int
                 ) -> None:
        self.__n_bits: int = n_bits

    def number_of_bits(self) -> int:
        return self.__n_bits

    def run_pseudo_boolean_function_NSGA2(self,
                                          pop_size: int,
                                          num_gen: int,
                                          max_depth: int,
                                          seed: int = None,
                                          multiprocess: bool = False,
                                          verbose: bool = False,
                                          binary_balancing: bool = False,
                                          make_it_balanced: bool = False,
                                          force_bent: bool = False,
                                          crossover_probability: float = 0.8,
                                          mutation_probability: float = 0.3
                                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        problem: BooleanFunctionProblem = BooleanFunctionProblem(n_bits=self.number_of_bits(),
                                                                 multiprocess=multiprocess,
                                                                 binary_balancing=binary_balancing)
        structure: TreeStructure = TreeStructure(operators=[Plus(), Minus(), Times(), Mod(), Even(), Odd()],
                                                 n_features=self.number_of_bits(),
                                                 max_depth=max_depth)

        generator: Generator = np.random.default_rng(seed)
        setting: PseudoBooleanFunctionSetting = PseudoBooleanFunctionSetting(structure,
                                                                             problem.walsh(),
                                                                             make_it_balanced=make_it_balanced,
                                                                             force_bent=force_bent,
                                                                             crossover_prob=crossover_probability,
                                                                             mutation_prob=mutation_probability,
                                                                             generator=generator)
        algorithm: NSGA2 = NSGA2(pop_size=pop_size,
                                 sampling=setting.get_sampling(),
                                 crossover=setting.get_crossover(),
                                 mutation=setting.get_mutation(),
                                 eliminate_duplicates=setting.get_duplicates_elimination())

        start_time: float = time.time()
        res: Result = minimize(problem,
                               algorithm,
                               termination=('n_gen', num_gen),
                               seed=seed,
                               verbose=verbose,
                               return_least_infeasible=False,
                               save_history=False
                               )
        end_time: float = time.time()
        execution_time_in_minutes: float = (end_time - start_time)*(1/60)
        problem: BooleanFunctionProblem = res.problem

        stats: StatsCollector = problem.stats_collector()
        all_stats: pd.DataFrame = stats.build_dataframe()
        pareto_per_gen:  Dict[int, List[Tuple[PseudoBooleanFunction, List[float]]]] = problem.pareto_per_gen()
        all_pop_per_gen: pd.DataFrame = ResultUtils.parse_population_per_generation(pareto_per_gen)
        opt: Population = res.opt

        pareto_front_df: pd.DataFrame = ResultUtils.parse_pareto_front_pseudo_boolean_functions(opt=opt,
                                                                                                seed=seed,
                                                                                                pop_size=pop_size,
                                                                                                num_gen=num_gen,
                                                                                                n_bits=self.number_of_bits(),
                                                                                                max_depth=max_depth,
                                                                                                make_it_balanced=make_it_balanced,
                                                                                                force_bent=force_bent,
                                                                                                binary_balancing=binary_balancing,
                                                                                                crossover_probability=crossover_probability,
                                                                                                mutation_probability=mutation_probability,
                                                                                                execution_time_in_minutes=execution_time_in_minutes)
        run_id: str = f"pseudobooleanfunctionsGPNSGA2-{self.number_of_bits()}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-makeitbalanced{int(make_it_balanced)}-forcebent{int(force_bent)}-binarybalancing{int(binary_balancing)}-SEED{seed}"
        if verbose:
            print(f"\nPSEUDO BOOLEAN FUNCTIONS GP NSGA2: Completed with seed {seed}, Number of Bits {self.number_of_bits()}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, MakeItBalanced {make_it_balanced}, ForceBent {force_bent}, BinaryBalancing {binary_balancing}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, all_stats, all_pop_per_gen, run_id

    def run_pseudo_boolean_function_only_initialization(self,
                                                        pop_size: int,
                                                        max_depth: int,
                                                        seed: int = None,
                                                        multiprocess: bool = False,
                                                        binary_balancing: bool = False,
                                                        make_it_balanced: bool = False,
                                                        force_bent: bool = False,
                                                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        walsh: WalshTransform = WalshTransform(self.number_of_bits())
        structure: TreeStructure = TreeStructure(operators=[Plus(), Minus(), Times(), Mod(), Even(), Odd()],
                                                 n_features=self.number_of_bits(),
                                                 max_depth=max_depth)
        generator: Generator = np.random.default_rng(seed)

        start_time: float = time.time()

        map_function: Callable = map
        pool: mp.Pool = None
        if multiprocess:
            pool = mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1)
            map_function = pool.map
        x: List[PseudoBooleanFunction] = []
        truth_tables: List[np.ndarray] = []

        random.seed(seed)
        for i in range(pop_size):
            curr_ind: PseudoBooleanFunction = PseudoBooleanFunction(walsh, structure.generate_tree(), make_it_balanced,
                                                                    force_bent, generator)
            x.append(curr_ind)
            truth_tables.append(curr_ind.output())

        pp: Callable = partial(perform_individual_eval, walsh=walsh, binary_balancing=binary_balancing)
        out: List[List[float]] = list(map_function(pp, truth_tables))

        if multiprocess:
            pool.close()
            pool.join()
        random.seed(None)

        result: List[Tuple[PseudoBooleanFunction, List[float]]] = [(x[i], out[i]) for i in range(len(x))]
        result: List[Tuple[PseudoBooleanFunction, List[float]]] = ParetoFrontUtils.filter_non_dominated_points(result)

        end_time: float = time.time()
        execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)

        pareto_front_df: pd.DataFrame = ResultUtils.parse_pareto_front_pseudo_boolean_functions_from_list(opt=result,
                                                                                                          seed=seed,
                                                                                                          pop_size=pop_size,
                                                                                                          num_gen=0,
                                                                                                          n_bits=self.number_of_bits(),
                                                                                                          max_depth=max_depth,
                                                                                                          make_it_balanced=make_it_balanced,
                                                                                                          force_bent=force_bent,
                                                                                                          binary_balancing=binary_balancing,
                                                                                                          crossover_probability=0.9,
                                                                                                          mutation_probability=0.5,
                                                                                                          execution_time_in_minutes=execution_time_in_minutes)
        run_id: str = f"pseudobooleanfunctionsRANDOM-{self.number_of_bits()}bit-popsize{pop_size}-numgen{0}-maxdepth{max_depth}-makeitbalanced{int(make_it_balanced)}-forcebent{int(force_bent)}-binarybalancing{int(binary_balancing)}-SEED{seed}"
        if True:
            print(f"\nPSEUDO BOOLEAN FUNCTIONS RANDOM: Completed with seed {seed}, Number of Bits {self.number_of_bits()}, PopSize {pop_size}, NumGen {0}, MaxDepth {max_depth}, MakeItBalanced {make_it_balanced}, ForceBent {force_bent}, BinaryBalancing {binary_balancing}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, pd.DataFrame(data={"Empty": [0]}), pd.DataFrame(data={"Empty": [0]}), run_id

    def run_truth_table_function_only_initialization(self,
                                                     pop_size: int,
                                                     seed: int = None,
                                                     multiprocess: bool = False,
                                                     binary_balancing: bool = False,
                                                     make_it_balanced: bool = False,
                                                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        walsh: WalshTransform = WalshTransform(self.number_of_bits())
        generator: Generator = np.random.default_rng(seed)
        space_cardinality: int = 2 ** self.number_of_bits()
        start_time: float = time.time()

        map_function: Callable = map
        pool: mp.Pool = None
        if multiprocess:
            pool = mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1)
            map_function = pool.map
        x: List[np.ndarray] = []

        random.seed(seed)
        if make_it_balanced:
            tmp: List[int] = [1] * (space_cardinality // 2) + [0] * (space_cardinality // 2)
            base_array: np.ndarray = np.array(tmp)
            for i in range(pop_size):
                x.append(generator.permutation(base_array))
        else:
            for i in range(pop_size):
                x.append(generator.integers(2, size=space_cardinality))

        pp: Callable = partial(perform_individual_eval, walsh=walsh, binary_balancing=binary_balancing)
        out: List[List[float]] = list(map_function(pp, x))

        if multiprocess:
            pool.close()
            pool.join()
        random.seed(None)

        result: List[Tuple[np.ndarray, List[float]]] = [(x[i], out[i]) for i in range(len(x))]
        result: List[Tuple[np.ndarray, List[float]]] = ParetoFrontUtils.filter_non_dominated_points(result)

        end_time: float = time.time()
        execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)

        pareto_front_df: pd.DataFrame = ResultUtils.parse_pareto_front_truth_table_functions_from_list(opt=result,
                                                                                                       seed=seed,
                                                                                                       pop_size=pop_size,
                                                                                                       n_bits=self.number_of_bits(),
                                                                                                       make_it_balanced=make_it_balanced,
                                                                                                       binary_balancing=binary_balancing,
                                                                                                       execution_time_in_minutes=execution_time_in_minutes)
        run_id: str = f"truthtablefunctionsRANDOM-{self.number_of_bits()}bit-popsize{pop_size}-numgen{0}-maxdepth{0}-makeitbalanced{int(make_it_balanced)}-forcebent{0}-binarybalancing{int(binary_balancing)}-SEED{seed}"
        if True:
            print(f"\nTRUTH TABLE FUNCTIONS RANDOM: Completed with seed {seed}, Number of Bits {self.number_of_bits()}, PopSize {pop_size}, NumGen {0}, MakeItBalanced {make_it_balanced}, BinaryBalancing {binary_balancing}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, pd.DataFrame(data={"Empty": [0]}), pd.DataFrame(data={"Empty": [0]}), run_id


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
