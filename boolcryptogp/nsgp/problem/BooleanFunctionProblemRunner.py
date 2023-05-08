from functools import partial
import random

from genepro.node import Node
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.core.population import Population
from pymoo.core.result import Result
import pandas as pd
import numpy as np
import time
from numpy.random import Generator
import multiprocessing as mp

from pymoo.operators.selection.tournament import TournamentSelection

import boolcryptogp
from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.util.ParetoFrontUtils import ParetoFrontUtils
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
                                          pressure: int,
                                          operators: List[Node],
                                          low_erc: float,
                                          high_erc: float,
                                          seed: int = None,
                                          multiprocess: bool = False,
                                          verbose: bool = False,
                                          binary_balancing: bool = False,
                                          non_linearity_only: bool = False,
                                          make_it_balanced: bool = False,
                                          force_bent: bool = False,
                                          spectral_inversion: bool = True,
                                          dataset_type: str = "binary",
                                          nearest_bool_mapping: str = "pos_neg_zero",
                                          bent_mapping: str = "pos_neg_zero",
                                          crossover_probability: float = 0.8,
                                          mutation_probability: float = 0.3
                                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        problem: BooleanFunctionProblem = BooleanFunctionProblem(n_bits=self.number_of_bits(),
                                                                 multiprocess=multiprocess,
                                                                 binary_balancing=binary_balancing,
                                                                 non_linearity_only=non_linearity_only)

        if dataset_type == "integer":
            n_vars: int = 1
            dataset: np.ndarray = problem.walsh().domain().integers()
        elif dataset_type == "binary":
            n_vars: int = self.number_of_bits()
            dataset: np.ndarray = problem.walsh().domain().data()
        elif dataset_type == "polar":
            n_vars: int = self.number_of_bits()
            dataset: np.ndarray = problem.walsh().domain().polar_data()
        else:
            raise AttributeError(f"{dataset_type} is not a valid dataset type.")

        if low_erc > high_erc:
            raise AttributeError(f"low erc is higher than high erc.")
        elif low_erc < high_erc:
            ephemeral_func: Callable = lambda: generator.uniform(low=low_erc, high=high_erc)
        else:
            ephemeral_func: Callable = None

        structure: TreeStructure = TreeStructure(operators=operators,
                                                 ephemeral_func=ephemeral_func,
                                                 n_features=n_vars,
                                                 max_depth=max_depth)

        setting: PseudoBooleanFunctionSetting = PseudoBooleanFunctionSetting(structure=structure,
                                                                             walsh=problem.walsh(),
                                                                             make_it_balanced=make_it_balanced,
                                                                             force_bent=force_bent,
                                                                             spectral_inversion=spectral_inversion,
                                                                             dataset=dataset,
                                                                             nearest_bool_mapping=nearest_bool_mapping,
                                                                             bent_mapping=bent_mapping,
                                                                             crossover_prob=crossover_probability,
                                                                             mutation_prob=mutation_probability,
                                                                             generator=generator)

        selector: TournamentSelection = TournamentSelection(pressure=2, func_comp=binary_tournament)
        if not non_linearity_only and pressure != 2:
            raise AttributeError(f"Generic size tournament selection (pressure different from 2) is not defined for multi-objective optimization.")
        if non_linearity_only and pressure != 2:
            selector = TournamentSelection(pressure=pressure, func_comp=boolcryptogp.nsgp.operator.TournamentSelection.TournamentSelection.generic_tournament)

        algorithm: NSGA2 = NSGA2(pop_size=pop_size,
                                 selection=selector,
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
                                                                                                pressure=pressure,
                                                                                                make_it_balanced=make_it_balanced,
                                                                                                force_bent=force_bent,
                                                                                                spectral_inversion=spectral_inversion,
                                                                                                dataset_type=dataset_type,
                                                                                                nearest_bool_mapping=nearest_bool_mapping,
                                                                                                bent_mapping=bent_mapping,
                                                                                                binary_balancing=binary_balancing,
                                                                                                non_linearity_only=non_linearity_only,
                                                                                                crossover_probability=crossover_probability,
                                                                                                mutation_probability=mutation_probability,
                                                                                                execution_time_in_minutes=execution_time_in_minutes)
        run_id: str = f"pseudobooleanfunctionsGPNSGA2-{self.number_of_bits()}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-pressure{pressure}-makeitbalanced{int(make_it_balanced)}-forcebent{int(force_bent)}-spectralinversion{int(spectral_inversion)}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{int(binary_balancing)}-nonlinearityonly{int(non_linearity_only)}-SEED{seed}"
        if verbose:
            print(f"\nPSEUDO BOOLEAN FUNCTIONS GP NSGA2: Completed with seed {seed}, Number of Bits {self.number_of_bits()}, PopSize {pop_size}, NumGen {num_gen}, MaxDepth {max_depth}, Pressure {pressure}, MakeItBalanced {make_it_balanced}, ForceBent {force_bent}, SpectralInversion {spectral_inversion}, DatasetType {dataset_type}, NearestBoolMapping {nearest_bool_mapping}, BentMapping {bent_mapping}, BinaryBalancing {binary_balancing}, NonLinearityOnly {non_linearity_only}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, all_stats, all_pop_per_gen, run_id

    def run_pseudo_boolean_function_only_initialization(self,
                                                        pop_size: int,
                                                        max_depth: int,
                                                        operators: List[Node],
                                                        low_erc: float,
                                                        high_erc: float,
                                                        seed: int = None,
                                                        multiprocess: bool = False,
                                                        binary_balancing: bool = False,
                                                        non_linearity_only: bool = False,
                                                        make_it_balanced: bool = False,
                                                        force_bent: bool = False,
                                                        spectral_inversion: bool = True,
                                                        dataset_type: str = "binary",
                                                        nearest_bool_mapping: str = "pos_neg_zero",
                                                        bent_mapping: str = "pos_neg_zero"
                                                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        walsh: WalshTransform = WalshTransform(self.number_of_bits())

        if dataset_type == "integer":
            n_vars: int = 1
            dataset: np.ndarray = walsh.domain().integers()
        elif dataset_type == "binary":
            n_vars: int = self.number_of_bits()
            dataset: np.ndarray = walsh.domain().data()
        elif dataset_type == "polar":
            n_vars: int = self.number_of_bits()
            dataset: np.ndarray = walsh.domain().polar_data()
        else:
            raise AttributeError(f"{dataset_type} is not a valid dataset type.")

        if low_erc > high_erc:
            raise AttributeError(f"low erc is higher than high erc.")
        elif low_erc < high_erc:
            ephemeral_func: Callable = lambda: generator.uniform(low=low_erc, high=high_erc)
        else:
            ephemeral_func: Callable = None

        structure: TreeStructure = TreeStructure(operators=operators,
                                                 ephemeral_func=ephemeral_func,
                                                 n_features=n_vars,
                                                 max_depth=max_depth)

        start_time: float = time.time()

        map_function: Callable = map
        pool: mp.Pool = None
        if multiprocess:
            pool = mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1)
            map_function = pool.map
        x: List[PseudoBooleanFunction] = []
        truth_tables: List[np.ndarray] = []

        for i in range(pop_size):
            curr_ind: PseudoBooleanFunction = PseudoBooleanFunction(walsh=walsh,
                                                                    tree=structure.generate_tree(),
                                                                    make_it_balanced=make_it_balanced,
                                                                    force_bent=force_bent,
                                                                    spectral_inversion=spectral_inversion,
                                                                    dataset=dataset,
                                                                    nearest_bool_mapping=nearest_bool_mapping,
                                                                    bent_mapping=bent_mapping,
                                                                    generator=generator)
            x.append(curr_ind)
            truth_tables.append(curr_ind.output())

        if non_linearity_only:
            pp: Callable = partial(perform_individual_eval_non_linearity_only, walsh=walsh)
        else:
            pp: Callable = partial(perform_individual_eval, walsh=walsh, binary_balancing=binary_balancing)
        out: List[List[float]] = list(map_function(pp, truth_tables))

        if multiprocess:
            pool.close()
            pool.join()

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
                                                                                                          pressure=0,
                                                                                                          make_it_balanced=make_it_balanced,
                                                                                                          force_bent=force_bent,
                                                                                                          spectral_inversion=spectral_inversion,
                                                                                                          dataset_type=dataset_type,
                                                                                                          nearest_bool_mapping=nearest_bool_mapping,
                                                                                                          bent_mapping=bent_mapping,
                                                                                                          binary_balancing=binary_balancing,
                                                                                                          non_linearity_only=non_linearity_only,
                                                                                                          crossover_probability=0.0,
                                                                                                          mutation_probability=0.0,
                                                                                                          execution_time_in_minutes=execution_time_in_minutes)
        run_id: str = f"pseudobooleanfunctionsRANDOM-{self.number_of_bits()}bit-popsize{pop_size}-numgen{0}-maxdepth{max_depth}-pressure{0}-makeitbalanced{int(make_it_balanced)}-forcebent{int(force_bent)}-spectralinversion{int(spectral_inversion)}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{int(binary_balancing)}-nonlinearityonly{int(non_linearity_only)}-SEED{seed}"
        if True:
            print(f"\nPSEUDO BOOLEAN FUNCTIONS RANDOM: Completed with seed {seed}, Number of Bits {self.number_of_bits()}, PopSize {pop_size}, NumGen {0}, MaxDepth {max_depth}, Pressure {0}, MakeItBalanced {make_it_balanced}, ForceBent {force_bent}, SpectralInversion {spectral_inversion}, DatasetType {dataset_type}, NearestBoolMapping {nearest_bool_mapping}, BentMapping {bent_mapping}, BinaryBalancing {binary_balancing}, NonLinearityOnly {non_linearity_only}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, pd.DataFrame(data={"Empty": [0]}), pd.DataFrame(data={"Empty": [0]}), run_id

    def run_truth_table_function_only_initialization(self,
                                                     pop_size: int,
                                                     seed: int = None,
                                                     multiprocess: bool = False,
                                                     binary_balancing: bool = False,
                                                     non_linearity_only: bool = False,
                                                     make_it_balanced: bool = False,
                                                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        walsh: WalshTransform = WalshTransform(self.number_of_bits())
        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        space_cardinality: int = 2 ** self.number_of_bits()
        start_time: float = time.time()

        map_function: Callable = map
        pool: mp.Pool = None
        if multiprocess:
            pool = mp.Pool(processes=mp.cpu_count() - 1, maxtasksperchild=1)
            map_function = pool.map
        x: List[np.ndarray] = []

        if make_it_balanced:
            tmp: List[int] = [1] * (space_cardinality // 2) + [0] * (space_cardinality // 2)
            base_array: np.ndarray = np.array(tmp)
            for i in range(pop_size):
                x.append(generator.permutation(base_array))
        else:
            for i in range(pop_size):
                x.append(generator.integers(2, size=space_cardinality))

        if non_linearity_only:
            pp: Callable = partial(perform_individual_eval_non_linearity_only, walsh=walsh)
        else:
            pp: Callable = partial(perform_individual_eval, walsh=walsh, binary_balancing=binary_balancing)
        out: List[List[float]] = list(map_function(pp, x))

        if multiprocess:
            pool.close()
            pool.join()

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
                                                                                                       non_linearity_only=non_linearity_only,
                                                                                                       execution_time_in_minutes=execution_time_in_minutes)
        run_id: str = f"truthtablefunctionsRANDOM-{self.number_of_bits()}bit-popsize{pop_size}-numgen{0}-maxdepth{0}-pressure{0}-makeitbalanced{int(make_it_balanced)}-forcebent{0}-spectralinversion{0}-datasettype_{'binary'}-nearestboolmapping_{'pos_neg_zero'}-bentmapping_{'pos_neg_zero'}-binarybalancing{int(binary_balancing)}-nonlinearityonly{int(non_linearity_only)}-SEED{seed}"
        if True:
            print(f"\nTRUTH TABLE FUNCTIONS RANDOM: Completed with seed {seed}, Number of Bits {self.number_of_bits()}, PopSize {pop_size}, NumGen {0}, MaxDepth {0}, Pressure {0}, MakeItBalanced {make_it_balanced}, ForceBent {False}, SpectralInversion {False}, DatasetType {'binary'}, NearestBoolMapping {'pos_neg_zero'}, BentMapping {'pos_neg_zero'}, BinaryBalancing {binary_balancing}, NonLinearityOnly {non_linearity_only}.\nExecutionTimeInMinutes: {execution_time_in_minutes}.\n")
        return pareto_front_df, pd.DataFrame(data={"Empty": [0]}), pd.DataFrame(data={"Empty": [0]}), run_id


def perform_individual_eval(individual: np.ndarray, walsh: WalshTransform, binary_balancing: bool) -> List[float]:
    output: np.ndarray = individual
    t: Tuple[np.ndarray, int] = walsh.apply(output)
    spectrum: np.ndarray = t[0]
    spectral_radius: int = t[1]
    deg: float = -1.0 * walsh.domain().degree(output)[1]
    bal: float = walsh.domain().balancing(output)
    if binary_balancing:
        n_bal: float = 0 if bal == 0 else 1
    else:
        n_bal: float = bal
    # nl: float = -1.0 * walsh.non_linearity(spectrum)
    nl: float = -1.0 * ((2 ** (walsh.domain().number_of_bits() - 1)) - 0.5 * spectral_radius)
    r: float = -1.0 * walsh.resiliency(spectrum)
    return [deg, n_bal, nl, r]


def perform_individual_eval_non_linearity_only(individual: np.ndarray, walsh: WalshTransform) -> List[float]:
    output: np.ndarray = individual
    t: Tuple[np.ndarray, int] = walsh.apply(output)
    spectral_radius: int = t[1]
    # nl: float = -1.0 * walsh.non_linearity(spectrum)
    nl: float = -1.0 * ((2 ** (walsh.domain().number_of_bits() - 1)) - 0.5 * spectral_radius)
    return [nl]
