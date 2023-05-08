import re
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from genepro.node import Node
from pymoo.core.population import Population
from pytexit import py2tex

from boolcryptogp.nsgp.structure.FullBinaryDomain import FullBinaryDomain
from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction


class ResultUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def safe_latex_format(tree: Node) -> str:
        readable_repr = tree.get_readable_repr().replace("u-", "-")
        try:
            latex_repr = ResultUtils.GetLatexExpression(tree)
        except (RuntimeError, TypeError, ZeroDivisionError, Exception) as e:
            latex_repr = readable_repr
        return re.sub(r"(\.[0-9][0-9])(\d+)", r"\1", latex_repr)

    @staticmethod
    def format_tree(tree: Node) -> Dict[str, str]:
        latex_repr = ResultUtils.safe_latex_format(tree)
        parsable_repr = str(tree.get_subtree())
        return {"latex": latex_repr, "parsable": parsable_repr}

    @staticmethod
    def GetHumanExpression(tree: Node):
        result = ['']  # trick to pass string by reference
        ResultUtils._GetHumanExpressionRecursive(tree, result)
        return result[0]

    @staticmethod
    def GetLatexExpression(tree: Node):
        human_expression = ResultUtils.GetHumanExpression(tree)
        # add linear scaling coefficients
        latex_render = py2tex(human_expression.replace("^", "**"),
                              print_latex=False,
                              print_formula=False,
                              simplify_output=False,
                              verbose=False,
                              simplify_fractions=False,
                              simplify_ints=False,
                              simplify_multipliers=False,
                              ).replace('$$', '').replace('--', '+')
        # fix {x11} and company and change into x_{11}
        latex_render = re.sub(
            r"x(\d+)",
            r"x_{\1}",
            latex_render
        )
        latex_render = latex_render.replace('\\timesx', '\\times x').replace('--', '+').replace('+-', '-').replace('-+',
                                                                                                                   '-')
        return latex_render

    @staticmethod
    def _GetHumanExpressionRecursive(tree: Node, result):
        args = []
        for i in range(tree.arity):
            ResultUtils._GetHumanExpressionRecursive(tree.get_child(i), result)
            args.append(result[0])
        result[0] = ResultUtils._GetHumanExpressionSpecificNode(tree, args)
        return result

    @staticmethod
    def _GetHumanExpressionSpecificNode(tree: Node, args):
        return tree._get_args_repr(args)

    @staticmethod
    def parse_pareto_front_pseudo_boolean_functions(opt: Population,
                                                    seed: int,
                                                    pop_size: int,
                                                    num_gen: int,
                                                    n_bits: int,
                                                    max_depth: int,
                                                    pressure: int,
                                                    make_it_balanced: bool,
                                                    force_bent: bool,
                                                    spectral_inversion: bool,
                                                    dataset_type: str,
                                                    nearest_bool_mapping: str,
                                                    bent_mapping: str,
                                                    binary_balancing: bool,
                                                    non_linearity_only: bool,
                                                    crossover_probability: float,
                                                    mutation_probability: float,
                                                    execution_time_in_minutes: float
                                                    ) -> pd.DataFrame:
        l: List[Tuple[PseudoBooleanFunction, List[float]]] = []
        for individual in opt:
            pseudo_boolean_function: PseudoBooleanFunction = individual.X[0]
            if non_linearity_only:
                f: List[float] = [individual.F[0]]
            else:
                f: List[float] = [individual.F[0], individual.F[1], individual.F[2], individual.F[3]]
            l.append((pseudo_boolean_function, f))

        return ResultUtils.parse_pareto_front_pseudo_boolean_functions_from_list(opt=l,
                                                                                 seed=seed,
                                                                                 pop_size=pop_size,
                                                                                 num_gen=num_gen,
                                                                                 n_bits=n_bits,
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

    @staticmethod
    def parse_pareto_front_pseudo_boolean_functions_from_list(opt: List[Tuple[PseudoBooleanFunction, List[float]]],
                                                              seed: int,
                                                              pop_size: int,
                                                              num_gen: int,
                                                              n_bits: int,
                                                              max_depth: int,
                                                              pressure: int,
                                                              make_it_balanced: bool,
                                                              force_bent: bool,
                                                              spectral_inversion: bool,
                                                              dataset_type: str,
                                                              nearest_bool_mapping: str,
                                                              bent_mapping: str,
                                                              binary_balancing: bool,
                                                              non_linearity_only: bool,
                                                              crossover_probability: float,
                                                              mutation_probability: float,
                                                              execution_time_in_minutes: float
                                                              ) -> pd.DataFrame:
        pareto_front_dict: Dict[str, List[Any]] = {"ParsableTree": [], "LatexTree": [],
                                                   "Degree": [], "Balancing": [],
                                                   "NonLinearity": [], "Resiliency": [],
                                                   "PercentageUncertainPositions": [], "PercentageDiffPositions": [],

                                                   "Seed": [], "PopSize": [], "NumGen": [],
                                                   "NumOffsprings": [], "NumBits": [], "MaxDepth": [], "Pressure": [],
                                                   "MakeItBalanced": [], "ForceBent": [],
                                                   "SpectralInversion": [], "DatasetType": [],
                                                   "NearestBoolMapping": [], "BentMapping": [],
                                                   "BinaryBalancing": [], "NonLinearityOnly": [],
                                                   "CrossOverProb": [], "MutationProb": [],
                                                   "ExecutionTimeInMinutes": [],

                                                   "TruthTable": []}
        for individual, fitness in opt:
            pseudo_boolean_function: PseudoBooleanFunction = individual
            current_tree: Node = pseudo_boolean_function.node()
            pareto_front_dict["ParsableTree"].append(str(current_tree.get_subtree()))
            pareto_front_dict["LatexTree"].append(ResultUtils.safe_latex_format(current_tree))

            if non_linearity_only:
                pareto_front_dict["Degree"].append(0)
                pareto_front_dict["Balancing"].append(0)
                pareto_front_dict["NonLinearity"].append(fitness[0])
                pareto_front_dict["Resiliency"].append(0)
            else:
                pareto_front_dict["Degree"].append(fitness[0])
                pareto_front_dict["Balancing"].append(fitness[1])
                pareto_front_dict["NonLinearity"].append(fitness[2])
                pareto_front_dict["Resiliency"].append(fitness[3])

            pareto_front_dict["PercentageUncertainPositions"].append(pseudo_boolean_function.perc_uncert_pos())
            pareto_front_dict["PercentageDiffPositions"].append(pseudo_boolean_function.perc_diff_pos())

            pareto_front_dict["Seed"].append(seed)
            pareto_front_dict["PopSize"].append(pop_size)
            pareto_front_dict["NumGen"].append(num_gen)
            pareto_front_dict["NumOffsprings"].append(pop_size)
            pareto_front_dict["NumBits"].append(n_bits)
            pareto_front_dict["MaxDepth"].append(max_depth)
            pareto_front_dict["Pressure"].append(pressure)
            pareto_front_dict["MakeItBalanced"].append(make_it_balanced)
            pareto_front_dict["ForceBent"].append(force_bent)
            pareto_front_dict["SpectralInversion"].append(spectral_inversion)
            pareto_front_dict["DatasetType"].append(dataset_type)
            pareto_front_dict["NearestBoolMapping"].append(nearest_bool_mapping)
            pareto_front_dict["BentMapping"].append(bent_mapping)
            pareto_front_dict["BinaryBalancing"].append(binary_balancing)
            pareto_front_dict["NonLinearityOnly"].append(non_linearity_only)
            pareto_front_dict["CrossOverProb"].append(crossover_probability)
            pareto_front_dict["MutationProb"].append(mutation_probability)
            pareto_front_dict["ExecutionTimeInMinutes"].append(execution_time_in_minutes)

            pareto_front_dict["TruthTable"].append(
                FullBinaryDomain.from_numpy_to_binary_string(pseudo_boolean_function.output()))

        pareto_front_df: pd.DataFrame = pd.DataFrame(pareto_front_dict)
        return pareto_front_df

    @staticmethod
    def parse_pareto_front_truth_table_functions_from_list(opt: List[Tuple[np.ndarray, List[float]]],
                                                           seed: int,
                                                           pop_size: int,
                                                           n_bits: int,
                                                           make_it_balanced: bool,
                                                           binary_balancing: bool,
                                                           non_linearity_only: bool,
                                                           execution_time_in_minutes: float
                                                           ) -> pd.DataFrame:
        pareto_front_dict: Dict[str, List[Any]] = {"ParsableTree": [], "LatexTree": [],
                                                   "Degree": [], "Balancing": [],
                                                   "NonLinearity": [], "Resiliency": [],

                                                   "Seed": [], "PopSize": [], "NumGen": [],
                                                   "NumOffsprings": [], "NumBits": [], "MaxDepth": [], "Pressure": [],
                                                   "MakeItBalanced": [], "ForceBent": [],
                                                   "SpectralInversion": [], "DatasetType": [],
                                                   "NearestBoolMapping": [], "BentMapping": [],
                                                   "BinaryBalancing": [], "NonLinearityOnly": [],
                                                   "CrossOverProb": [], "MutationProb": [],
                                                   "ExecutionTimeInMinutes": [],

                                                   "TruthTable": []}
        for individual, fitness in opt:
            pareto_front_dict["ParsableTree"].append("-")
            pareto_front_dict["LatexTree"].append("-")

            if non_linearity_only:
                pareto_front_dict["Degree"].append(0)
                pareto_front_dict["Balancing"].append(0)
                pareto_front_dict["NonLinearity"].append(fitness[0])
                pareto_front_dict["Resiliency"].append(0)
            else:
                pareto_front_dict["Degree"].append(fitness[0])
                pareto_front_dict["Balancing"].append(fitness[1])
                pareto_front_dict["NonLinearity"].append(fitness[2])
                pareto_front_dict["Resiliency"].append(fitness[3])

            pareto_front_dict["Seed"].append(seed)
            pareto_front_dict["PopSize"].append(pop_size)
            pareto_front_dict["NumGen"].append(0)
            pareto_front_dict["NumOffsprings"].append(pop_size)
            pareto_front_dict["NumBits"].append(n_bits)
            pareto_front_dict["MaxDepth"].append(0)
            pareto_front_dict["Pressure"].append(0)
            pareto_front_dict["MakeItBalanced"].append(make_it_balanced)
            pareto_front_dict["ForceBent"].append(False)
            pareto_front_dict["SpectralInversion"].append(False)
            pareto_front_dict["DatasetType"].append("binary")
            pareto_front_dict["NearestBoolMapping"].append("pos_neg_zero")
            pareto_front_dict["BentMapping"].append("pos_neg_zero")
            pareto_front_dict["BinaryBalancing"].append(binary_balancing)
            pareto_front_dict["NonLinearityOnly"].append(non_linearity_only)
            pareto_front_dict["CrossOverProb"].append(0.0)
            pareto_front_dict["MutationProb"].append(0.0)
            pareto_front_dict["ExecutionTimeInMinutes"].append(execution_time_in_minutes)

            pareto_front_dict["TruthTable"].append(
                FullBinaryDomain.from_numpy_to_binary_string(individual))

        pareto_front_df: pd.DataFrame = pd.DataFrame(pareto_front_dict)
        return pareto_front_df

    @staticmethod
    def write_result_to_csv(path: str, run_id: str, pareto_front_df: pd.DataFrame, all_stats: pd.DataFrame, population_per_generation: pd.DataFrame) -> None:
        pareto_front_df.to_csv(path_or_buf=path + "best-" + run_id + ".csv")
        all_stats.to_csv(path_or_buf=path + "stat-" + run_id + ".csv")
        population_per_generation.to_csv(path_or_buf=path + "pareto-" + run_id + ".csv")

    @staticmethod
    def parse_population_per_generation(pop_per_gen: Dict[int, List[Tuple[PseudoBooleanFunction, List[float]]]]) -> pd.DataFrame:
        keys: List[int] = list(pop_per_gen.keys())
        pop_dict: Dict[str, List[Any]] = {"Generation": [], "ParsableTree": [], "LatexTree": [],
                                                   "Degree": [], "Balancing": [],
                                                   "NonLinearity": [], "Resiliency": [], "PercentageUncertainPositions": [],
                                                   "PercentageDiffPositions": [],
                                                   "TruthTable": []}
        for k in keys:
            curr_pop: List[Tuple[PseudoBooleanFunction, List[float]]] = pop_per_gen[k]
            for individual, fitness in curr_pop:
                pop_dict["Generation"].append(k)
                pseudo_boolean_function: PseudoBooleanFunction = individual
                current_tree: Node = pseudo_boolean_function.node()
                pop_dict["ParsableTree"].append(str(current_tree.get_subtree()))
                pop_dict["LatexTree"].append(ResultUtils.safe_latex_format(current_tree))

                if len(fitness) == 1:
                    pop_dict["Degree"].append(0)
                    pop_dict["Balancing"].append(0)
                    pop_dict["NonLinearity"].append(fitness[0])
                    pop_dict["Resiliency"].append(0)
                else:
                    pop_dict["Degree"].append(fitness[0])
                    pop_dict["Balancing"].append(fitness[1])
                    pop_dict["NonLinearity"].append(fitness[2])
                    pop_dict["Resiliency"].append(fitness[3])

                pop_dict["PercentageUncertainPositions"].append(pseudo_boolean_function.perc_uncert_pos())
                pop_dict["PercentageDiffPositions"].append(pseudo_boolean_function.perc_diff_pos())
                pop_dict["TruthTable"].append(
                    FullBinaryDomain.from_numpy_to_binary_string(pseudo_boolean_function.output()))
        return pd.DataFrame(pop_dict)
