from typing import Tuple, List

import pandas as pd
import time
import os
from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Mod, Even, Odd, Div, Log, Square, Cube, Max, Min, UnaryMinus
from boolcryptogp.nsgp.problem.BooleanFunctionProblemRunner import BooleanFunctionProblemRunner
from boolcryptogp.util.ResultUtils import ResultUtils
import argparse


if __name__ == "__main__":
    codebase_folder: str = os.environ['CURRENT_CODEBASE_FOLDER']
    folder_name: str = codebase_folder + 'python_projects/BooleanCryptoGP/boolcryptogp/exps/' + "results_1"
    pressure: int = 5

    pop_size: int = 500
    num_gen: int = 5
    crossover_probability: float = 0.8
    mutation_probability: float = 0.3
    spectral_inversion: bool = True
    dataset_type: str = "integer"
    nearest_bool_mapping: str = "pos_neg_zero"
    bent_mapping: str = "pos_neg"

    non_linearity_only: bool = True

    operators: List[Node] = [Plus(), Times(), Square()]
    #operators: List[Node] = [Plus(), Minus(), Times(), Mod(), Even(), Odd()]

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=str, help='Seed to be adopted for the experiment run.', required=False)
    parser.add_argument('-n', '--nbits', type=str, help='Number of bits.', required=False)

    args: argparse.Namespace = parser.parse_args()
    seed: str = args.seed
    n_bits: str = args.nbits

    if seed is None:
        seed_list: List[int] = list(range(1, 30 + 1))
    else:
        seed_list: List[int] = [int(i) for i in seed.split(",")]

    if n_bits is None:
        n_bits_list: List[int] = list(range(6, 16 + 1))
    else:
        n_bits_list: List[int] = [int(i) for i in n_bits.split(",")]

    start_time: float = time.time()
    for n_bits_i in n_bits_list:
        runner: BooleanFunctionProblemRunner = BooleanFunctionProblemRunner(n_bits_i)
        # low_erc: float = -(2 ** (n_bits_i - 1))
        # high_erc: float = (2 ** (n_bits_i - 1)) + 1e-4
        low_erc: float = -1.0
        high_erc: float = 1.0 + 1e-4
        for max_depth in [5]:
            for seed_i in seed_list:
                for make_it_balanced in [True, False]:
                    for force_bent in [False]:
                        t: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str] = runner.run_pseudo_boolean_function_NSGA2(pop_size=pop_size,
                                                                                                                           num_gen=num_gen,
                                                                                                                           max_depth=max_depth,
                                                                                                                           pressure=pressure,
                                                                                                                           operators=operators,
                                                                                                                           low_erc=low_erc,
                                                                                                                           high_erc=high_erc,
                                                                                                                           seed=seed_i,
                                                                                                                           multiprocess=True,
                                                                                                                           verbose=True,
                                                                                                                           binary_balancing=True,
                                                                                                                           non_linearity_only=non_linearity_only,
                                                                                                                           make_it_balanced=make_it_balanced,
                                                                                                                           force_bent=force_bent,
                                                                                                                           spectral_inversion=spectral_inversion,
                                                                                                                           dataset_type=dataset_type,
                                                                                                                           nearest_bool_mapping=nearest_bool_mapping,
                                                                                                                           bent_mapping=bent_mapping,
                                                                                                                           crossover_probability=crossover_probability,
                                                                                                                           mutation_probability=mutation_probability
                                                                                                                           )

                        ResultUtils.write_result_to_csv(path=folder_name+"/", run_id=t[3], pareto_front_df=t[0], all_stats=t[1], population_per_generation=t[2])
                        print("NEXT")
    end_time: float = time.time()
    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
