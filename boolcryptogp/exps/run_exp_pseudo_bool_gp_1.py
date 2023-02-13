from typing import Tuple, List

import pandas as pd
import time
from boolcryptogp.nsgp.problem.BooleanFunctionProblemRunner import BooleanFunctionProblemRunner
from boolcryptogp.util.ResultUtils import ResultUtils
import argparse


if __name__ == "__main__":
    folder_name: str = "results_1"
    pop_size: int = 500
    num_gen: int = 5
    crossover_probability: float = 0.8
    mutation_probability: float = 0.3

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
        for max_depth in [5]:
            for seed_i in seed_list:
                for make_it_balanced in [True, False]:
                    for force_bent in [False]:
                        t: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str] = runner.run_pseudo_boolean_function_NSGA2(pop_size=pop_size,
                                                                                                                           num_gen=num_gen,
                                                                                                                           max_depth=max_depth,
                                                                                                                           crossover_probability=crossover_probability,
                                                                                                                           mutation_probability=mutation_probability,
                                                                                                                           seed=seed_i,
                                                                                                                           verbose=True,
                                                                                                                           multiprocess=True,
                                                                                                                           make_it_balanced=make_it_balanced,
                                                                                                                           force_bent=force_bent,
                                                                                                                           binary_balancing=True)

                        ResultUtils.write_result_to_csv(path=folder_name+"/", run_id=t[3], pareto_front_df=t[0], all_stats=t[1], population_per_generation=t[2])
                        print("NEXT")
    end_time: float = time.time()
    execution_time_in_minutes: float = (end_time - start_time) * (1 / 60)
    print("TOTAL TIME (minutes): " + str(execution_time_in_minutes))
