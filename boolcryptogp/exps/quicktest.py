import statistics
import random
from typing import Tuple, List

from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Mod, Even, Odd, Div, Square, Max, Min, UnaryMinus, And, Or, Xor, Not, \
    Cube, Sin, Cos
from numpy.random import Generator
from boolcryptogp.nsgp.stat.StatsCollector import StatsCollector
from boolcryptogp.nsgp.structure.FullBinaryDomain import FullBinaryDomain
import numpy as np
import math
from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


def check_results_walsh_mobius_objective_single_truth_table(truth_table: np.ndarray) -> str:
    output_str: str = ""
    n_bits: int = int(math.log(len(truth_table), 2))
    walsh: WalshTransform = WalshTransform(n_bits)
    t: Tuple[np.ndarray, int] = walsh.apply(truth_table)
    spectrum: np.ndarray = t[0]
    spectral_radius: int = t[1]
    t: Tuple[np.ndarray, int] = walsh.invert(spectrum, True)
    truth_table_2: np.ndarray = t[0]
    max_auto_correlation_coefficient: int = t[1]
    if not np.array_equal(truth_table, truth_table_2):
        raise ValueError(f"Walsh error somewhere.")
    if np.max(np.abs(spectrum)) != spectral_radius:
        raise ValueError(f"Spectral radius is not correct.")
    if max_auto_correlation_coefficient != 1:
        raise ValueError(f"Max auto correlation coefficient is not correct.")
    s: str = "["
    for i in truth_table.tolist():
        s += str(i) + ", "
    s = s[:-2] + "]"
    output_str += s
    output_str += "\n"
    s: str = "["
    for i in spectrum.tolist():
        s += str(i) + ", "
    s = s[:-2] + "]"
    output_str += s
    output_str += "\n"
    t: Tuple[List[int], int] = walsh.domain().degree(truth_table)
    degree: int = t[1]
    s: str = "["
    for i in t[0]:
        s += str(i) + ", "
    s = s[:-2] + "]"
    output_str += s
    output_str += "\n"
    output_str += "Degree: " + str(degree) + " - " + "Balancedness: " + str(
        walsh.domain().balancing(truth_table)) + " - " + "NonLinearity: " + str(
        int(walsh.non_linearity(spectrum))) + " - " + "Resiliency: " + str(
        walsh.resiliency(spectrum)) + " - " + "SpectralRadius: " + str(
        spectral_radius) + " - " + "MaxAutoCorrelationCoefficient: " + str(max_auto_correlation_coefficient)
    output_str += "\n"
    return output_str


def check_results_walsh_mobius_objectives(n_bits: int, k: int, seed: int) -> str:
    output_str: str = ""
    generator: np.random.Generator = np.random.default_rng(seed)
    output_str += "NUMBER OF BITS: "+str(n_bits)
    output_str += "\n"
    space_cardinality: int = 2 ** n_bits
    for _ in range(k):
        truth_table: np.ndarray = generator.integers(2, size=space_cardinality)
        output_str += check_results_walsh_mobius_objective_single_truth_table(truth_table)
    return output_str


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def original_test() -> None:
    s = ""
    for i in range(1, 13 + 1):
        s += check_results_walsh_mobius_objectives(i, 10, 1) + "\n"
    with open("/home/luigi/Desktop/codebase/python_data/BooleanCryptoGP/output_wt_regression_test_2.txt", 'w') as text_file:
        text_file.write("%s" % s)


def num_uncertain_pos_test_with_bent() -> None:
    global_uncertain_pos: List[float] = []
    global_diff_pos: List[float] = []
    for seed in range(1, 30 + 1):
        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        uncertain_pos: List[float] = []
        diff_pos: List[float] = []
        n_bits: int = 10
        val: int = 2 ** (n_bits // 2)
        space_cardinality: int = 2 ** n_bits
        walsh: WalshTransform = WalshTransform(n_bits)
        for _ in range(500):
            a: np.ndarray = generator.choice([-val, val], space_cardinality, p=[0.5, 0.5])
            a: np.ndarray = walsh.invert(a, False)[0]
            uncertain_pos.append((a == 0).sum() / space_cardinality)
            if (space_cardinality - (a == 0).sum()) == 0:
                diff_pos.append(0.0)
            else:
                diff_pos.append(abs((a > 0).sum() - (a < 0).sum()) / (space_cardinality - (a == 0).sum()))
        global_uncertain_pos.append(statistics.mean(uncertain_pos))
        global_diff_pos.append(statistics.mean(diff_pos))
    print(np.mean(global_uncertain_pos), " ", np.min(global_uncertain_pos), " ", np.max(global_uncertain_pos))
    print(np.mean(global_diff_pos), " ", np.min(global_diff_pos), " ", np.max(global_diff_pos))


def num_uncertain_pos_test_with_bent_from_random() -> None:
    global_uncertain_pos: List[float] = []
    global_diff_pos: List[float] = []
    for seed in range(1, 30 + 1):
        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        uncertain_pos: List[float] = []
        diff_pos: List[float] = []
        n_bits: int = 10
        val: int = 2 ** (n_bits // 2)
        space_cardinality: int = 2 ** n_bits
        walsh: WalshTransform = WalshTransform(n_bits)
        for _ in range(500):
            a: np.ndarray = generator.integers(low=-500, high=500, size=space_cardinality)
            #a = np.where(a >= 0, val, -val)
            a: np.ndarray = walsh.invert(a, False)[0]
            uncertain_pos.append((a == 0).sum() / space_cardinality)
            if (space_cardinality - (a == 0).sum()) == 0:
                diff_pos.append(0.0)
            else:
                diff_pos.append(abs((a > 0).sum() - (a < 0).sum()) / (space_cardinality - (a == 0).sum()))
        global_uncertain_pos.append(statistics.mean(uncertain_pos))
        global_diff_pos.append(statistics.mean(diff_pos))
    print(np.mean(global_uncertain_pos), " ", np.min(global_uncertain_pos), " ", np.max(global_uncertain_pos))
    print(np.mean(global_diff_pos), " ", np.min(global_diff_pos), " ", np.max(global_diff_pos))


def num_uncertain_pos_test_with_bent_with_trees() -> None:
    global_uncertain_pos: List[float] = []
    global_diff_pos: List[float] = []
    for seed in range(1, 30 + 1):
        generator: Generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        uncertain_pos: List[float] = []
        diff_pos: List[float] = []
        n_bits: int = 8
        val: int = 2 ** (n_bits // 2)
        space_cardinality: int = 2 ** n_bits
        walsh: WalshTransform = WalshTransform(n_bits)
        operators: List[Node] = [Plus(), Times(), Square()]
        #operators: List[Node] = [Plus(), Minus(), Times(), Div(), Square(), Max(), Min(), UnaryMinus()]
        #operators: List[Node] = [Plus(), Minus(), Times(), Mod(), Even(), Odd()]
        #operators: List[Node] = [And(), Or(), Xor(), Not()]
        structure: TreeStructure = TreeStructure(operators=operators,
                                                 #ephemeral_func=lambda: generator.uniform(low=-(2 ** (n_bits - 4)), high=(2 ** (n_bits - 4) + 1e-4)),
                                                 ephemeral_func=lambda: generator.uniform(low=-1.0, high=1.0 + 1e-4),
                                                 n_features=1,
                                                 max_depth=5)

        for _ in range(500):
            t: Node = structure.generate_tree()
            #print(t.get_string_as_lisp_expr())
            p: PseudoBooleanFunction = PseudoBooleanFunction(walsh=walsh, tree=t,
                                                             make_it_balanced=True, force_bent=False,
                                                             spectral_inversion=True, dataset=walsh.domain().integers(),
                                                             nearest_bool_mapping='pos_neg_zero', bent_mapping='binary',
                                                             generator=generator)
            uncertain_pos.append(p.perc_uncert_pos())
            diff_pos.append((p.perc_diff_pos()))
        global_uncertain_pos.append(statistics.mean(uncertain_pos))
        global_diff_pos.append(statistics.mean(diff_pos))
    print(np.mean(global_uncertain_pos), " ", np.min(global_uncertain_pos), " ", np.max(global_uncertain_pos))
    print(np.mean(global_diff_pos), " ", np.min(global_diff_pos), " ", np.max(global_diff_pos))


if __name__ == "__main__":
    num_uncertain_pos_test_with_bent_from_random()

    #print( -87 / 13)
    #print( (-87) // 13)
    #print( int((-87)/13.0) )
    #a: np.ndarray = np.array([ 4,  4,  4,  4, -4, -4,  4, -4,  4,  4,  4,  4,  4, -4,  4, -4])
    #print(WalshTransform(4).invert(a, False)[0])

    #original_test()
    #l: List[int] = [0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    #print(check_results_walsh_mobius_objective_single_truth_table(np.array(l)))

    '''
    n_bits: int = 3
    domain: FullBinaryDomain = FullBinaryDomain(n_bits)

    output: np.ndarray = np.array([0, 0, 0, 1, 0, 1, 1, 0])

    output_2: np.ndarray = np.array([0, 1, 1, 1, 1, 0, 0, 0])

    walsh: WalshTransform = WalshTransform(domain)

    spectrum: np.ndarray = walsh.apply(output)
    print(spectrum)
    print(walsh.invert(spectrum))
    print(domain.degree(output_2))
    print(domain.balancing(output_2))
    spectrum: np.ndarray = walsh.apply(output_2)
    print(spectrum)
    print(walsh.invert(spectrum))
    print(walsh.non_linearity(spectrum))
    print(walsh.resiliency(spectrum))
    print(walsh.resiliency(np.array([0, 0, 0, 0, 0, 0, 0, 2])))
    print(walsh.invert(np.array([1, 3, -4, -1, -5, 2, -1, 8])))
    print("="*30)
    operators = [Plus(), Minus(), Times(), Mod(), Even(), Odd()]
    structure: TreeStructure = TreeStructure(operators, n_features=n_bits, max_depth=5)
    a: Node = structure.generate_tree()
    print(a.get_readable_repr())
    func: PseudoBooleanFunction = PseudoBooleanFunction(domain, walsh, a, False)

    print(func.output())

    s: StatsCollector = StatsCollector(["Degree", "Balancing", "NonLinearity", "Resiliency"],
                                                                [True, False, True, True])
    s.update_fitness_stat_dict(0, np.array([[-5, 2, -2, 0], [-2, 0, -1, -1], [-1, 3, -4, 1]]))
    s.update_fitness_stat_dict(1, np.array([[-8, 1, -3, -4], [-3, 2, -3, -5], [-2, 1, -5, 0]]))
    s.update_fitness_stat_dict(2, np.array([[-10, 3, -5, -8], [-15, 0, -7, -10], [-12, 0, -8, -5]]))

    print(s.build_dataframe().head(100))
    '''
