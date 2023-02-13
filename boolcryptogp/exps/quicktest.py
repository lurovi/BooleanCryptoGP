from genepro.node import Node
from genepro.node_impl import Plus, Minus, Times, Mod, Even, Odd

from boolcryptogp.nsgp.stat.StatsCollector import StatsCollector
from boolcryptogp.nsgp.structure.FullBinaryDomain import FullBinaryDomain
import numpy as np

from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


if __name__ == "__main__":
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

