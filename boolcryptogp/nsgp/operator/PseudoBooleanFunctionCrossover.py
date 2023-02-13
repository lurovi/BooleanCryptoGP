from numpy.random import Generator
from pymoo.core.crossover import Crossover
import numpy as np

from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


class PseudoBooleanFunctionCrossover(Crossover):
    def __init__(self, structure: TreeStructure,
                 walsh: WalshTransform,
                 make_it_balanced: bool = False,
                 force_bent: bool = False,
                 generator: Generator = None,
                 prob: float = 0.8
                 ) -> None:
        # define the crossover: number of parents and number of offsprings
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.__structure: TreeStructure = structure
        self.__prob: float = prob
        self.__make_it_balanced: bool = make_it_balanced
        self.__force_bent: bool = force_bent
        self.__generator: Generator = generator
        self.__walsh: WalshTransform = walsh

    def _do(self, problem, x, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = x.shape

        # The output with the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        y = np.full_like(x, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            p1, p2 = x[0, k, 0], x[1, k, 0]

            # prepare the offsprings
            aa, bb = self.__structure.safe_subtree_crossover_two_children(p1.node(), p2.node())
            y[0, k, 0], y[1, k, 0] = PseudoBooleanFunction(self.__walsh, aa, self.__make_it_balanced, self.__force_bent, self.__generator), PseudoBooleanFunction(self.__walsh, bb, self.__make_it_balanced, self.__force_bent, self.__generator)

        return y
