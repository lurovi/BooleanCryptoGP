from pymoo.core.crossover import Crossover
import numpy as np

from boolcryptogp.nsgp.structure.ANFBooleanFunction import ANFBooleanFunction
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure


class ANFBooleanFunctionCrossover(Crossover):
    def __init__(self, structure: TreeStructure,
                 walsh: WalshTransform,
                 prob: float = 0.8,
                 max_length: int = None
                 ) -> None:
        # define the crossover: number of parents and number of offsprings
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.__walsh: WalshTransform = walsh
        self.__structure: TreeStructure = structure
        self.__prob: float = prob
        self.__max_length: int = max_length

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
            aa, bb = self.__structure.safe_subforest_one_point_crossover_two_children(p1.forest(), p2.forest(), self.__max_length)
            y[0, k, 0], y[1, k, 0] = ANFBooleanFunction(self.__walsh, aa), ANFBooleanFunction(self.__walsh, bb)

        return y
