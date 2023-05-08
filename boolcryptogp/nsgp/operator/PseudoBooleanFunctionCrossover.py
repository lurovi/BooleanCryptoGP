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
                 spectral_inversion: bool = True,
                 dataset: np.ndarray = None,
                 nearest_bool_mapping: str = "pos_neg_zero",
                 bent_mapping: str = "pos_neg_zero",
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
        self.__spectral_inversion: bool = spectral_inversion
        self.__dataset: np.ndarray = dataset
        self.__nearest_bool_mapping: str = nearest_bool_mapping
        self.__bent_mapping: str = bent_mapping

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
            y[0, k, 0], y[1, k, 0] = PseudoBooleanFunction(walsh=self.__walsh,
                                                           tree=aa,
                                                           make_it_balanced=self.__make_it_balanced,
                                                           force_bent=self.__force_bent,
                                                           spectral_inversion=self.__spectral_inversion,
                                                           dataset=self.__dataset,
                                                           nearest_bool_mapping=self.__nearest_bool_mapping,
                                                           bent_mapping=self.__bent_mapping,
                                                           generator=self.__generator),\
                                     PseudoBooleanFunction(walsh=self.__walsh,
                                                           tree=bb,
                                                           make_it_balanced=self.__make_it_balanced,
                                                           force_bent=self.__force_bent,
                                                           spectral_inversion=self.__spectral_inversion,
                                                           dataset=self.__dataset,
                                                           nearest_bool_mapping=self.__nearest_bool_mapping,
                                                           bent_mapping=self.__bent_mapping,
                                                           generator=self.__generator)

        return y
