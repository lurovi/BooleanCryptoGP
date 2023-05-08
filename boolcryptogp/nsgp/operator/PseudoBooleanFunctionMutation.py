import numpy as np
from pymoo.core.mutation import Mutation
from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from numpy.random import Generator
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


class PseudoBooleanFunctionMutation(Mutation):

    def __init__(self, structure: TreeStructure,
                 walsh: WalshTransform,
                 make_it_balanced: bool = False,
                 force_bent: bool = False,
                 spectral_inversion: bool = True,
                 dataset: np.ndarray = None,
                 nearest_bool_mapping: str = "pos_neg_zero",
                 bent_mapping: str = "pos_neg_zero",
                 generator: Generator = None,
                 prob: float = 0.3
                 ) -> None:
        super().__init__(prob=prob)
        self.__prob: float = prob
        self.__structure: TreeStructure = structure
        self.__make_it_balanced: bool = make_it_balanced
        self.__force_bent: bool = force_bent
        self.__generator: Generator = generator
        self.__walsh: WalshTransform = walsh
        self.__spectral_inversion: bool = spectral_inversion
        self.__dataset: np.ndarray = dataset
        self.__nearest_bool_mapping: str = nearest_bool_mapping
        self.__bent_mapping: str = bent_mapping

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = PseudoBooleanFunction(walsh=self.__walsh,
                                            tree=self.__structure.safe_subtree_mutation(x[i, 0].node()),
                                            make_it_balanced=self.__make_it_balanced,
                                            force_bent=self.__force_bent,
                                            spectral_inversion=self.__spectral_inversion,
                                            dataset=self.__dataset,
                                            nearest_bool_mapping=self.__nearest_bool_mapping,
                                            bent_mapping=self.__bent_mapping,
                                            generator=self.__generator)
        return x
