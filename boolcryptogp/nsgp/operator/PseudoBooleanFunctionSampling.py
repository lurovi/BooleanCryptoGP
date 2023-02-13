from pymoo.core.sampling import Sampling
import numpy as np
from boolcryptogp.nsgp.structure.PseudoBooleanFunction import PseudoBooleanFunction
from numpy.random import Generator
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


class PseudoBooleanFunctionSampling(Sampling):

    def __init__(self,
                 structure: TreeStructure,
                 walsh: WalshTransform,
                 make_it_balanced: bool = False,
                 force_bent: bool = False,
                 generator: Generator = None
                 ) -> None:
        super().__init__()
        self.__structure: TreeStructure = structure
        self.__make_it_balanced: bool = make_it_balanced
        self.__force_bent: bool = force_bent
        self.__generator: Generator = generator
        self.__walsh: WalshTransform = walsh

    def _do(self, problem, n_samples, **kwargs):
        x = np.empty((n_samples, 1), dtype=object)

        for i in range(n_samples):
            x[i, 0] = PseudoBooleanFunction(self.__walsh, self.__structure.generate_tree(), self.__make_it_balanced, self.__force_bent, self.__generator)

        return x
