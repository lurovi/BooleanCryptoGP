from pymoo.core.sampling import Sampling
import numpy as np
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.nsgp.structure.ANFBooleanFunction import ANFBooleanFunction
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure


class ANFBooleanFunctionSampling(Sampling):

    def __init__(self,
                 structure: TreeStructure,
                 walsh: WalshTransform,
                 n_trees: int = None,
                 n_trees_min: int = 2,
                 n_trees_max: int = 10,
                 tree_prob: float = 0.70):
        super().__init__()
        self.__walsh: WalshTransform = walsh
        self.__structure: TreeStructure = structure
        self.__n_trees: int = n_trees
        self.__n_trees_min: int = n_trees_min
        self.__n_trees_max: int = n_trees_max
        self.__tree_prob: float = tree_prob

    def _do(self, problem, n_samples, **kwargs):
        x = np.empty((n_samples, 1), dtype=object)

        for i in range(n_samples):
            x[i, 0] = ANFBooleanFunction(self.__walsh, self.__structure.generate_forest(
                                         n_trees=self.__n_trees,
                                         n_trees_min=self.__n_trees_min,
                                         n_trees_max=self.__n_trees_max,
                                         tree_prob=self.__tree_prob)
                                         )

        return x
