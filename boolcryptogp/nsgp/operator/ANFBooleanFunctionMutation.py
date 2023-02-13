from pymoo.core.mutation import Mutation

from boolcryptogp.nsgp.structure.ANFBooleanFunction import ANFBooleanFunction
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure


class ANFBooleanFunctionMutation(Mutation):

    def __init__(self, structure: TreeStructure,
                 walsh: WalshTransform,
                 prob: float = 0.3):
        super().__init__(prob=prob)
        self.__walsh: WalshTransform = walsh
        self.__prob: float = prob
        self.__structure: TreeStructure = structure

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = ANFBooleanFunction(self.__walsh, self.__structure.safe_subforest_mutation(x[i, 0].forest()))
        return x
