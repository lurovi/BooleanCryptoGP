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

    def _do(self, problem, x, **kwargs):
        # for each individual
        for i in range(len(x)):
            x[i, 0] = PseudoBooleanFunction(self.__walsh, self.__structure.safe_subtree_mutation(x[i, 0].node()), self.__make_it_balanced, self.__force_bent, self.__generator)
        return x
