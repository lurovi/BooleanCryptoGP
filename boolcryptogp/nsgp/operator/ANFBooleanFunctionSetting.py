from boolcryptogp.nsgp.operator.DuplicateSimpleBooleanFunctionElimination import DuplicateSimpleBooleanFunctionElimination
from boolcryptogp.nsgp.operator.ANFBooleanFunctionCrossover import ANFBooleanFunctionCrossover
from boolcryptogp.nsgp.operator.ANFBooleanFunctionMutation import ANFBooleanFunctionMutation
from boolcryptogp.nsgp.operator.ANFBooleanFunctionSampling import ANFBooleanFunctionSampling
from boolcryptogp.nsgp.operator.IndividualSetting import IndividualSetting
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure


class ANFBooleanFunctionSetting(IndividualSetting):
    def __init__(self,
                 structure: TreeStructure,
                 walsh: WalshTransform,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.3,
                 n_trees: int = None,
                 n_trees_min: int = 2,
                 n_trees_max: int = 10,
                 tree_prob: float = 0.70,
                 max_length: int = None
                 ) -> None:
        super().__init__()
        self.__structure: TreeStructure = structure
        self.__sampling: ANFBooleanFunctionSampling = ANFBooleanFunctionSampling(structure, walsh,
                                                                                 n_trees=n_trees,
                                                                                 n_trees_min=n_trees_min,
                                                                                 n_trees_max=n_trees_max, tree_prob=tree_prob)
        self.__crossover: ANFBooleanFunctionCrossover = ANFBooleanFunctionCrossover(structure, walsh, prob=crossover_prob, max_length=max_length)
        self.__mutation: ANFBooleanFunctionMutation = ANFBooleanFunctionMutation(structure, walsh, prob=mutation_prob)
        self.__duplicates_elimination: DuplicateSimpleBooleanFunctionElimination = DuplicateSimpleBooleanFunctionElimination()
        self.set(sampling=self.__sampling, crossover=self.__crossover, mutation=self.__mutation, duplicates_elimination=self.__duplicates_elimination)
