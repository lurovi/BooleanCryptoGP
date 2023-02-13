from boolcryptogp.nsgp.operator.DuplicateSimpleBooleanFunctionElimination import DuplicateSimpleBooleanFunctionElimination
from boolcryptogp.nsgp.operator.IndividualSetting import IndividualSetting
from boolcryptogp.nsgp.operator.PseudoBooleanFunctionCrossover import PseudoBooleanFunctionCrossover
from boolcryptogp.nsgp.operator.PseudoBooleanFunctionMutation import PseudoBooleanFunctionMutation
from boolcryptogp.nsgp.operator.PseudoBooleanFunctionSampling import PseudoBooleanFunctionSampling
from boolcryptogp.nsgp.structure.TreeStructure import TreeStructure
from numpy.random import Generator

from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


class PseudoBooleanFunctionSetting(IndividualSetting):
    def __init__(self,
                 structure: TreeStructure,
                 walsh: WalshTransform,
                 make_it_balanced: bool = False,
                 force_bent: bool = False,
                 generator: Generator = None,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.3
                 ) -> None:
        super().__init__()
        self.__structure: TreeStructure = structure
        self.__sampling: PseudoBooleanFunctionSampling = PseudoBooleanFunctionSampling(structure,
                                                                                       walsh=walsh, make_it_balanced=make_it_balanced, force_bent=force_bent, generator=generator)
        self.__crossover: PseudoBooleanFunctionCrossover = PseudoBooleanFunctionCrossover(structure, walsh=walsh, make_it_balanced=make_it_balanced, force_bent=force_bent, generator=generator, prob=crossover_prob)
        self.__mutation: PseudoBooleanFunctionMutation = PseudoBooleanFunctionMutation(structure, walsh=walsh, make_it_balanced=make_it_balanced, force_bent=force_bent, generator=generator, prob=mutation_prob)
        self.__duplicates_elimination: DuplicateSimpleBooleanFunctionElimination = DuplicateSimpleBooleanFunctionElimination()
        self.set(sampling=self.__sampling, crossover=self.__crossover, mutation=self.__mutation, duplicates_elimination=self.__duplicates_elimination)
