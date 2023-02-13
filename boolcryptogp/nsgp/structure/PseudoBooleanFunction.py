from genepro.node import Node
import numpy as np
from numpy.random import Generator
from typing import List
from boolcryptogp.nsgp.structure.SimpleBooleanFunction import SimpleBooleanFunction
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform


class PseudoBooleanFunction(SimpleBooleanFunction):
    def __init__(self,
                 walsh: WalshTransform,
                 tree: Node,
                 make_it_balanced: bool = False,
                 force_bent: bool = False,
                 generator: Generator = None
                 ) -> None:
        super().__init__(walsh=walsh)
        self.__node: Node = tree
        self.__make_it_balanced: bool = make_it_balanced
        self.__generator: Generator = generator
        if self.__generator is None:
            self.__generator: Generator = np.random.default_rng()
        result: np.ndarray = self.__node(self.walsh().domain().data())
        if force_bent:
            rrr: int = (self.walsh().domain().number_of_bits() // 2) + 1
            rrr_up: int = 2 ** rrr
            rrr_down: int = - (2 ** rrr)
            result = np.where(result == 0, 0, np.where(result > 0, rrr_up, rrr_down))
        result = self.walsh().invert(result, False)

        if self.__make_it_balanced:
            result = np.where(result == 0, 2, np.where(result > 0, 0, 1))
            curr_ones = (result == 1).sum()
            curr_zeros = (result == 0).sum()
            new_bits: List[int] = []
            indexes_of_twos: List[int] = np.where(result == 2)[0].tolist()
            self.__perc_uncert_pos: float = len(indexes_of_twos) / self.walsh().domain().space_cardinality()
            for _ in indexes_of_twos:
                if curr_ones > curr_zeros:
                    new_bits.append(0)
                    curr_zeros += 1
                else:
                    new_bits.append(1)
                    curr_ones += 1
            new_bits: List[int] = self.__generator.permutation(new_bits).tolist()
            t: int = 0
            for i in indexes_of_twos:
                result[i] = new_bits[t]
                t += 1
        else:
            self.__random_array: np.ndarray = self.__generator.integers(2, size=self.walsh().domain().space_cardinality())
            self.__perc_uncert_pos: float = (result == 0).sum() / self.walsh().domain().space_cardinality()
            result = np.where(result == 0, self.__random_array, np.where(result > 0, 0, 1))

        self.__output: np.ndarray = result.astype(int)

    def node(self) -> Node:
        return self.__node

    def forest(self) -> List[Node]:
        return [self.__node]

    def output(self) -> np.ndarray:
        return self.__output

    def perc_uncert_pos(self) -> float:
        return self.__perc_uncert_pos
