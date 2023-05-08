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
                 spectral_inversion: bool = True,
                 dataset: np.ndarray = None,
                 nearest_bool_mapping: str = "pos_neg_zero",
                 bent_mapping: str = "pos_neg_zero",
                 generator: Generator = None
                 ) -> None:
        super().__init__(walsh=walsh)
        self.__node: Node = tree
        self.__generator: Generator = generator
        if self.__generator is None:
            self.__generator: Generator = np.random.default_rng()

        result: np.ndarray = self.__node(dataset)
        result = np.rint(result).astype(int)

        if force_bent:
            rrr: int = (self.walsh().domain().number_of_bits() // 2)
            rrr_up: int = 2 ** rrr
            rrr_down: int = - rrr_up

            if bent_mapping == "pos_neg_zero":
                result = np.where(result == 0, 0, np.where(result > 0, rrr_up, rrr_down))
            elif bent_mapping == "pos_neg":
                #print(str((result >= 0).sum() / self.walsh().domain().space_cardinality()), " ", str((result < 0).sum() / self.walsh().domain().space_cardinality()))
                result = np.where(result >= 0, rrr_up, rrr_down)
            elif bent_mapping == "even_odd_zero":
                result = np.where(result == 0, 0, np.where(result % 2 == 0, rrr_up, rrr_down))
            elif bent_mapping == "even_odd":
                #print(str((result % 2 == 0).sum() / self.walsh().domain().space_cardinality()), " ", str((result % 2 != 0).sum() / self.walsh().domain().space_cardinality()))
                result = np.where(result % 2 == 0, rrr_up, rrr_down)
            elif bent_mapping == "binary":
                result = np.where(result == 0, rrr_up, rrr_down)
            elif bent_mapping == "try":
                result = np.where(((result % 2 == 0) & (result >= 0)) | ((result % 2 != 0) & (result < 0)), rrr_up, rrr_down)
            elif bent_mapping == "try2":
                result = np.where(result % 4 == 0, rrr_up, rrr_down)
            else:
                raise AttributeError(f"{bent_mapping} is not a valid bent mapping.")

        if spectral_inversion:
            result = self.walsh().invert(result, False)[0]

        if nearest_bool_mapping == "pos_neg_zero":
            result = np.where(result == 0, 2, np.where(result > 0, 0, 1))
        elif nearest_bool_mapping == "pos_neg":
            result = np.where(result >= 0, 0, 1)
        elif nearest_bool_mapping == "even_odd_zero":
            result = np.where(result == 0, 2, np.where(result % 2 == 0, 0, 1))
        elif nearest_bool_mapping == "even_odd":
            result = np.where(result % 2 == 0, 0, 1)
        else:
            raise AttributeError(f"{nearest_bool_mapping} is not a valid nearest bool mapping.")

        curr_ones: int = (result == 1).sum()
        curr_zeros: int = (result == 0).sum()
        indexes_of_twos: List[int] = np.where(result == 2)[0].tolist()
        number_of_twos: int = len(indexes_of_twos)
        self.__perc_uncert_pos: float = number_of_twos / self.walsh().domain().space_cardinality()
        if number_of_twos == self.walsh().domain().space_cardinality():
            self.__perc_diff_pos: float = 0.0
        else:
            self.__perc_diff_pos: float = abs(curr_ones - curr_zeros) / (self.walsh().domain().space_cardinality() - number_of_twos)

        if make_it_balanced:
            new_bits: List[int] = []
            if number_of_twos != 0:

                if curr_ones == curr_zeros:
                    new_bits.extend([1] * (number_of_twos // 2) + [0] * (number_of_twos // 2))
                    if number_of_twos % 2 != 0:
                        new_bits.append(0)
                else:
                    diff: int = abs(curr_ones - curr_zeros)
                    if curr_ones > curr_zeros:
                        if diff >= number_of_twos:
                            new_bits.extend([0] * number_of_twos)
                        else:
                            tmp: List[int] = [0] * diff
                            residual: int = number_of_twos - diff
                            tmp_2: List[int] = [1] * (residual // 2) + [0] * (residual // 2)
                            if residual % 2 != 0:
                                tmp_2.append(0)
                            new_bits.extend(tmp_2 + tmp)
                    else:
                        if diff >= number_of_twos:
                            new_bits.extend([1] * number_of_twos)
                        else:
                            tmp: List[int] = [1] * diff
                            residual: int = number_of_twos - diff
                            tmp_2: List[int] = [1] * (residual // 2) + [0] * (residual // 2)
                            if residual % 2 != 0:
                                tmp_2.append(0)
                            new_bits.extend(tmp + tmp_2)

                new_bits: List[int] = self.__generator.permutation(new_bits).tolist()
                t: int = 0
                for iii in indexes_of_twos:
                    result[iii] = new_bits[t]
                    t += 1
        else:
            random_array: np.ndarray = self.__generator.integers(2, size=self.walsh().domain().space_cardinality())
            result = np.where(result == 2, random_array, result)

        self.__output: np.ndarray = result.astype(int)

    def node(self) -> Node:
        return self.__node

    def forest(self) -> List[Node]:
        return [self.__node]

    def output(self) -> np.ndarray:
        return self.__output

    def perc_uncert_pos(self) -> float:
        return self.__perc_uncert_pos

    def perc_diff_pos(self) -> float:
        return self.__perc_diff_pos
