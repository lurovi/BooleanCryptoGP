import numpy as np
from typing import Dict, List
from boolcryptogp.nsgp.structure.FullBinaryDomain import FullBinaryDomain


class WalshTransform:
    def __init__(self,
                 n_bits: int
                 ) -> None:
        self.__domain: FullBinaryDomain = FullBinaryDomain(n_bits)
        # self.__pairwise_scalar_prod: np.ndarray = WalshTransform.linear_function_xor_of_bitwise_and(self.__domain.data())
        self.__number_of_ones_for_each_number: np.ndarray = np.array([bin(i)[2:].count('1') for i in range(self.__domain.space_cardinality())])
        self.__boolean_mask_number_of_ones_for_each_number: Dict[int, np.ndarray] = {t: self.__number_of_ones_for_each_number <= t for t in range(self.__domain.number_of_bits() + 1)}

    def domain(self) -> FullBinaryDomain:
        return self.__domain

    def number_of_ones_for_each_number(self) -> np.ndarray:
        return self.__number_of_ones_for_each_number

    def boolean_mask_number_of_ones_for_each_number(self, t: int) -> np.ndarray:
        return self.__boolean_mask_number_of_ones_for_each_number[t]

    def resiliency(self, spectrum: np.ndarray) -> int:
        max_resiliency_found_so_far: int = -1
        t: int = 0
        if spectrum[0] == 0:
            max_resiliency_found_so_far = 0
            for iii in range(self.__domain.number_of_bits()):
                t += 1
                m: np.ndarray = self.boolean_mask_number_of_ones_for_each_number(t)
                if np.any(spectrum[m]):
                    return max_resiliency_found_so_far
                else:
                    max_resiliency_found_so_far += 1
                    if iii == self.__domain.number_of_bits() - 1:
                        return max_resiliency_found_so_far
        else:
            return max_resiliency_found_so_far

    def non_linearity(self, spectrum: np.ndarray) -> float:
        x: np.ndarray = np.absolute(spectrum)
        m: float = float(np.max(x))
        return (2 ** (self.__domain.number_of_bits() - 1)) - 0.5 * m

    def apply(self, result: np.ndarray) -> np.ndarray:
        #res: np.ndarray = np.tile(result, (self.__domain.space_cardinality(), 1))
        #res = np.logical_xor(res, self.__pairwise_scalar_prod)
        #res = np.power(-1, res)
        #return np.sum(res, axis=1)
        return self.__fast_walsh_transform_init(result)

    def invert(self, spectrum: np.ndarray, directly_go_to_truth_table: bool = False) -> np.ndarray:
        return self.__inverse_fast_walsh_transform_init(spectrum, directly_go_to_truth_table)

    def __fast_walsh_transform_init(self, result: np.ndarray) -> np.ndarray:
        polar_form: np.ndarray = FullBinaryDomain.convert_truth_table_to_polar_form(result)
        l: List[int] = polar_form.tolist()
        _ = self.__fast_walsh_transform(l, 0, len(l))
        return np.array(l)

    def __fast_walsh_transform(self, v: List[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            temp: int = v[i]
            v[i] += v[i + half]
            v[i + half] = temp - v[i + half]

        if half > 1:
            val1: int = self.__fast_walsh_transform(v, start, half)
            val2: int = self.__fast_walsh_transform(v, start + half, half)
            return max(val1, val2)
        else:
            if abs(v[start]) > abs(v[start + half]):
                return abs(v[start])
            else:
                return abs(v[start + half])

    def __inverse_fast_walsh_transform_init(self, result: np.ndarray, directly_go_to_truth_table: bool = False) -> np.ndarray:
        l: List[int] = result.tolist()
        _ = self.__inverse_fast_walsh_transform(l, 0, len(l))
        r: np.ndarray = np.array(l, dtype=np.int32)
        if not directly_go_to_truth_table:
            return r
        return FullBinaryDomain.convert_polar_form_to_truth_table(r)

    def __inverse_fast_walsh_transform(self, v: List[int], start: int, length: int) -> int:
        half: int = length // 2
        for i in range(start, start + half):
            temp: int = v[i]
            v[i] = (v[i] + v[i + half]) // 2
            v[i + half] = (temp - v[i + half]) // 2

        if half > 1:
            val1: int = self.__inverse_fast_walsh_transform(v, start, half)
            val2: int = self.__inverse_fast_walsh_transform(v, start + half, half)
            return max(val1, val2)
        else:
            if start == 0:
                return abs(v[1])
            else:
                if abs(v[start]) > abs(v[start + half]):
                    return abs(v[start])
                else:
                    return abs(v[start + half])

    @staticmethod
    def linear_function_xor_of_bitwise_and(data: np.ndarray) -> np.ndarray:
        res: np.ndarray = np.empty((data.shape[0], data.shape[0]))
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                first: np.ndarray = data[i]
                second: np.ndarray = data[j]
                and_res: np.ndarray = np.logical_and(first, second)
                final_res: int = and_res[0]
                for iii in range(1, len(and_res)):
                    final_res = final_res ^ and_res[iii]
                res[i][j] = final_res
        return res
