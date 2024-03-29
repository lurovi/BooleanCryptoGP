from typing import Any, TypeVar
from collections.abc import Callable

T = TypeVar('T')


class Sort:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def __heapify(arr: list[T], ind: list[int], comparator: Callable[[T, T], bool], n: int, i: int, reverse: bool) -> None:
        largest: int = i
        l: int = 2 * i + 1
        r: int = 2 * i + 2
        if not(reverse):
            if l < n and comparator(arr[i], arr[l]):
                largest = l

            if r < n and comparator(arr[largest], arr[r]):
                largest = r
        else:
            if l < n and not(comparator(arr[i], arr[l])):
                largest = l

            if r < n and not(comparator(arr[largest], arr[r])):
                largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            ind[i], ind[largest] = ind[largest], ind[i]
            Sort.__heapify(arr, ind, comparator, n, largest, reverse)

    @staticmethod
    def heapsort(v: list[T], comparator: Callable[[T, T], bool], inplace: bool = False, reverse: bool = False) -> tuple[list[T], list[int]]:
        n: int = len(v)
        ind: list[int] = list(range(n))
        if not(inplace):
            arr: list[T] = [x for x in v]
        else:
            arr: list[T] = v

        for i in range(n // 2 - 1, -1, -1):
            Sort.__heapify(arr, ind, comparator, n, i, reverse)

        for i in range(n - 1, 0, -1):
            arr[i], arr[0] = arr[0], arr[i]
            ind[i], ind[0] = ind[0], ind[i]
            Sort.__heapify(arr, ind, comparator, i, 0, reverse)

        return arr, ind
