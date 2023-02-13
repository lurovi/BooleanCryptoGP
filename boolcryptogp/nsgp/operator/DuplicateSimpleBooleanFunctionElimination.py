import numpy as np

from pymoo.core.duplicate import ElementwiseDuplicateElimination

from boolcryptogp.nsgp.structure.SimpleBooleanFunction import SimpleBooleanFunction


class DuplicateSimpleBooleanFunctionElimination(ElementwiseDuplicateElimination):
    def __init__(self) -> None:
        super().__init__()

    def is_equal(self, a, b) -> bool:
        a: SimpleBooleanFunction = a.X[0]
        b: SimpleBooleanFunction = b.X[0]
        return np.array_equal(a.output(), b.output())
