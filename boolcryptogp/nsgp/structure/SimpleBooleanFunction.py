from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from abc import ABC, abstractmethod
import numpy as np
from genepro.node import Node
from typing import List


class SimpleBooleanFunction(ABC):
    def __init__(self,
                 walsh: WalshTransform,
                 ) -> None:
        super().__init__()
        self.__walsh: WalshTransform = walsh

    def walsh(self) -> WalshTransform:
        return self.__walsh

    @abstractmethod
    def node(self) -> Node:
        pass

    @abstractmethod
    def forest(self) -> List[Node]:
        pass

    @abstractmethod
    def output(self) -> np.ndarray:
        pass
