from genepro.node import Node
import numpy as np
from genepro.node_impl import Feature, And, Xor
from genepro.util import concatenate_nodes_with_binary_operator
from typing import List
from boolcryptogp.nsgp.structure.WalshTransform import WalshTransform
from boolcryptogp.nsgp.structure.SimpleBooleanFunction import SimpleBooleanFunction


class ANFBooleanFunction(SimpleBooleanFunction):
    def __init__(self,
                 walsh: WalshTransform,
                 forest: List[Node]
                 ) -> None:
        super().__init__(walsh=walsh)
        fores: List[Node] = forest
        variables_str: List[List[str]] = sorted([sorted(list(set(n.retrieve_features_from_tree()))) for n in fores])
        variables: List[List[Node]] = [[Feature(int(var[2:])) for var in l] for l in variables_str]
        self.__forest: List[Node] = [l[0] if len(l) == 1 else concatenate_nodes_with_binary_operator(l, And(), copy_tree=False) for l in variables]
        self.__node: Node = concatenate_nodes_with_binary_operator(self.__forest, Xor(), copy_tree=False)
        result: np.ndarray = self.__node(self.walsh().domain().data())
        self.__output: np.ndarray = result

    def node(self) -> Node:
        return self.__node

    def forest(self) -> List[Node]:
        return self.__forest

    def output(self) -> np.ndarray:
        return self.__output
