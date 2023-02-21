from typing import List, Dict, Callable, Tuple, Any

from genepro.variation import generate_random_tree, safe_subtree_mutation, safe_subtree_crossover_two_children, \
    generate_random_forest, safe_subforest_mutation, safe_subforest_one_point_crossover_two_children

from genepro.node_impl import *
from genepro.node import Node

from genepro.util import compute_linear_model_discovered_in_math_formula_interpretability_paper, \
    concatenate_nodes_with_binary_operator
from boolcryptogp.nsgp.encoder.TreeEncoder import TreeEncoder

from copy import deepcopy


class TreeStructure:
    def __init__(self, operators: List[Node],
                 n_features: int,
                 max_depth: int,
                 constants: List[Constant] = None,
                 ephemeral_func: Callable = None,
                 normal_distribution_parameters: List[Tuple[float, float]] = None, p: List[float] = None):
        self.__p: List[float] = p
        self.__size: int = len(operators) + n_features + 1
        if normal_distribution_parameters is not None:
            if len(normal_distribution_parameters) != self.__size:
                raise AttributeError("The number of elements in normal distribution parameters must be equal to size (num_operators + num_features + 1).")
            self.__normal_distribution_parameters: List[Tuple[float, float]] = deepcopy(normal_distribution_parameters)
        else:
            self.__normal_distribution_parameters: List[Tuple[float, float]] = None
        self.__symbols: List[str] = [str(op.symb) for op in operators]
        self.__operators: List[Node] = deepcopy(operators)
        self.__n_operators: int = len(operators)
        if self.__p is None:
            self.__p = [1.0/float(self.__n_operators)] * self.__n_operators
        if self.__n_operators != len(self.__p):
            raise AttributeError(f"The length of probability distribution for internal nodes p is {len(self.__p)} but the number of operators is {self.__n_operators}. These two numbers must be equal.")
        self.__n_features: int = n_features
        self.__features: List[Feature] = [Feature(i) for i in range(n_features)]
        self.__max_depth: int = max_depth
        self.__n_layers: int = max_depth + 1
        self.__max_arity: int = max([int(op.arity) for op in operators])
        self.__max_n_nodes: int = int((self.__max_arity ** self.__n_layers - 1)/float(self.__max_arity - 1))
        self.__constants: List[Constant] = []
        if constants is not None:
            self.__constants = deepcopy(constants)
        self.__ephemeral_func: Callable = ephemeral_func
        self.__n_constants: int = len(self.__constants)
        self.__terminals: List[Node] = self.__features + self.__constants
        self.__n_terminals: int = len(self.__terminals) + (1 if self.__ephemeral_func is not None else 0)

        self.__encoding_func_dict: Dict[str, TreeEncoder] = {}

    def get_p(self) -> List[float]:
        return deepcopy(self.__p)

    def get_encoding_type_strings(self) -> List[str]:
        return list(self.__encoding_func_dict.keys())

    def get_normal_distribution_parameters(self) -> List[Tuple[float, float]]:
        if self.__normal_distribution_parameters is None:
            raise ValueError("Normal distribution parameters have not been set yet.")
        return deepcopy(self.__normal_distribution_parameters)

    def set_normal_distribution_parameters(self, normal_distribution_parameters: List[Tuple[float, float]] = None) -> None:
        if normal_distribution_parameters is not None:
            if len(normal_distribution_parameters) != self.__size:
                raise AttributeError("The number of elements in normal distribution parameters must be equal to size (num_operators + num_features + 1).")
            self.__normal_distribution_parameters: List[Tuple[float, float]] = deepcopy(normal_distribution_parameters)
        else:
            self.__normal_distribution_parameters: List[Tuple[float, float]] = None

    def __sample_weight(self, idx: int) -> float:
        if not (0 <= idx < self.__size):
            raise IndexError(f"{idx} is out of range as size.")
        return np.random.normal(self.__normal_distribution_parameters[idx][0], self.__normal_distribution_parameters[idx][1])

    def sample_operator_weight(self, idx: int) -> float:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__sample_weight(idx)

    def sample_feature_weight(self, idx: int) -> float:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__sample_weight(self.get_number_of_operators() + idx)

    def sample_constant_weight(self) -> float:
        return self.__sample_weight(self.__size - 1)

    def get_symbol(self, idx: int) -> str:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of symbols.")
        return self.__symbols[idx]

    def get_operator(self, idx: int) -> Node:
        if not (0 <= idx < self.get_number_of_operators()):
            raise IndexError(f"{idx} is out of range as index of operators.")
        return self.__operators[idx]

    def get_feature(self, idx: int) -> Feature:
        if not (0 <= idx < self.get_number_of_features()):
            raise IndexError(f"{idx} is out of range as index of features.")
        return self.__features[idx]

    def get_constant(self, idx: int) -> Constant:
        if not (0 <= idx < self.get_number_of_constants()):
            raise IndexError(f"{idx} is out of range as index of constants.")
        return self.__constants[idx]

    def sample_ephemeral_random_constant(self) -> float:
        if self.__ephemeral_func is None:
            raise AttributeError("Ephemeral function has not been defined in the constructor of this instance.")
        return self.__ephemeral_func()

    def get_number_of_operators(self) -> int:
        return self.__n_operators

    def get_number_of_features(self) -> int:
        return self.__n_features

    def get_number_of_constants(self) -> int:
        return self.__n_constants

    def get_number_of_terminals(self) -> int:
        return self.__n_terminals

    def get_max_depth(self) -> int:
        return self.__max_depth

    def get_max_arity(self) -> int:
        return self.__max_arity

    def get_max_n_nodes(self) -> int:
        return self.__max_n_nodes

    def get_number_of_layers(self) -> int:
        return self.__n_layers

    def get_size(self) -> int:
        return self.__size

    def generate_tree(self) -> Node:
        return generate_random_tree(self.__operators, self.__terminals, max_depth=self.get_max_depth(),
                                    curr_depth=0, ephemeral_func=self.__ephemeral_func, p=self.__p)

    def safe_subtree_mutation(self, tree: Node) -> Node:
        return safe_subtree_mutation(tree, self.__operators, self.__terminals, max_depth=self.__max_depth,
                                     ephemeral_func=self.__ephemeral_func, p=self.__p)

    def safe_subtree_crossover_two_children(self, tree_1: Node, tree_2: Node) -> Tuple[Node, Node]:
        return safe_subtree_crossover_two_children(tree_1, tree_2, max_depth=self.__max_depth)

    def get_dict_representation(self, tree: Node) -> Dict[int, str]:
        return tree.get_dict_repr(self.get_max_arity())

    def register_encoder(self, encoder: TreeEncoder) -> None:
        if encoder.get_name() in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoder.get_name()} already exists as key of the dictionary of encodings in this tree structure.")
        self.__encoding_func_dict[encoder.get_name()] = encoder

    def register_encoders(self, encoders: List[TreeEncoder]) -> None:
        names = []
        for e in encoders:
            names.append(e.get_name())
            if e.get_name() in self.__encoding_func_dict.keys():
                raise AttributeError(f"{e.get_name()} already exists as key of the dictionary of encodings in this tree structure.")
        if len(names) != len(list(set(names))):
            raise AttributeError(f"Names of the input encoders must all be distinct.")
        for e in encoders:
            self.register_encoder(e)

    def unregister_encoder(self, encoding_type: str) -> None:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        self.__encoding_func_dict.pop(encoding_type)

    def get_encoder(self, encoding_type: str) -> TreeEncoder:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type]

    def generate_encoding(self, encoding_type: str, tree: Node, apply_scaler: bool = True) -> np.ndarray:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].encode(tree, apply_scaler)

    def scale_encoding(self, encoding_type: str, encoding: np.ndarray) -> np.ndarray:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].scale(encoding)

    def get_scaler_on_encoding(self, encoding_type: str) -> Any:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].get_scaler()

    def get_encoding_size(self, encoding_type: str) -> int:
        if encoding_type not in self.__encoding_func_dict.keys():
            raise AttributeError(f"{encoding_type} is not a valid encoding type.")
        return self.__encoding_func_dict[encoding_type].size()

    @staticmethod
    def calculate_linear_model_discovered_in_math_formula_interpretability_paper(tree: Node,
                                                                                 difficult_operators: List[str] = None) -> float:
        return compute_linear_model_discovered_in_math_formula_interpretability_paper(tree, difficult_operators)

    @staticmethod
    def concatenate_nodes_with_binary_operator(forest: List[Node], binary_operator: Node, copy_tree: bool = False) -> Node:
        return concatenate_nodes_with_binary_operator(forest=forest, binary_operator=binary_operator,
                                                      copy_tree=copy_tree)

    def generate_forest(self, n_trees: int = None, n_trees_min: int = 2, n_trees_max: int = 10, tree_prob: float = 0.70) -> List[Node]:
        return generate_random_forest(internal_nodes=self.__operators, leaf_nodes=self.__terminals,
                                      max_depth=self.get_max_depth(), ephemeral_func=self.__ephemeral_func,
                                      p=self.__p, n_trees=n_trees, n_trees_min=n_trees_min, n_trees_max=n_trees_max,
                                      tree_prob=tree_prob)

    def safe_subforest_mutation(self, forest: List[Node]) -> List[Node]:
        return safe_subforest_mutation(forest, internal_nodes=self.__operators, leaf_nodes=self.__terminals,
                                       max_depth=self.get_max_depth(),
                                       ephemeral_func=self.__ephemeral_func, p=self.__p)

    @staticmethod
    def safe_subforest_one_point_crossover_two_children(forest_1: List[Node], forest_2: List[Node], max_length: int = None) -> Tuple[List[Node], List[Node]]:
        return safe_subforest_one_point_crossover_two_children(forest_1, forest_2, max_length=max_length)