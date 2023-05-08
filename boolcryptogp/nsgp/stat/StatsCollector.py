from typing import Dict, List, Any
import numpy as np
import pandas as pd


class StatsCollector:
    def __init__(self,
                 objective_names: List[str],
                 revert_sign: List[bool] = None
                 ) -> None:
        self.__data: Dict[int, Dict[str, np.ndarray]] = {}
        self.__perc_uncert_pos: Dict[int, Dict[str, float]] = {}
        self.__perc_diff_pos: Dict[int, Dict[str, float]] = {}
        self.__objective_names: List[str] = objective_names
        self.__revert_sign: List[bool] = revert_sign
        if self.__revert_sign is None:
            self.__revert_sign = [False] * len(self.__objective_names)
        if len(self.__objective_names) != len(self.__revert_sign):
            raise AttributeError(f"The length of objective names ({len(self.__objective_names)}) is not equal to the length of revert sign ({len(self.__revert_sign)}).")
        self.__sign_array: np.ndarray = np.array([-1.0 if b else 1.0 for b in self.__revert_sign])

    def update_fitness_stat_dict(self, n_gen: int, data: np.ndarray, perc_uncert_pos: np.ndarray, perc_diff_pos: np.ndarray) -> None:
        da: np.ndarray = data * np.tile(self.__sign_array, (data.shape[0], 1))
        d: Dict[str, np.ndarray] = {"mean": np.mean(da, axis=0),
                                    "median": np.median(da, axis=0),
                                    "min": np.min(da, axis=0),
                                    "max": np.max(da, axis=0),
                                    "sum": np.sum(da, axis=0),
                                    "std": np.std(da, axis=0)}
        self.__data[n_gen] = d

        dd: Dict[str, float] = {"mean": np.mean(perc_uncert_pos),
                                "median": np.median(perc_uncert_pos),
                                "min": np.min(perc_uncert_pos),
                                "max": np.max(perc_uncert_pos),
                                "sum": np.sum(perc_uncert_pos),
                                "std": np.std(perc_uncert_pos)}
        self.__perc_uncert_pos[n_gen] = dd

        ddd: Dict[str, float] = {"mean": np.mean(perc_diff_pos),
                                "median": np.median(perc_diff_pos),
                                "min": np.min(perc_diff_pos),
                                "max": np.max(perc_diff_pos),
                                "sum": np.sum(perc_diff_pos),
                                "std": np.std(perc_diff_pos)}
        self.__perc_diff_pos[n_gen] = ddd

    def get_fitness_stat(self, n_gen: int, stat: str) -> np.ndarray:
        return self.__data[n_gen][stat]

    def get_perc_uncert_pos_stat(self, n_gen: int, stat: str) -> float:
        return self.__perc_uncert_pos[n_gen][stat]

    def get_perc_diff_pos_stat(self, n_gen: int, stat: str) -> float:
        return self.__perc_diff_pos[n_gen][stat]

    def build_dataframe(self) -> pd.DataFrame:
        d: Dict[str, List[Any]] = {"Generation": [],
                                   "Objective": [],
                                   "Statistic": [],
                                   "Value": []}
        for n_gen in self.__data:
            dd: Dict[str, np.ndarray] = self.__data[n_gen]
            perc_uncert_pos: Dict[str, float] = self.__perc_uncert_pos[n_gen]
            perc_diff_pos: Dict[str, float] = self.__perc_diff_pos[n_gen]
            for stat in dd:
                val: np.ndarray = dd[stat]
                for i in range(len(val)):
                    objective_name: str = self.__objective_names[i]
                    value: float = val[i]
                    d["Generation"].append(n_gen)
                    d["Objective"].append(objective_name)
                    d["Statistic"].append(stat)
                    d["Value"].append(value)

                val_2: float = perc_uncert_pos[stat]
                d["Generation"].append(n_gen)
                d["Objective"].append("PercentageUncertainPositions")
                d["Statistic"].append(stat)
                d["Value"].append(val_2)

                val_3: float = perc_diff_pos[stat]
                d["Generation"].append(n_gen)
                d["Objective"].append("PercentageDiffPositions")
                d["Statistic"].append(stat)
                d["Value"].append(val_3)

        return pd.DataFrame(d)
