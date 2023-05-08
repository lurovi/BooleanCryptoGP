import random
from typing import List, Dict, Callable, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['axes.formatter.use_mathtext'] = True
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math


class PlotUtils:

    '''
    @staticmethod
    def print_non_linearity_zero_balancing_resiliency_max_degree_all_p_values_combo(folder_name: str,
                                                                                    n_bits_list: List[int],
                                                                                    max_depth: int,
                                                                                    force_bent: bool,
                                                                                    binary_balancing: bool,
                                                                                    seed_list: List[int]) -> None:
        all_pop_gen: List[Tuple[int, int]] = [(100, 10), (1000, 25)]
        macro_methods: List[str] = ["GPNSGA2", "RANDOM"]
        filling_criteria: List[str] = ["Balancing", "Random"]

        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        all_sample_size: List[int] = [pop_size * num_gen for pop_size, num_gen in all_pop_gen]
        all_keys: List[str] = ["RANDOM_" + str(i) for i in all_sample_size]
        all_keys = all_keys + ["GPNSGA2_" + str(pop_size) + "_" + str(num_gen) for pop_size, num_gen in all_pop_gen]
        non_linearity: Dict[str, Dict[int, Dict[str, List[int]]]] = {k: {} for k in all_keys}
        all_methods: List[str] = list(non_linearity.keys())
        for n_bits in n_bits_list:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    for method in all_methods:
                        if method.startswith("RANDOM"):
                            curr_sample_size: int = int(method[7:])
                            path: str = f"best-pseudobooleanfunctionsRANDOM-{n_bits}bit-popsize{curr_sample_size}-numgen{0}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                        elif method.startswith("GPNSGA2"):
                            curr_sample_size: List[str] = method[7:].split("_")[1:]
                            curr_pop_size: int = int(curr_sample_size[0])
                            curr_num_gen: int = int(curr_sample_size[1])
                            path: str = f"best-pseudobooleanfunctionsGPNSGA2-{n_bits}bit-popsize{curr_pop_size}-numgen{curr_num_gen}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                        else:
                            raise AttributeError(f"Method not recognized.")
                        curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                        curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                        curr_data = curr_data[["Degree", "Balancing", "NonLinearity", "Resiliency"]]
                        curr_data = curr_data[curr_data["Balancing"] == 0]
                        curr_data = curr_data[curr_data["Resiliency"] == 0]
                        curr_data = curr_data[curr_data["Degree"] == -(n_bits - 1)]
                        curr_data.drop_duplicates(inplace=True, ignore_index=True)
                        uncertainty_filling: str = "Balancing" if make_it_balanced_int else "Random"
                        if n_bits not in non_linearity[method]:
                            non_linearity[method][n_bits] = {"Balancing": [], "Random": []}
                        if len(curr_data) > 0:
                            curr_non_lin: int = abs(curr_data.at[0, "NonLinearity"])
                        else:
                            curr_non_lin: int = 0
                        non_linearity[method][n_bits][uncertainty_filling].append(curr_non_lin)

        print("P-value")

        s_pr: str = ""
        for n_i in range(len(n_bits_list)):
            n: int = n_bits_list[n_i]
            s_pr += "\\midrule\n"

            for macro_method_i in range(len(macro_methods)):
                macro_method: str = macro_methods[macro_method_i]

                for t_i in range(len(all_pop_gen)):
                    pop_size: int = all_pop_gen[t_i][0]
                    num_gen: int = all_pop_gen[t_i][1]

                    for uncertainty_filling_i in range(len(filling_criteria)):
                        uncertainty_filling: str = filling_criteria[uncertainty_filling_i]

                        if macro_method_i == 0 and t_i == 0 and uncertainty_filling_i == 0:
                            s_pr += "\\multirow{8}{*}{" + str(n) + "} "
                        else:
                            s_pr += " "
                        if t_i == 0 and uncertainty_filling_i == 0:
                            s_pr += "& \\multirow{4}{*}{" + ("NSGA-2" if macro_method == "GPNSGA2" else "RANDOM") + "} "
                        else:
                            s_pr += "& "
                        if uncertainty_filling_i == 0:
                            tmp: str = ""
                            if macro_method == "RANDOM":
                                tmp = str(pop_size * num_gen)
                            elif macro_method == "GPNSGA2":
                                tmp = str(pop_size) + " * " + str(num_gen)
                            s_pr += "& \\multirow{2}{*}{" + tmp + "} "
                        else:
                            s_pr += "& "
                        s_pr += "& {" + ("Unbiased" if uncertainty_filling == "Random" else "Biased") + "} "
                        aaa: List[int] = non_linearity[macro_method + "_" + (
                            str(pop_size) + "_" + str(num_gen) if macro_method == "GPNSGA2" else str(
                                pop_size * num_gen))][n][uncertainty_filling]

                        for macro_method_j in range(len(macro_methods)):
                            macro_method_2: str = macro_methods[macro_method_j]

                            for t_j in range(len(all_pop_gen)):
                                pop_size_2: int = all_pop_gen[t_j][0]
                                num_gen_2: int = all_pop_gen[t_j][1]

                                for uncertainty_filling_j in range(len(filling_criteria)):
                                    uncertainty_filling_2: str = filling_criteria[uncertainty_filling_j]
                                    bbb: List[int] = non_linearity[macro_method_2 + "_" + (
                                        str(pop_size_2) + "_" + str(num_gen_2) if macro_method_2 == "GPNSGA2" else str(
                                            pop_size_2 * num_gen_2))][n][uncertainty_filling_2]
                                    if aaa == bbb:
                                        s_pr += "& - "
                                    else:
                                        p_value: str = str(
                                            round(stats.wilcoxon(aaa, bbb, alternative="greater").pvalue, 2))
                                        s_pr += "& " + p_value + " "
                        s_pr += "\\\\"
                        s_pr += "\n"

        print(s_pr)

    @staticmethod
    def big_stat_plot_relplot(folder_name: str,
                      n_bits_range: List[int],
                      pop_size: int,
                      num_gen: int,
                      max_depth: int,
                      force_bent: bool,
                      binary_balancing: bool,
                      seed_list: List[int]) -> sns.FacetGrid:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        dataframes: List[pd.DataFrame] = []
        for n_bits in n_bits_range:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    path: str = f"stat-pseudobooleanfunctionsGPNSGA2-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                    curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                    curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                    curr_data["Bit"] = n_bits
                    curr_data["Uncertainty Filling"] = "Balancing" if make_it_balanced_int else "Random"
                    dataframes.append(curr_data)
        data: pd.DataFrame = pd.concat(dataframes, ignore_index=True)

        data = data[data["Statistic"] != "std"]
        data = data[data["Bit"] >= 7]
        data = data[data["Objective"] == "Balancing"]
        data = data[data["Statistic"] == 'mean']
        data["Statistic"] = data["Statistic"].apply(lambda x: x[0].upper() + x[1:])
        sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=2.5,
                      rc={'figure.figsize': (8, 8), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
        g = sns.relplot(data=data, x="Generation", y="Value", style="Uncertainty Filling", col="Bit", col_wrap=5,
                        kind="line", palette="colorblind")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Mean balancedness trend over the evolution')
        plt.savefig("/home/luigi/Desktop/img/" + "MeanBalancing-every-bit.png")
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        return g

    @staticmethod
    def stat_plot_2(folder_name: str,
                  n_bits: int,
                  pop_size: int,
                  num_gen: int,
                  max_depth: int,
                  force_bent: bool,
                  binary_balancing: bool,
                  seed_list: List[int],
                  objective: str
                  ) -> sns.FacetGrid:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        dataframes: List[pd.DataFrame] = []
        for seed in seed_list:
            for make_it_balanced_int in [1, 0]:
                path: str = f"stat-pseudobooleanfunctionsGPNSGA2-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                curr_data["Uncertainty Filling"] = "Balancing" if make_it_balanced_int else "Random"
                dataframes.append(curr_data)
        data: pd.DataFrame = pd.concat(dataframes, ignore_index=True)

        data = data[data["Statistic"] != "std"]
        data = data[data["Objective"] == objective]
        data["Statistic"] = data["Statistic"].apply(lambda x: x[0].upper() + x[1:])
        sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=2.6,
                      rc={'figure.figsize': (13, 13), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
        # g = sns.relplot(data=data, x="Generation", y="Value", hue="Statistic", col="Objective", kind="line", palette="colorblind")
        g = sns.lineplot(data=data, x="Generation", y="Value", hue="Statistic", style="Uncertainty Filling", palette="colorblind")
        #g.set_title(objective + " Trend Over the Evolution When n = "+str(n_bits))
        #fig = g.get_figure()
        #fig.savefig("/home/luigi/Desktop/img/"+objective+"-"+str(n_bits)+"bit.png")
        if objective == "NonLinearity":
            plt.title("Non-linearity" + " trend over the evolution when n = "+str(n_bits))
        else:
            plt.title(objective + " trend over the evolution when n = " + str(n_bits))
        plt.legend(fontsize='small', title_fontsize='small', loc="center right")
        plt.savefig("/home/luigi/Desktop/img/" + objective + "-" + str(n_bits) + "bit.png")
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        return g

    @staticmethod
    def stat_plot_3(folder_name: str,
                    n_bits_list: List[int],
                    pop_size: int,
                    num_gen: int,
                    max_depth: int,
                    make_it_balanced: bool,
                    force_bent: bool,
                    binary_balancing: bool,
                    seed_list: List[int],
                    objective: str
                    ) -> sns.FacetGrid:
        make_it_balanced_int: int = int(make_it_balanced)
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        dataframes: List[pd.DataFrame] = []
        for n_bits in n_bits_list:
            for seed in seed_list:
                path: str = f"stat-pseudobooleanfunctionsGPNSGA2-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                curr_data["Bit"] = n_bits
                dataframes.append(curr_data)
        data: pd.DataFrame = pd.concat(dataframes, ignore_index=True)

        data = data[data["Statistic"] != "std"]
        data = data[data["Objective"] == objective]
        data["Statistic"] = data["Statistic"].apply(lambda x: x[0].upper() + x[1:])
        sns.set_theme(font="STIXGeneral", palette="colorblind", style="whitegrid",
                      rc={'figure.figsize': (8, 8), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
        # g = sns.relplot(data=data, x="Generation", y="Value", hue="Statistic", col="Objective", kind="line", palette="colorblind")
        g = sns.lineplot(data=data, x="Generation", y="Value", hue="Statistic", style="Bit",
                         palette="colorblind")
        # g.set_title(objective + " Trend Over the Evolution When n = "+str(n_bits))
        # fig = g.get_figure()
        # fig.savefig("/home/luigi/Desktop/img/"+objective+"-"+str(n_bits)+"bit.png")
        plt.title(objective + " trend over the evolution")
        # plt.yscale('log')
        plt.legend(fontsize='small', title_fontsize='small')
        plt.savefig("/home/luigi/Desktop/img/" + objective + "-more-bit.png")
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        return g

    @staticmethod
    def stat_plot_3_make_it_balanced(folder_name: str,
                    n_bits_list: List[int],
                    pop_size: int,
                    num_gen: int,
                    max_depth: int,
                    force_bent: bool,
                    binary_balancing: bool,
                    seed_list: List[int],
                    objective: str
                    ) -> sns.FacetGrid:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        dataframes: List[pd.DataFrame] = []
        for n_bits in n_bits_list:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    path: str = f"stat-pseudobooleanfunctionsGPNSGA2-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                    curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                    curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                    curr_data["Bit"] = n_bits
                    curr_data["Uncertainty Filling"] = "Balancing" if make_it_balanced_int else "Random"
                    dataframes.append(curr_data)
        data: pd.DataFrame = pd.concat(dataframes, ignore_index=True)

        data = data[data["Statistic"] != "std"]
        data = data[data["Objective"] == objective]
        data["Statistic"] = data["Statistic"].apply(lambda x: x[0].upper() + x[1:])
        sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.22,
                      rc={'figure.figsize': (8, 8), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
        g = sns.relplot(data=data, x="Generation", y="Value", hue="Statistic", style="Bit", col="Uncertainty Filling", kind="line", palette="colorblind")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(objective + " trend over the evolution")
        plt.savefig("/home/luigi/Desktop/img/" + objective + "-more-bit-uncertainty.png")
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()
        return g

    @staticmethod
    def plot_scatter_bi_objective_pareto_front(folder_name: str,
                                               n_bits: int,
                                               pop_size: int,
                                               num_gen: int,
                                               max_depth: int,
                                               make_it_balanced: bool,
                                               force_bent: bool,
                                               binary_balancing: bool,
                                               seed: int) -> None:
        make_it_balanced_int: int = int(make_it_balanced)
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        path: str = f"best-pseudobooleanfunctionsGPNSGA2-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
        data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
        data.drop("Unnamed: 0", inplace=True, axis=1)
        data = data[["Degree", "Balancing", "NonLinearity", "Resiliency"]]
        print(data.head(2000))
    '''

    @staticmethod
    def simple_stat_objective_plot(folder_name: str,
                                   method: str,
                                   n_bits: int,
                                   pop_size: int,
                                   num_gen: int,
                                   max_depth: int,
                                   pressure: int,
                                   make_it_balanced: bool,
                                   force_bent: bool,
                                   spectral_inversion: bool,
                                   dataset_type: str,
                                   nearest_bool_mapping: str,
                                   bent_mapping: str,
                                   binary_balancing: bool,
                                   non_linearity_only: bool,
                                   seed_list: List[int],
                                   objective: str) -> None:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        spectral_inversion_int: int = int(spectral_inversion)
        non_linearity_only_int: int = int(non_linearity_only)
        make_it_balanced_int: int = int(make_it_balanced)
        dataframes: List[pd.DataFrame] = []
        for seed in seed_list:
            path: str = f"stat-pseudobooleanfunctions{method}-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-pressure{pressure}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-spectralinversion{spectral_inversion_int}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{binary_balancing_int}-nonlinearityonly{non_linearity_only_int}-SEED{seed}.csv"
            curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
            curr_data.drop("Unnamed: 0", inplace=True, axis=1)
            dataframes.append(curr_data)
        data: pd.DataFrame = pd.concat(dataframes, ignore_index=True)

        data = data[data["Statistic"] != "std"]
        data = data[data["Statistic"] != "median"]
        #data = data[data["Statistic"] != "mean"]
        data = data[data["Statistic"] != "sum"]
        data = data[data["Objective"] == objective]
        data["Statistic"] = data["Statistic"].apply(lambda x: x[0].upper() + x[1:])
        sns.set_theme(font="STIXGeneral", palette="colorblind", style="whitegrid", font_scale=1.6,
                      rc={'figure.figsize': (8, 8), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
        g = sns.lineplot(data=data, x="Generation", y="Value", hue="Statistic", palette="colorblind")
        plt.title(objective)
        plt.legend(fontsize='small', title_fontsize='small')
        plt.savefig("/home/luigi/Desktop/codebase/python_data/BooleanCryptoGP/img/"+objective+"-"+str(n_bits)+"bit"+"-"+("Balancing" if make_it_balanced else "Random")+".png")
        # plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    @staticmethod
    def plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats(folder_name: str,
                                                                          method: str,
                                                                          n_bits_list: List[int],
                                                                          pop_size: int,
                                                                          num_gen: int,
                                                                          max_depth: int,
                                                                          pressure: int,
                                                                          force_bent: bool,
                                                                          spectral_inversion: bool,
                                                                          dataset_type: str,
                                                                          nearest_bool_mapping: str,
                                                                          bent_mapping: str,
                                                                          binary_balancing: bool,
                                                                          non_linearity_only: bool,
                                                                          seed_list: List[int]) -> None:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        spectral_inversion_int: int = int(spectral_inversion)
        non_linearity_only_int: int = int(non_linearity_only)
        non_linearity: Dict[int, Dict[str, List[int]]] = {}  # K1: n_bits, K2: UncertaintyFillingCriteria, V: list of non-linearity (one for each seed)
        balancedness: Dict[int, Dict[str, List[int]]] = {} # K1: n_bits, K2: UncertaintyFillingCriteria, V: list of balancedness (one for each seed)
        for n_bits in n_bits_list:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    path: str = f"best-pseudobooleanfunctions{method}-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-pressure{pressure}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-spectralinversion{spectral_inversion_int}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{binary_balancing_int}-nonlinearityonly{non_linearity_only_int}-SEED{seed}.csv"
                    curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                    curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                    curr_data = curr_data[["Degree", "Balancing", "NonLinearity", "Resiliency", "TruthTable"]]
                    #print()
                    #print(curr_data.head(1000))
                    #curr_data = curr_data[curr_data["Balancing"] == 0]
                    #curr_data = curr_data[curr_data["Resiliency"] == 0]
                    #curr_data = curr_data[curr_data["Degree"] == -(n_bits - 1)]
                    curr_data["Balancing"] = abs(curr_data["TruthTable"].apply(lambda x: x.count('1')) - curr_data["TruthTable"].apply(lambda x: x.count('0')))
                    curr_data.drop("TruthTable", inplace=True, axis=1)
                    curr_data.drop("Degree", inplace=True, axis=1)
                    curr_data.drop("Resiliency", inplace=True, axis=1)
                    #curr_data.drop_duplicates(inplace=True, ignore_index=True)
                    uncertainty_filling: str = "Balancing" if make_it_balanced_int else "Random"

                    if n_bits not in non_linearity:
                        non_linearity[n_bits] = {"Balancing": [], "Random": []}
                    if len(curr_data) > 0:
                        curr_non_lin: int = abs(curr_data.at[0, "NonLinearity"])
                    else:
                        curr_non_lin: int = 0
                    non_linearity[n_bits][uncertainty_filling].append(curr_non_lin)

                    if n_bits not in balancedness:
                        balancedness[n_bits] = {"Balancing": [], "Random": []}
                    if len(curr_data) > 0:
                        curr_balan: float = round(curr_data["Balancing"].mean(), 2)
                    else:
                        curr_balan: int = 1
                    balancedness[n_bits][uncertainty_filling].append(curr_balan)

        func_dict: Dict[str, Callable] = {"Max": np.max, "Mean": np.mean, "Std": np.std, "Min": np.min}
        for uncertainty_filling in ["Balancing", "Random"]:
            print()
            print(uncertainty_filling)
            for func in ["Max", "Mean", "Std", "Min"]:
                print()
                s = ""
                for n_bits in n_bits_list:
                    l: List[int] = balancedness[n_bits][uncertainty_filling]
                    stat: float = round(func_dict[func](l), 2)
                    #s += "\\num{"+str(int(stat) if func not in ("Mean", "Std") else stat)+"} & "
                    s += ""+str(int(stat) if func not in ("Mean", "Std") else stat)+" & "
                s = s[:-2]
                print(s)

        print("P-value")

        s = ""
        for n_bits in n_bits_list:
            b: List[int] = balancedness[n_bits]["Balancing"]
            r: List[int] = balancedness[n_bits]["Random"]
            p: float = round(stats.wilcoxon(b, r, alternative="less").pvalue, 2)
            s += ""+str(p)+" & "
        s = s[:-2]
        print(s)

        '''
        df: Dict[str, List[Any]] = {"Bit": [], "NonLinearity": [], "UncertaintyFilling": []}
        for n_bits in n_bits_list:
            if n_bits not in [12, 13, 14, 15, 16]:
                continue
            crb: int = 2 ** (n_bits - 1) - 2 ** ((n_bits // 2) - 1)
            for uncertainty_filling in ["Balancing", "Random"]:
                a: List[int] = non_linearity[n_bits][uncertainty_filling]
                for nl in a:
                    df["Bit"].append(n_bits)
                    df["UncertaintyFilling"].append(uncertainty_filling)
                    df["NonLinearity"].append(nl/crb)

        sns.set_theme(font="STIXGeneral", palette="colorblind", style="white", font_scale=1.50,
                      rc={'figure.figsize': (12, 8), 'pdf.fonttype': 42, 'ps.fonttype': 42,
                          'axes.formatter.use_mathtext': True, 'axes.unicode_minus': False})
        g = sns.boxplot(data=pd.DataFrame(df), x="Bit", y="NonLinearity", hue="UncertaintyFilling", orient="v")
        plt.savefig("/home/luigi/Desktop/img/Boxplot-more-bit.png")
        '''
        '''
        txt_data: Dict[str, List[Any]] = {"Bit": [], "UncertaintyFilling": [], "Min": [], "Q1": [], "Median": [], "Q3": [], "Max": []}
        for n_bits in n_bits_list:
            if n_bits not in [12, 13, 14, 15, 16]:
                continue
            crb: int = 2 * math.floor(2 ** (n_bits - 2) - 2 ** ((n_bits / 2) - 2))
            for uncertainty_filling in ["Balancing", "Random"]:
                a: List[int] = non_linearity[n_bits][uncertainty_filling]
                scaled_a: List[float] = [nl/crb for nl in a]
                txt_data["Bit"].append(n_bits)
                txt_data["UncertaintyFilling"].append(uncertainty_filling)
                txt_data["Min"].append(np.min(scaled_a))
                txt_data["Q1"].append(np.percentile(scaled_a, 25))
                txt_data["Median"].append(np.median(scaled_a))
                txt_data["Q3"].append(np.percentile(scaled_a, 75))
                txt_data["Max"].append(np.max(scaled_a))
        pd.DataFrame(txt_data).to_csv(path_or_buf="/home/luigi/Desktop/img/boxplot_data.txt", header=True, index=False, sep=" ")
        '''
        '''
        actual_bit_list: List[int] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        aa: List[str] = ["Balancing"+str(n_bits) for n_bits in actual_bit_list]
        bb: List[str] = ["Random" + str(n_bits) for n_bits in actual_bit_list]
        txt_data: Dict[str, List[Any]] = {k: [] for k in aa + bb}
        for n_bits in n_bits_list:
            if n_bits not in actual_bit_list:
                continue
            crb: int = 2 * math.floor(2 ** (n_bits - 2) - 2 ** ((n_bits / 2) - 2))
            for uncertainty_filling in ["Balancing", "Random"]:
                a: List[int] = non_linearity[n_bits][uncertainty_filling]
                scaled_a: List[float] = [nl / crb for nl in a]
                #if n_bits == 8:
                #    scaled_a = [nl + random.uniform(-9e-6, 9e-6) for nl in scaled_a]
                txt_data[uncertainty_filling+str(n_bits)].extend(scaled_a)
        pd.DataFrame(txt_data).to_csv(path_or_buf="/home/luigi/Desktop/img/boxplot_data_"+method+".dat", header=True, index=False, sep=" ")
        '''

    @staticmethod
    def plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats_all_algorithms(folder_name: str,
                                                                                         n_bits_list: List[int],
                                                                                         pop_size: int,
                                                                                         num_gen: int,
                                                                                         max_depth: int,
                                                                                         pressure: int,
                                                                                         force_bent: bool,
                                                                                         spectral_inversion: bool,
                                                                                         dataset_type: str,
                                                                                         nearest_bool_mapping: str,
                                                                                         bent_mapping: str,
                                                                                         binary_balancing: bool,
                                                                                         non_linearity_only: bool,
                                                                                         seed_list: List[int]) -> None:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        non_linearity_only_int: int = int(non_linearity_only)
        spectral_inversion_int: int = int(spectral_inversion)

        sample_size_initialization: int = pop_size
        sample_size_tot: int = pop_size * num_gen
        non_linearity: Dict[str, Dict[int, Dict[str, List[int]]]] = {"GPNSGA2": {},
                                                                     #"RANDOM"+str(sample_size_initialization): {},
                                                                     "RANDOM"+str(sample_size_tot): {}}
        all_methods: List[str] = sorted(list(non_linearity.keys()))
        for n_bits in n_bits_list:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    for method in all_methods:
                        if method.startswith("RANDOM"):
                            curr_sample_size: int = int(method[6:])
                            path: str = f"best-pseudobooleanfunctionsRANDOM-{n_bits}bit-popsize{curr_sample_size}-numgen{0}-maxdepth{max_depth}-pressure{0}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-spectralinversion{spectral_inversion_int}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{binary_balancing_int}-nonlinearityonly{non_linearity_only_int}-SEED{seed}.csv"
                        else:
                            path: str = f"best-pseudobooleanfunctions{method}-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-pressure{pressure}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-spectralinversion{spectral_inversion_int}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{binary_balancing_int}-nonlinearityonly{non_linearity_only_int}-SEED{seed}.csv"
                        curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                        curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                        curr_data = curr_data[["Degree", "Balancing", "NonLinearity", "Resiliency"]]
                        #curr_data = curr_data[curr_data["Balancing"] == 0]
                        #curr_data = curr_data[curr_data["Resiliency"] == 0]
                        #curr_data = curr_data[curr_data["Degree"] == -(n_bits - 1)]
                        curr_data.drop_duplicates(inplace=True, ignore_index=True)
                        uncertainty_filling: str = "Balancing" if make_it_balanced_int else "Random"
                        if n_bits not in non_linearity[method]:
                            non_linearity[method][n_bits] = {"Balancing": [], "Random": []}
                        if len(curr_data) > 0:
                            curr_non_lin: int = abs(curr_data.at[0, "NonLinearity"])
                        else:
                            print("boh")
                            curr_non_lin: int = 0
                        non_linearity[method][n_bits][uncertainty_filling].append(curr_non_lin)
        print("P-value")
        print()
        for uncertainty_filling in ["Balancing", "Random"]:
            print(uncertainty_filling)
            print()
            for n_bits in n_bits_list:
                aaa: List[int] = non_linearity["GPNSGA2"][n_bits][uncertainty_filling]
                bbb: List[int] = non_linearity["RANDOM"+str(sample_size_tot)][n_bits][uncertainty_filling]
                p_value: float = round(stats.wilcoxon(aaa, bbb, alternative="greater").pvalue, 2) if aaa != bbb else 1.0
                print(str(n_bits) + " : " + str(p_value))
            print()
        print("------------------------------------------------------------------------------------------------------")

        actual_bit_list: List[int] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        all_methods: List[str] = list(non_linearity.keys())
        aa: List[str] = ["Balancing" + str(n_bits) + method for method in all_methods for n_bits in actual_bit_list]
        bb: List[str] = ["Random" + str(n_bits) + method for method in all_methods for n_bits in actual_bit_list]
        txt_data: Dict[str, List[Any]] = {k: [] for k in aa + bb}
        for n_bits in n_bits_list:
            if n_bits not in actual_bit_list:
                continue
            crb: int = 2 * math.floor(2 ** (n_bits - 2) - 2 ** ((n_bits / 2.0) - 2))
            for uncertainty_filling in ["Balancing", "Random"]:
                for method in all_methods:
                    a: List[int] = non_linearity[method][n_bits][uncertainty_filling]
                    scaled_a: List[float] = [nl / crb for nl in a]
                    #if pop_size == 500 and num_gen == 5 and max_depth == 5 and pressure == 5:
                    #    if n_bits == 8 and uncertainty_filling == "Balancing":
                    #        scaled_a = [nl + random.uniform(-9e-7, 9e-7) for nl in scaled_a]
                    txt_data[uncertainty_filling + str(n_bits) + method].extend(scaled_a)
        pd.DataFrame(txt_data).to_csv(path_or_buf="/home/luigi/Desktop/codebase/python_data/BooleanCryptoGP/img/boxplot_data_" + "every_" + str(pop_size) + "_" + str(num_gen) + ".dat",
                                      header=True, index=False, sep=" ")

    @staticmethod
    def plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats_random_all_representations(folder_name: str,
                                                                                                     n_bits_list: List[int],
                                                                                                     pop_size: int,
                                                                                                     num_gen: int,
                                                                                                     max_depth: int,
                                                                                                     force_bent: bool,
                                                                                                     binary_balancing: bool,
                                                                                                     seed_list: List[int]) -> None:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        sample_size_initialization: int = pop_size
        sample_size_tot: int = pop_size * num_gen
        non_linearity: Dict[str, Dict[int, Dict[str, List[int]]]] = {"TRUTHTABLERANDOM" + str(sample_size_tot): {},
                                                                     "RANDOM" + str(sample_size_tot): {}}
        all_methods: List[str] = list(non_linearity.keys())
        for n_bits in n_bits_list:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    for method in all_methods:
                        if method.startswith("RANDOM"):
                            path: str = f"best-pseudobooleanfunctionsRANDOM-{n_bits}bit-popsize{sample_size_tot}-numgen{0}-maxdepth{max_depth}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                        else:
                            path: str = f"best-truthtablefunctionsRANDOM-{n_bits}bit-popsize{sample_size_tot}-numgen{0}-maxdepth{0}-makeitbalanced{make_it_balanced_int}-forcebent{0}-binarybalancing{binary_balancing_int}-SEED{seed}.csv"
                        curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                        curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                        curr_data = curr_data[["Degree", "Balancing", "NonLinearity", "Resiliency"]]
                        curr_data = curr_data[curr_data["Balancing"] == 0]
                        curr_data = curr_data[curr_data["Resiliency"] == 0]
                        curr_data = curr_data[curr_data["Degree"] == -(n_bits - 1)]
                        curr_data.drop_duplicates(inplace=True, ignore_index=True)
                        uncertainty_filling: str = "Balancing" if make_it_balanced_int else "Random"
                        if n_bits not in non_linearity[method]:
                            non_linearity[method][n_bits] = {"Balancing": [], "Random": []}
                        if len(curr_data) > 0:
                            curr_non_lin: int = abs(curr_data.at[0, "NonLinearity"])
                        else:
                            curr_non_lin: int = 0
                        non_linearity[method][n_bits][uncertainty_filling].append(curr_non_lin)
        print("P-value")
        print()
        for uncertainty_filling in ["Balancing", "Random"]:
            print(uncertainty_filling)
            print()
            for n_bits in n_bits_list:
                aaa: List[int] = non_linearity["TRUTHTABLERANDOM" + str(sample_size_tot)][n_bits][uncertainty_filling]
                bbb: List[int] = non_linearity["RANDOM" + str(sample_size_tot)][n_bits][uncertainty_filling]
                p_value: float = round(stats.wilcoxon(aaa, bbb, alternative="greater").pvalue, 2) if aaa != bbb else 1.0
                print(str(n_bits) + " : " + str(p_value))
            print()
        print("------------------------------------------------------------------------------------------------------")

        actual_bit_list: List[int] = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        all_methods: List[str] = list(non_linearity.keys())
        aa: List[str] = ["Balancing" + str(n_bits) + method for method in all_methods for n_bits in actual_bit_list]
        bb: List[str] = ["Random" + str(n_bits) + method for method in all_methods for n_bits in actual_bit_list]
        txt_data: Dict[str, List[Any]] = {k: [] for k in aa + bb}
        for n_bits in n_bits_list:
            if n_bits not in actual_bit_list:
                continue
            crb: int = 2 * math.floor(2 ** (n_bits - 2) - 2 ** ((n_bits / 2) - 2))
            for uncertainty_filling in ["Balancing", "Random"]:
                for method in all_methods:
                    a: List[int] = non_linearity[method][n_bits][uncertainty_filling]
                    scaled_a: List[float] = [nl / crb for nl in a]
                    txt_data[uncertainty_filling + str(n_bits) + method].extend(scaled_a)
        pd.DataFrame(txt_data).to_csv(
            path_or_buf="/home/luigi/Desktop/img/boxplot_data_" + "just_random_" + str(sample_size_tot) + ".dat",
            header=True, index=False, sep=" ")

    @staticmethod
    def check_pareto_front_values(folder_name: str,
                                  method: str,
                                  n_bits_list: List[int],
                                  pop_size: int,
                                  num_gen: int,
                                  max_depth: int,
                                  pressure: int,
                                  force_bent: bool,
                                  spectral_inversion: bool,
                                  dataset_type: str,
                                  nearest_bool_mapping: str,
                                  bent_mapping: str,
                                  binary_balancing: bool,
                                  non_linearity_only: bool,
                                  seed_list: List[int]) -> None:
        binary_balancing_int: int = int(binary_balancing)
        force_bent_int: int = int(force_bent)
        spectral_inversion_int: int = int(spectral_inversion)
        non_linearity_only_int: int = int(non_linearity_only)
        non_linearity: Dict[int, Dict[str, List[int]]] = {}
        for n_bits in n_bits_list:
            for seed in seed_list:
                for make_it_balanced_int in [1, 0]:
                    path: str = f"best-pseudobooleanfunctions{method}-{n_bits}bit-popsize{pop_size}-numgen{num_gen}-maxdepth{max_depth}-pressure{pressure}-makeitbalanced{make_it_balanced_int}-forcebent{force_bent_int}-spectralinversion{spectral_inversion_int}-datasettype_{dataset_type}-nearestboolmapping_{nearest_bool_mapping}-bentmapping_{bent_mapping}-binarybalancing{binary_balancing_int}-nonlinearityonly{non_linearity_only_int}-SEED{seed}.csv"
                    curr_data: pd.DataFrame = pd.read_csv(folder_name + "/" + path)
                    curr_data.drop("Unnamed: 0", inplace=True, axis=1)
                    curr_data = curr_data[["Degree", "Balancing", "NonLinearity", "Resiliency"]]
                    #curr_data = curr_data[curr_data["Balancing"] == 0]
                    #curr_data = curr_data[curr_data["Resiliency"] == 0]
                    #curr_data = curr_data[curr_data["Degree"] == -(n_bits - 1)]
                    curr_data = curr_data[["NonLinearity"]]
                    curr_data.drop_duplicates(inplace=True, ignore_index=True)
                    uncertainty_filling: str = "Balancing" if make_it_balanced_int else "Random"
                    if n_bits not in non_linearity:
                        non_linearity[n_bits] = {"Balancing": [], "Random": []}
                    #non_linearity[n_bits][uncertainty_filling].append(abs(curr_data.at[0, "NonLinearity"]))
                    if not make_it_balanced_int and n_bits == 6:
                        print(curr_data.head(100))
                        print()
                        print()


if __name__ == "__main__":
    folder: str = "/home/luigi/Desktop/codebase/python_data/BooleanCryptoGP/"
    exp_name: str = "results_1"
    #print(mpl.font_manager.get_font_names())
    '''
    s = ""
    for n in range(6, 16 + 1):
        #b: int = 2 ** (n - 1) - 2 ** ((n // 2) - 1)
        b: int = 2 * math.floor(2 ** (n - 2) - 2 ** ((n / 2) - 2))
        s += "\\num{" + str(round(b, 2)) + "} & "
    print(s)
    '''

    #b: List[float] = [24.6, 52.8, 110.0, 228.8, 470.2, 959.8, 1950.8, 3949.6, 7975.0, 16058.0, 32284.4]
    #r: List[float] = [24.0, 52.0, 110.0, 227.6, 468.2, 956.4, 1944.8, 3941.2, 7960.8, 16041.8, 32258.2]

    #b: List[float] = [26, 54, 110, 230, 472, 960, 1952, 3952, 7980, 16060, 32288]
    #r: List[float] = [24, 52, 110, 228, 470, 958, 1948, 3946, 7966, 16046, 32266]
    #print(stats.wilcoxon(b, r, alternative="greater"))

    #PlotUtils.stat_plot_3(folder+"results", [13, 14, 15, 16], 1000, 25, 5, True, False, True, list(range(1, 30 + 1)), "Degree")
    #PlotUtils.stat_plot_3_make_it_balanced(folder+"results", [13, 14, 15, 16], 1000, 25, 5, False, True, list(range(1, 30 + 1)), "Degree")

    #for n_bits in range(6, 16 + 1):
    #    for objective in ["Resiliency"]:
    #        PlotUtils.stat_plot_2(folder+"results", n_bits, 1000, 25, 5, False, True, list(range(1, 30 + 1)), objective)
    #        exit(1)
    #for i in range(1, 30+1):
    #    PlotUtils.plot_scatter_bi_objective_pareto_front(folder+"results", 10, 1000, 25, 5, True, False, True, i)
    #PlotUtils.big_stat_plot_relplot(folder+"results", list(range(6, 16 + 1)), 1000, 25, 5, False, True, list(range(1, 30 + 1)))
    #PlotUtils.plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats(folder+"results", "RANDOM", list(range(6, 16 + 1)), 1000, 0, 5, False, True, list(range(1, 30 + 1)))
    #PlotUtils.check_pareto_front_values(folder+"results", "RANDOM", list(range(6, 16 + 1)), 1000, 0, 5, False, True, list(range(1, 30 + 1)))
    #PlotUtils.print_non_linearity_zero_balancing_resiliency_max_degree_all_p_values_combo(folder+"results", list(range(6, 15 + 1)), 5, False, True, list(range(1, 30 + 1)))
    #PlotUtils.plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats_random_all_representations(folder+"results", list(range(6, 16 + 1)), 250, 4, 5, False, True, list(range(1, 30 + 1)))

    #PlotUtils.plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats(folder + exp_name, 'GPNSGA2', n_bits_list=list(range(6, 16 + 1)), pop_size=500, num_gen=5, max_depth=5, pressure=5, force_bent=False, spectral_inversion=True, dataset_type='integer', nearest_bool_mapping='pos_neg_zero', bent_mapping='pos_neg', binary_balancing=True, non_linearity_only=True, seed_list=list(range(1, 30 + 1)))
    #PlotUtils.plot_all_non_linearity_zero_balancing_resiliency_max_degree_stats_all_algorithms(folder + exp_name, n_bits_list=list(range(6, 16 + 1)), pop_size=500, num_gen=5, max_depth=5, pressure=5, force_bent=False, spectral_inversion=True, dataset_type='integer', nearest_bool_mapping='pos_neg_zero', bent_mapping='pos_neg', binary_balancing=True, non_linearity_only=True, seed_list=list(range(1, 30 + 1)))
    #for iii in [8, 9, 10]:
    #    for bbb in [True, False]:
    #        PlotUtils.simple_stat_objective_plot(folder+exp_name, "GPNSGA2", n_bits=iii, pop_size=500, num_gen=5, max_depth=5, pressure=5, make_it_balanced=bbb, force_bent=False, spectral_inversion=True, dataset_type='integer', nearest_bool_mapping='pos_neg_zero', bent_mapping='pos_neg', binary_balancing=True, non_linearity_only=True, seed_list=list(range(1, 30 + 1)), objective="PercentageUncertainPositions")
    PlotUtils.check_pareto_front_values(folder+exp_name, "GPNSGA2", n_bits_list=list(range(6, 16 + 1)), pop_size=500, num_gen=5, max_depth=5, pressure=5, force_bent=False, spectral_inversion=True, dataset_type='integer', nearest_bool_mapping='pos_neg_zero', bent_mapping='pos_neg', binary_balancing=True, non_linearity_only=True, seed_list=list(range(1, 30 + 1)))
