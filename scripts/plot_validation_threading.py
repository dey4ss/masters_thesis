#!/usr/bin/env python3

import os
import re

import pandas as pd
import numpy as np
import altair as alt
import math
import matplotlib.pyplot as plt
import altair_saver
import seaborn as sns

from collections import defaultdict
from matplotlib import rc
from palettable.cartocolors.qualitative import Antique_6, Bold_6, Pastel_6, Prism_6, Safe_6, Vivid_6

from grep_mining_time_per_config import MiningResult

def main():

    benchmarks = {"TPC-H": "TPCH", "TPC-DS": "TPCDS", "JOB": "JoinOrder"}
    run_times = defaultdict(list)

    max_validation_times = defaultdict(list)
    min_validation_times = defaultdict(float)



    regexes = [r'\d+(?=\ss)', r'\d+(?=\sms)', r'\d+(?=\sµs)', r'\d+(?=\sns)']
    divs = list(reversed([1, 10**3, 10**6, 10**9]))
    directory = "benchmark_threads"
    num_validator_regex = r'\d+(?=-validators)'

    for b in sorted(benchmarks.keys()):
        benchmark = benchmarks[b]
        print(b)
        benchmark_files = [f for f in os.listdir(directory) if benchmark in f]
        for file_name in benchmark_files:
            r = re.search(num_validator_regex, file_name)
            num_validators = 1 if r is None else int(r.group(0))

            with open(os.path.join(directory, file_name)) as f:
                current_dep = ""
                for line in f:
                    string = line.strip()
                    is_valid = MiningResult.valid_indicator() in line
                    is_invalid = MiningResult.invalid_indicator() in line
                    if  is_valid or is_invalid:
                        res = 0
                        for regex, div in zip(regexes, divs):
                            r = re.search(regex, string)
                            if not r:
                                continue
                            t = int(r.group(0))
                            res += t * div
                        valid_time = res / divs[1]
                        if len(max_validation_times[b]) == 0 or max_validation_times[b][0] < valid_time:
                            max_validation_times[b] = [valid_time, current_dep]

                    if MiningResult.mining_time_indicator() in string:
                        res = 0
                        for regex, div in zip(regexes, divs):
                            r = re.search(regex, string)
                            if not r:
                                continue
                            t = int(r.group(0))
                            res += t * div
                        run_times["Benchmark"].append(b)
                        run_times["Validators"].append(num_validators)
                        val_time = res / divs[0]
                        run_times["t"].append(val_time)

                        if min_validation_times[b] == 0 or min_validation_times[b] > val_time:
                            min_validation_times[b] = val_time


                    if MiningResult.candidates_indicator() in string:
                        # print(f"'{MiningResult.candidates_indicator()}'", line)
                        current_dep = string

    df = pd.DataFrame(data=run_times)

    sns.set()
    sns.set_theme(style="whitegrid")
    plt.style.use('seaborn-colorblind')
    plt.rcParams["figure.figsize"] = [x * 1 for x  in plt.rcParams["figure.figsize"]]
    plt.rcParams["font.family"] = "serif"

    for b, r in max_validation_times.items():
        print(b, r, min_validation_times[b])



    palette = Safe_6.hex_colors[:len(benchmarks)]
    palette = {b: c for b, c in zip(sorted(list(benchmarks.keys())), palette)}
    markers = ["^", "X", "s", "D", ".", "o"]


    sns.lineplot(data=df, x="Validators", y="t", style="Benchmark", markers=markers[:len(benchmarks)], hue="Benchmark", dashes=None, palette=palette, linewidth=2, markersize=9)
    #sns.lineplot(x=x_values, y=benchmark_gains, label="Latency Improvement", marker=markers_per_config[config], color=sns.desaturate(colors_per_config[config], 0.7), dashes=[(2,2)])
    plt.xlabel("Dependency Validators", fontsize=16)
    y_label = "Run-time [s]"
    plt.ylabel(y_label, fontsize=16)
    # plt.title(f"{title}")
    min_y = min(run_times["t"])
    if min_y >= 0:
        plt.axis(ymin=0)
    plt.axis(xmin=0)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.tight_layout(pad=0)

    plt.savefig(f"compare_validation_threading.eps", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
