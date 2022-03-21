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


def short(lst):
    # return [x[0] for x in lst]
    return lst

def collect_values(d, kw, bs, dirs):
    res = list()
    for b in bs:
        m = list()
        for i in list(dirs.keys()):
            m.append(sum(d[b][kw][i]) / 10**6)
        res.append(m)
    return res


def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ['c1', 'c2', 'values']
    df['Candidates'] = name
    return df


def use_b(b, dep):
    if b == "TPC-H" and dep == "ind":
        return False
    return True


def main():
    dirs = {"narrow": "benchmark_useful_candidates","wide": "benchmark_useless_candidates"}
    benchmarks = {"TPC-H": "TPCH", "TPC-DS": "TPCDS", "JOB": "JoinOrder"}
    run_times_ind = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    run_times_ucc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    run_times_od = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    deps_ind = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    deps_ucc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    deps_od = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


    run_times_all = {"Inclusion": run_times_ind, "Order": run_times_od, "Unique": run_times_ucc}
    deps_all = {"Inclusion": deps_ind, "Order": deps_od, "Unique": deps_ucc}

    deps_alias = {"IND": "Inclusion", "OD": "Order", "UCC": "Unique"}


    regexes = [r'\d+(?=\ss)', r'\d+(?=\sms)', r'\d+(?=\sµs)', r'\d+(?=\sns)']
    divs = list(reversed([1, 10**3, 10**6, 10**9]))
    type_regex = r'(?<=Type\s)\w+'
    name_regex = r'(?<=Columns\s).+'

    # version -> benchmark
    mining_summaries = defaultdict(dict)

    # benchmark -> validity -> impl -> run times
    for impl, directory in dirs.items():
        for b, benchmark in benchmarks.items():
            print(directory, benchmark)
            sf = "_s-10" if b != "JOB" else ""
            with open(os.path.join(directory, f"hyriseBenchmark{benchmark}_jts_join2pred_join_elim{sf}.log")) as f:
                current_type = ""
                current_dep = ""
                generation_time = 0
                num_candidates = 0
                num_valid = 0

                for line in f:
                    string = line.strip()
                    is_valid = MiningResult.valid_indicator() in line
                    is_invalid = MiningResult.invalid_indicator() in line
                    if is_valid or is_invalid:
                        res = 0
                        for regex, div in zip(regexes, divs):
                            r = re.search(regex, string)
                            if not r:
                                continue
                            t = int(r.group(0))
                            res += t * div

                        # result is ns, should be ms
                        #mining_results[config_name][benchmark] = res / 10**6
                        #if current_type ==

                        result_dict = run_times_all[current_type]
                        k = "valid" if is_valid else "invalid"

                        result_dict[b][k][impl].append(res)

                        result_dict = deps_all[current_type]
                        result_dict[b][k][impl].append((current_dep, res / 10**6))
                    if is_valid:
                        num_valid += 1

                    if "PQPAnalyzer generated " in string:
                        res = 0
                        for regex, div in zip(regexes, divs):
                            r = re.search(regex, string)
                            if not r:
                                continue
                            t = int(r.group(0))
                            res += t * div
                        generation_time = res


                    if MiningResult.candidates_indicator() in line:
                        # print(f"'{MiningResult.candidates_indicator()}'", line)
                        r = re.search(type_regex, string)
                        current_type = r.group(0)
                        r = re.search(name_regex, string)
                        if not r:
                                print(string)
                        current_dep = r.group(0)
                        num_candidates += 1

                    if "DependencyValidator" in string and "finished" in string:
                        res = 0
                        for regex, div in zip(regexes, divs):
                            r = re.search(regex, string)
                            if not r:
                                continue
                            t = int(r.group(0))
                            res += t * div
                        summary = MiningResult(num_candidates, num_valid, res)
                        summary.generation_time = generation_time
                        mining_summaries[impl][b] = summary



    for benchmark in benchmarks:
        print(" " * 3, benchmark)
        for impl in dirs:
            summary = mining_summaries[impl][benchmark]
            print(" " * 7, impl, summary, summary.generation_time / 10**6)



    for dep in ["UCC", "IND", "OD"]:
        print("\n", dep, "\n", sep="")
        for benchmark in benchmarks:
            print(" " * 3, benchmark)
            for impl in dirs:
                alias = deps_alias[dep]

                summary = mining_summaries[impl][benchmark]
                #print(run_times_all[alias][benchmark]["valid"][impl])

                print(" " * 7, impl, len(run_times_all[alias][benchmark]["valid"][impl]), "/", len(run_times_all[alias][benchmark]["invalid"][impl]))
                if impl == "wide":
                    known_valid = {x[0] for x in deps_all[alias][benchmark]["valid"]["narrow"]}
                    known_invalid = {x[0] for x in deps_all[alias][benchmark]["invalid"]["narrow"]}
                    print("\t\tValid:")
                    for c_d, t in sorted(deps_all[alias][benchmark]["valid"][impl], key=lambda x: x[1], reverse=True):
                        if c_d not in known_valid:
                            print("\t\t\t", c_d, "\t", t)
                    print("\t\tInvalid:")
                    for c_d, t in sorted(deps_all[alias][benchmark]["invalid"][impl], key=lambda x: x[1], reverse=True):
                        if c_d not in known_invalid:
                            print("\t\t\t", c_d, "\t", t)

    # index --> benchmark
    # df --> valid/invalid
    # columns --> impl

    sns.set()
    sns.set_theme(style="whitegrid")
    plt.style.use('seaborn-colorblind')
    plt.rcParams["figure.figsize"] = [x * 1 for x  in plt.rcParams["figure.figsize"]]
    plt.rcParams["font.family"] = "serif"

    to_legend_entry = lambda kw: f"{kw[0].upper()}{kw[1:]}"
    bar_width = 0.15
    epsilon = 0.015
    margin = 0.01



    bens = [b for b in list(benchmarks.keys())]

    group_centers = np.arange(len(bens))
    offsets = [-0.5, 0.5]
    ax = plt.gca()
    colors_invalid = {b: c for b, c in zip(dirs.keys(), Safe_6.hex_colors[:len(dirs)])}
    colors_valid = {b: c for b, c in zip(dirs.keys(), Safe_6.hex_colors[len(dirs):len(dirs) * 2])}


    for impl, color, offset in zip(dirs, Safe_6.hex_colors[:len(dirs)], offsets):
        bar_positions = [p + offset * (bar_width + margin) for p in group_centers]
        #vals = [max(sum(data[b]["invalid"][impl]) / 10**6, min_val) for b in bens]
        vals = [(mining_summaries[impl][b].time + mining_summaries[impl][b].generation_time) / 10**9 for b in bens]
        vals = [(mining_summaries[impl][b].time) / 10**9 for b in bens]
        ax.bar(bar_positions, vals, bar_width, color=color, label=to_legend_entry(impl))




    plt.xticks(group_centers, bens, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.ylabel('Run-time [s]', fontsize=16)
    plt.xlabel('Benchmark', fontsize=16)
    legend_loc = "best"
    ax.legend(loc=legend_loc, title="Generation Rule")
    plt.tight_layout(pad=0)
    plt.savefig(f"compare_sharpness_runtime.eps", dpi=300, bbox_inches="tight")
    plt.close()

    ax = plt.gca()
    for impl, offset in zip(dirs, offsets):
        bar_positions = [p + offset * (bar_width + margin) for p in group_centers]
        #vals = [max(sum(data[b]["invalid"][impl]) / 10**6, min_val) for b in bens]
        num_valids = [mining_summaries[impl][b].num_valid for b in bens]
        num_invalids = [mining_summaries[impl][b].num_candidates for b in bens]
        ax.bar(bar_positions, num_invalids, bar_width, color=colors_invalid[impl], label=f"Invalid {to_legend_entry(impl)}")
        ax.bar(bar_positions, num_valids, bar_width, color=colors_valid[impl], label=f"Valid {to_legend_entry(impl)}")

    plt.xticks(group_centers, bens, rotation=0)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.ylabel('Number of Candidates', fontsize=16)
    plt.xlabel('Benchmark', fontsize=16)
    legend_loc = "best"
    ax.legend(loc=legend_loc, title="Candidates")
    plt.tight_layout(pad=0)
    plt.savefig(f"compare_sharpness_candidates.eps", dpi=300, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()
