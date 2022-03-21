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
    return b != "TPC-H" or dep != "ind"


def main():
    dirs = {"operator": "benchmark_impl_hash","dictionary": "benchmark_impl_spider", "set": "benchmark_impl_set"}
    benchmarks = {"TPC-H": "TPCH", "TPC-DS": "TPCDS", "JOB": "JoinOrder"}
    run_times_ind = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    run_times_ucc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    deps_ind = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    deps_ucc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


    regexes = [r'\d+(?=\ss)', r'\d+(?=\sms)', r'\d+(?=\sµs)', r'\d+(?=\sns)']
    divs = list(reversed([1, 10**3, 10**6, 10**9]))
    type_regex = r'(?<=Type\s)\w+'
    name_regex = r'(?<=Columns\s).+'

    # benchmark -> validity -> impl -> run times
    for impl, directory in dirs.items():
        for b, benchmark in benchmarks.items():
            print(directory, benchmark)
            sf = "_s-10" if b != "JOB" else ""
            with open(os.path.join(directory, f"hyriseBenchmark{benchmark}_all_on{sf}.log")) as f:
                current_type = ""
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

                        # result is ns, should be ms
                        #mining_results[config_name][benchmark] = res / 10**6
                        #if current_type ==
                        if current_type not in ["Inclusion", "Unique"]:
                            continue

                        result_dict = run_times_ind if current_type == "Inclusion" else run_times_ucc
                        k = "valid" if is_valid else "invalid"

                        result_dict[b][k][impl].append(res)

                        result_dict = deps_ind if current_type == "Inclusion" else deps_ucc
                        result_dict[b][k][impl].append(current_dep)

                    elif MiningResult.candidates_indicator() in line:
                        # print(f"'{MiningResult.candidates_indicator()}'", line)
                        r = re.search(type_regex, string)
                        current_type = r.group(0)
                        r = re.search(name_regex, string)
                        current_dep = r.group(0)

    for dep, d, ds in zip(["UCC", "IND"], [run_times_ucc, run_times_ind], [deps_ucc, deps_ind]):
        print("\n", dep, "\n", sep="")
        for benchmark in benchmarks:
            print(benchmark)
            for impl in dirs:
                print("\t", impl, len(d[benchmark]["valid"][impl]), len(d[benchmark]["invalid"][impl]))
                #print("\t\tValid:")
                #for c_d in ds[benchmark]["valid"][impl]:
                #    print("\t\t\t", c_d)
                #print("\t\tInvalid:")
                #for c_d in ds[benchmark]["invalid"][impl]:
                #    print("\t\t\t", c_d)

    # index --> benchmark
    # df --> valid/invalid
    # columns --> impl

    sns.set()
    sns.set_theme(style="whitegrid")
    plt.style.use('seaborn-colorblind')
    plt.rcParams["figure.figsize"] = [x * 1 for x  in plt.rcParams["figure.figsize"]]
    plt.rcParams["font.family"] = "serif"

    to_legend_entry = lambda kw: f"{kw[0].upper()}{kw[1:]}-based"
    bar_width = 0.15
    epsilon = 0.015
    margin = 0.01


    for d_type, data in zip(["ucc"], [run_times_ucc]):
        print(f"\n{d_type.upper()}")
        bens = [b for b in list(benchmarks.keys()) if use_b(b, d_type)]

        group_centers = np.arange(len(bens))
        offsets = [-1, 0, 1]

        f, (ax, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 6]})

        df1 = pd.DataFrame(collect_values(data, "valid", bens, dirs), index=bens, columns=short(list(dirs.keys())))
        df2 = pd.DataFrame(collect_values(data, "invalid", bens, dirs), index=bens, columns=short(list(dirs.keys())))

        print(df1.head())
        print("\t".join([f"{c}: {round(sum(df1[c]))}" for c in df1.columns]))
        print("")
        print(df2.head())
        print("\t".join([f"{c}: {round(sum(df2[c]))}" for c in df2.columns]))
        df1 = prep_df(df1, 'Valid')
        df2 = prep_df(df2, 'Invalid')
        # print(df2.head())

        df = pd.concat([df1, df2])

        all_durations = list()

        for impl, color, offset in zip(dirs, Safe_6.hex_colors[:len(dirs)], offsets):
            bar_positions = [p + offset * (bar_width + margin) for p in group_centers]
            vals = [sum(data[b]["valid"][impl]) / 10**9 for b in bens]
            all_durations += vals

            ax.bar(bar_positions, vals, bar_width, color=color, label=to_legend_entry(impl))
            ax2.bar(bar_positions, vals, bar_width, color=color)

            validation_times = list()
            for b in bens:
                validation_times += data[b]["valid"][impl]
            print(f"    {to_legend_entry(impl)}: avg {(sum(validation_times) / len(validation_times) / divs[1])} ms")

        sorted_durations = sorted(all_durations, reverse=True)
        print(sorted_durations)
        max_dur = sorted_durations[0]
        remaining_limit = sorted_durations[1]
        ax.set_ylim(math.floor(max_dur), math.ceil(max_dur))  # outliers only
        ax2.set_ylim(0, math.ceil(remaining_limit))  # most of the data

        # hide the spines between ax and ax2
        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        #ax.xaxis.tick_top()
        ax.tick_params(labeltop=False)  # don't put tick labels at the top
        #ax2.xaxis.tick_bottom()

        # This looks pretty good, and was fairly painless, but you can get that
        # cut-out diagonal lines look with just a bit more work. The important
        # thing to know here is that in axes coordinates, which are always
        # between 0-1, spine endpoints are at these locations (0,0), (0,1),
        # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
        # appropriate corners of each of our axes, and so long as we use the
        # right transform and disable clipping.

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='darkgrey', clip_on=False)
        ax.plot((-d, +d), (-6*d, +6*d), **kwargs)        # top-left diagonal
        ax.plot((1 - d, 1 + d), (-6*d, +6*d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        plt.xticks(group_centers, bens, rotation=0)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='minor', labelsize=14)
        plt.ylabel('Run-time [s]', fontsize=16)
        plt.xlabel('Benchmark', fontsize=16)
        ax.legend(loc="upper right", title='Validation Strategy')
        # sns.despine()
        plt.tight_layout(pad=0)
        plt.savefig(f"compare_validation_{d_type}.eps", dpi=300, bbox_inches="tight")
        plt.close()

    for d_type, data in zip(["ind"], [run_times_ind]):
        print(f"\n{d_type.upper()}")
        bens = [b for b in list(benchmarks.keys()) if use_b(b, d_type)]

        bar_width = 0.15
        epsilon = 0.015
        margin = 0.01

        group_centers = np.arange(len(bens))
        offsets = [-1, 0, 1]

        df1 = pd.DataFrame(collect_values(data, "valid", bens, dirs), index=bens, columns=short(list(dirs.keys())))
        df2 = pd.DataFrame(collect_values(data, "invalid", bens, dirs), index=bens, columns=short(list(dirs.keys())))

        print(df1.head())
        print("\t".join([f"{c}: {round(sum(df1[c]))}" for c in df1.columns]))
        print("")
        print(df2.head())
        print("\t".join([f"{c}: {round(sum(df2[c]))}" for c in df2.columns]))
        df1 = prep_df(df1, 'Valid')
        df2 = prep_df(df2, 'Invalid')
        # print(df2.head())

        df = pd.concat([df1, df2])

        for impl, color, offset in zip(dirs, Safe_6.hex_colors[:len(dirs)], offsets):
            bar_positions = [p + offset * (bar_width + margin) for p in group_centers]
            vals = [sum(data[b]["valid"][impl]) / 10**9 for b in bens]

            plt.bar(bar_positions, vals, bar_width, color=color, label=to_legend_entry(impl))

            validation_times = list()
            for b in bens:
                validation_times += data[b]["valid"][impl]
            print(f"    {to_legend_entry(impl)}: avg {(sum(validation_times) / len(validation_times) / divs[1])} ms")

        plt.xticks(group_centers, bens, rotation=0)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.xlabel('Benchmark', fontsize=16)
        plt.ylabel('Run-time [s]', fontsize=16)
        plt.legend(loc='best')
        plt.legend(title='Validation Strategy')
        plt.ylim([0, 1.6])
        # sns.despine()
        plt.tight_layout(pad=0)
        plt.savefig(f"compare_validation_{d_type}.eps", dpi=300, bbox_inches="tight")
        plt.close()



if __name__ == "__main__":
    main()
