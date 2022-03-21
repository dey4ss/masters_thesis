#!/usr/bin/env python3

import argparse
import os
import re
from collections import defaultdict
from subprocess import Popen, PIPE

import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import rc
from palettable.cartocolors.qualitative import Antique_6, Bold_6, Pastel_6, Prism_6, Safe_6, Vivid_6

class MiningResult():
    def __init__(self, num_candidates, num_valid, time):
        self.num_candidates = num_candidates
        self.num_valid = num_valid
        self.time = time
        self.generation_time = 0

    @staticmethod
    def candidates_indicator():
        return "Check candidate: "

    @staticmethod
    def valid_indicator():
        return "    Valid"

    @staticmethod
    def invalid_indicator():
        return "    Invalid"

    @staticmethod
    def mining_time_indicator():
        return "DependencyMiningPlugin finished in "

    def __str__(self):
        return(f"{self.num_candidates} candidates, {self.num_valid} valid, {self.time / 10**6} ms")

    def num_invalid(self):
        return self.num_candidates - self.num_valid


class BenchmarkResult():

    def __init__(self, abs_change, rel_change, num_losses, num_gains, max_loss, max_gain):
        self.abs_change = abs_change
        self.rel_change = rel_change
        self.num_losses = num_losses
        self.num_gains = num_gains
        self.max_loss = max_loss
        self.max_gain = max_gain

    def __str__(self):
        return f"{self.abs_change}s, {self.rel_change}%, {self.num_losses} loss (max {self.max_loss}%), {self.num_gains} gain (max {self.max_gain}%)"


def parse_args():
    ap = argparse.ArgumentParser(description="Greps validation and benchmark results for all benchmarks, configs, and scale factors, and generates plots")
    ap.add_argument("dir", type=str, help="Path to directory containing benchmark result files")
    ap.add_argument("--output", "-o", type=str, help="Output directory", default="mining_comparison_plots")
    ap.add_argument("--extension", "-e", type=str, help="Plot file format", choices=["png", "pdf", "eps", "svg"], default="png")
    ap.add_argument("--compare_benchmark_path", "-c", type=str, help="Path to compare_benchmarks.py", default="./compare_benchmarks.py")
    return ap.parse_args()

def indent(level, msg):
    return f"{' ' * 4 * level}{msg}"


def main(directory, output, extension, compare_benchmark_path):
    benchmarks = list(reversed(["TPCH", "TPCDS", "JoinOrder"]))
    benchmark_short = {"TPCH": "TPC-H", "TPCDS": "TPC-DS", "JoinOrder": "JOB"}
    mining_time_indicator = "DependencyMiningPlugin finished in "
    # order: Config -> benchmark -> scale -> num_candidates
    mining_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
    benchmark_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
    regexes = [r'\d+(?=\ss)', r'\d+(?=\sms)', r'\d+(?=\sµs)', r'\d+(?=\sns)']

    configs = {"Join2Semi": "only_jts", "DGR": "only_dgr", "Join2Predicate": "only_join2pred", "JoinElimination": "only_join_elim", "All w/o JoinElimination": "dgr_jts_join2pred", "Combined": "all_on"}
    config_names = {v: k for k, v in configs.items()}

    #print(sorted(style for style in plt.style.available if style != 'classic'))

    base_palette = Safe_6.hex_colors
    print(plt.rcParams["figure.figsize"], [x * 1 for x  in plt.rcParams["figure.figsize"]])

    colors = sns.color_palette("colorblind")
    colors = sns.color_palette(cc.glasbey, n_colors=len(configs))
    colors = base_palette[:len(configs)]
    colors_per_config = {k: colors[i] for k, i in zip(list(configs.values()), range(len(configs)))}
    markers = ["^", "X", "s", "D", ".", "o"]
    markers_per_config = {k: markers[i] for k, i in zip(list(configs.values()), range(len(configs)))}

    baselines = dict()
    divs = list(reversed([1, 10**3, 10**6, 10**9]))
    scales = [1, 10, 20, 50, 100]
    scales_per_benchmark = defaultdict(lambda: set())
    configs_per_benchmark = defaultdict(lambda: set())
    candidates_per_benchmark = defaultdict(lambda: set())
    all_files = os.listdir(directory)
    json_files = [f for f in os.listdir(directory) if f.endswith(".json") and not "all_off" in f]
    log_files = {f for f in os.listdir(directory) if f.endswith(".log")}
    sf_regex = re.compile(r'(?<=_s-)\d+')
    candidates_regex = re.compile(r'(?<=_max-cand-)\d+')
    for benchmark in benchmarks:
        config_regex = re.compile(f'(?<=hyriseBenchmark{benchmark}_)\\w+(?=(_s-|.json|_max-cand))')
        my_files = [f for f in json_files if benchmark in f]
        for f in my_files:
            if f.startswith("__"):
                continue
            log_file_name = f[:-len("json")] + "log"
            assert log_file_name in log_files

            configs_per_benchmark[benchmark].add(re.search(config_regex, f).group(0))
            if "_s" in f:
                scales_per_benchmark[benchmark].add(re.search(sf_regex, f).group(0))
            if "_max-cand" in f:
                candidates_per_benchmark[benchmark].add(re.search(candidates_regex, f).group(0))
            else:
                candidates_per_benchmark[benchmark].add("all")
    scales_per_benchmark["JoinOrder"].add(10)

    print("GATHER DATA")
    benchmark_stats = defaultdict(lambda: dict())
    # candidate_regex = re.compile(r'(?<=PQPAnalyzer generated )\d+(?= candidates)')
    query_num_regex = re.compile(r'(?<=>= \d%:)\s+\d+')
    for benchmark, configurations in configs_per_benchmark.items():
        print(benchmark)
        for config in configurations:
            print(indent(1, f"{config} // {config_names[config]}"))
            for sf in scales_per_benchmark[benchmark]:
                if sf != 10 and benchmark == "JoinOrder": continue
                sf_key = int(sf) if sf != "001" else 0.01
                #print(indent(2, f"SF {sf_key}"))
                sf_indicator = f"_s-{sf}" if benchmark != "JoinOrder" else ""
                compare_file_name = os.path.join(directory, f"hyriseBenchmark{benchmark}_all_off{sf_indicator}.json")
                for num_cand in candidates_per_benchmark[benchmark]:
                    #print(indent(3, f"{num_cand} candidates"))
                    num_cand_indicator = "" if num_cand == "all" else f"_max-cand-{num_cand}"
                    base_file_name = os.path.join(directory, f"hyriseBenchmark{benchmark}_{config}{sf_indicator}{num_cand_indicator}")

                    #print(indent(4, "- Mining"))
                    with open(f"{base_file_name}.log") as f:
                        num_candidates = 0
                        num_valid = 0
                        for line in f:
                            if MiningResult.mining_time_indicator() in line:
                                string = line.strip()[len(MiningResult.mining_time_indicator()):]
                                res = 0
                                for regex, div in zip(regexes, divs):
                                    r = re.search(regex, string)
                                    if not r:
                                        continue
                                    t = int(r.group(0))
                                    res += t * div

                                # result is ns, should be ms
                                #mining_results[config_name][benchmark] = res / 10**6

                            elif MiningResult.candidates_indicator() in line:
                                # print(f"'{MiningResult.candidates_indicator()}'", line)
                                num_candidates += 1

                            elif MiningResult.valid_indicator() in line:
                                num_valid += 1
                        mining_results[config][benchmark][sf_key][num_candidates] = MiningResult(num_candidates, num_valid, res)

                    #print(indent(4, "- Benchmark"))
                    with Popen(f"{compare_benchmark_path} {compare_file_name} {base_file_name}.json", shell=True, stdout=PIPE) as p:
                        p.wait()
                        for l in p.stdout:
                            line = l.decode("utf-8").strip()
                            if "abs. change" in line:
                                abs_change = float(line[len("abs. change: "):-1])
                            elif "rel. change" in line:
                                rel_change = int(line[len("rel. change: "):-1])
                            elif "# losses" in line:
                                #print(line, re.search(query_num_regex, line))
                                num_losses = int(re.search(query_num_regex, line).group(0))
                            elif "# gains" in line:
                                #print(line, re.search(query_num_regex, line))
                                num_gains = int(re.search(query_num_regex, line).group(0))
                            elif "baseline:" in line:
                                baseline = float(line[len("baseline: "):-1])
                            elif "# overall: " in line:
                                # print(line)
                                num_overall = int(line[len("# overall: "):])
                            elif "max loss:" in line:
                                max_loss = int(line[len("max loss:"):].strip().replace("%", ""))
                            elif "max gain:" in line:
                                max_gain = int(line[len("max gain:"):].strip().replace("%", ""))
                        benchmark_stats[benchmark][sf_key] = (num_overall, baseline)
                        benchmark_results[config][benchmark][sf_key][num_candidates] = BenchmarkResult(abs_change, rel_change, num_losses, num_gains, max_loss, max_gain)

                    if sf_key == 10:
                        print(indent(2, benchmark_results[config][benchmark][sf_key][num_candidates]))
                        print(indent(2, mining_results[config][benchmark][sf_key][num_candidates]))

    print("")
    for benchmark in benchmarks:
        print(benchmark, benchmark_stats[benchmark][10])


    print("")
    # print(benchmark_stats)
    print("GENERATE PLOTS")
    # plots for mining zime vs win per benchmark + config, scale/#candidates
    if not os.path.isdir(output):
        os.makedirs(output)
    sns.set()
    sns.set_theme(style="whitegrid")
    #sns.color_palette("viridis")
    #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    #rc('font',**{'family':'serif','serif':['Times']})
    #rc('text', usetex=True)
    plt.rcParams["font.family"] = "serif"
    #axes.titlesize : 24
    # plt.rcParams["axes.labelsize"] = 16
    #axes.labelsize : 20
    #lines.linewidth : 3
    #lines.markersize : 10
    #xtick.labelsize : 16
    #ytick.labelsize : 16
    plt.style.use('seaborn-colorblind')
    plt.rcParams["figure.figsize"] = [x * 1 for x  in plt.rcParams["figure.figsize"]]

    def check_single(keys):
        assert len(keys) == 1
        return list(keys)[0]

    all_x = list()
    all_y = list()
    all_conf = list()
    all_ind = list()
    all_bench = list()
    all_metric = list()

    dashes = {"Discovery Time": (3, 3), "Latency Improvement": ""}

    do_scales = False
    do_candidates = False
    got_scales = False
    for benchmark in configs_per_benchmark:
        print(benchmark)

        do_scales = len(scales_per_benchmark[benchmark]) > 1
        do_candidates = len(candidates_per_benchmark[benchmark]) > 1

        if do_scales:
            print(indent(1, "Discovery Time vs. Latency Change: Scale factors"))
        if do_candidates:
            print(indent(1, "Discovery Time vs. Latency Change: # Candidates"))

        if not any([do_scales, do_candidates]):
            continue

        got_scales = got_scales or do_scales

        for config in configs_per_benchmark[benchmark]:
            config_name = config_names[config]
            print(indent(2, config_name))
            if do_scales or (not do_candidates and benchmark == "JoinOrder"):
                scales = list(sorted(benchmark_results[config][benchmark].keys()))
                candidate_keys = [check_single(benchmark_results[config][benchmark][s].keys()) for s in scales]
                x_values = scales
            elif do_candidates:
                scale = list(sorted(benchmark_results[config][benchmark].keys()))[0]
                candidate_keys = list(sorted(benchmark_results[config][benchmark][scale].keys()))
                scales = [scale for _ in range(len(candidate_keys))]
                x_values = candidate_keys
            else:
                assert False
            #print(config_name, scales)
            # time is in ns, needs to be transformed to s
            benchmark_base_times = [benchmark_stats[benchmark][s][1] for s in scales]
            #print(benchmark_base_times)

            for metric in ["absolute", "relative"]:
                if metric == "relative":
                    mining_times = [(mining_results[config][benchmark][s][k].time / divs[0] / t) * 100 for s, k, t in zip(scales, candidate_keys, benchmark_base_times)]
                    benchmark_gains = [benchmark_results[config][benchmark][s][k].rel_change * -1 for s, k in zip(scales, candidate_keys)]
                elif metric == "absolute":
                    mining_times = [(mining_results[config][benchmark][s][k].time / divs[0]) for s, k in zip(scales, candidate_keys)]
                    benchmark_gains = [benchmark_results[config][benchmark][s][k].abs_change * -1 for s, k in zip(scales, candidate_keys)]
                else:
                    raise ValueError(f"unknown metric: {metric}")

                #print(indent(1, mining_times))
                #print(indent(1, benchmark_gains))
                x = x_values + x_values
                ind =  ["Latency Improvement"] * len(x_values) + ["Discovery Time"] * len(x_values)
                vals =  benchmark_gains + mining_times
                # print(len(x), len(vals))
                all_x += x
                all_y += vals
                all_ind += ind
                all_conf += [config_name for _ in range(len(x))]
                all_bench += [benchmark_short[benchmark] for _ in range(len(x))]
                all_metric += [metric for _ in range(len(x))]
                #print(len(x), x)
                #print(len(ind), ind)
                #print(len(vals), vals)
                if not any([do_scales, do_candidates]):
                    continue
                my_data = pd.DataFrame(data={"x": x, "Measurement": ind, "val": vals})
                #sns.lineplot(x=x_values, y=mining_times, label="Discovery Time", marker=markers_per_config[config], color=colors_per_config[config])
                p = sns.lineplot(data=my_data, x="x", y="val", hue="Measurement", style="Measurement", dashes=dashes, markers=[markers_per_config[config] for _ in range(2)], palette=[colors_per_config[config], sns.desaturate(colors_per_config[config], 0.5)], linewidth=2, markersize=9)
                #sns.lineplot(x=x_values, y=benchmark_gains, label="Latency Improvement", marker=markers_per_config[config], color=sns.desaturate(colors_per_config[config], 0.7), dashes=[(2,2)])
                plt.xlabel("Scale Factor" if do_scales else "Candidates", fontsize=16)
                y_label = "Share of Benchmark Run-time [%]" if metric == "relative" else "Run-time [s]"
                plt.ylabel(y_label, fontsize=16)
                title = "Discovery Time vs. Latency Improvement"
                # plt.title(f"{title} {benchmark} {config_name}")
                min_y = min(min(benchmark_gains), min(mining_times))
                if min_y >= 0:
                    plt.axis(ymin=0)
                plt.axis(xmin=0)
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.tick_params(axis='both', which='minor', labelsize=14)
                # plt.setp(p.get_legend().get_title(), fontsize='0.5')
                plt.tight_layout(pad=0)
                file_indicator = "scales" if do_scales else "candidates"
                # print(os.path.join(output, f"{benchmark}_{file_indicator}_{config}_{metric}.{extension}"))
                plt.savefig(os.path.join(output, f"{benchmark}_{file_indicator}_{config}_{metric}.{extension}"), dpi=300, bbox_inches="tight")
                plt.close()

        '''
        for a, i in zip([all_x, all_y, all_ind, all_bench, all_conf, all_metric], ["x", "y", "ind", "bench", "conf", "metric"]):
            print(f"\n{i}")
            if type(a) in [defaultdict, dict]:
                for x, y in a.items():
                    print(indent(1, f"{x} {len(y)}"))
                    print(indent(2, y))
            else:
                print(indent(1, len(a)))
                #print(indent(2, a))
        '''

        if not do_scales: continue
        # all together
        print(indent(2, "Configs Merged"))
        for metric in ["absolute", "relative"]:
            my_data = pd.DataFrame(data={"x": all_x, "Configuration": all_conf, "Measurement": all_ind, "Benchmark": all_bench, "val": all_y, "metric": all_metric})
            my_data = my_data[my_data.Benchmark.eq(benchmark)]
            my_data = my_data[my_data.metric.eq(metric)]
            #markers = list()
            #for config in configs_per_benchmark[benchmark]:
            #    markers += [markers_per_config[config] for _ in range(2)]
            my_dashes = ["", (2, 2)] * len(configs_per_benchmark[benchmark])
            #sns.lineplot(x=x_values, y=mining_times, label="Discovery Time", marker=markers_per_config[config], color=colors_per_config[config])
            my_markers = list()
            for config in configs_per_benchmark[benchmark]:
                my_markers += [markers_per_config[config] for _ in range(2)]
            palette = {config_names[k]: v for k, v in colors_per_config.items()}

            sns.lineplot(data=my_data, x="x", y="val", hue="Configuration", style="Measurement", dashes=dashes, markers=markers[:2], palette=palette, linewidth=2, markersize=9)
            #sns.lineplot(x=x_values, y=benchmark_gains, label="Latency Improvement", marker=markers_per_config[config], color=sns.desaturate(colors_per_config[config], 0.7), dashes=[(2,2)])
            plt.xlabel("Scale Factor" if do_scales else "Candidates", fontsize=16)
            y_label = "Share of Benchmark Run-time [%]" if metric == "relative" else "Run-time [s]"
            plt.ylabel(y_label, fontsize=16)
            title = "Discovery Time vs. Latency Improvement"
            # plt.title(f"{title} {benchmark}")
            min_y = min(all_y)
            if min_y >= 0:
                plt.axis(ymin=0)
            plt.axis(xmin=0)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            plt.tight_layout(pad=0)
            file_indicator = "scales" if do_scales else "candidates"
            # print(os.path.join(output, f"{benchmark}_{file_indicator}_merged_{metric}.{extension}"))
            plt.savefig(os.path.join(output, f"{benchmark}_{file_indicator}_merged_{metric}.{extension}"), dpi=300, bbox_inches="tight")
            plt.close()


    for metric in ["absolute", "relative"]:
        combined_data = pd.DataFrame(data={"x": all_x, "Configuration": all_conf, "Measurement": all_ind, "Benchmark": all_bench, "val": all_y, "metric": all_metric})
        combined_data = combined_data[combined_data.metric.eq(metric)]
        palette = sns.color_palette(cc.glasbey, n_colors=len(benchmarks))
        palette = base_palette[:len(benchmarks)]

        palette = {b: c for b, c in zip([benchmark_short[b] for b in benchmarks], palette)}
        if do_candidates:
            # all together

            #markers = list()
            #for config in configs_per_benchmark[benchmark]:
            #    markers += [markers_per_config[config] for _ in range(2)]
            # dashes = ["", (2, 2)] * len(configs_per_benchmark[benchmark])
            #sns.lineplot(x=x_values, y=mining_times, label="Discovery Time", marker=markers_per_config[config], color=colors_per_config[config])


            # palette = {config_names[k]: v for k, v in colors_per_config.items()}
            #dashes = ["", (2, 2)] * 3

            #my_colors = list()
            #from matplotlib import colors as cs
            #for color in palette:
            #    my_colors.append(color)
            #    color_hsl = cs.rgb_to_hsv(color)
            #    saturation = color_hsl[1]
            #    target_color = cs.hsv_to_rgb((color_hsl[0], saturation * 0.5, color_hsl[2]))
            #    #my_colors.append(target_color)
            #    my_colors.append(sns.desaturate(color, 0.5))
            #    print(color, target_color)
            #my_markers = list()
            #for i in range(len(benchmarks)):
            #    my_markers += [markers[i] for _ in range(2)]

            #hues = list()
            #i = 0
            #for benchmark in benchmarks:
            #    num_val = len(my_data[my_data.Benchmark.eq(benchmark)]) / 2
            #    print(num_val)
            #    hues += [i for _ in range(int(num_val))]
            #    i += 1
            #    hues += [i for _ in range(int(num_val))]
            #    i += 1

            print("Benchmark Candidates Merged", metric)
            sns.lineplot(data=combined_data, x="x", y="val", style="Measurement", markers=markers[:2], hue="Benchmark", dashes=dashes, palette=palette, linewidth=2, markersize=9)
            #sns.lineplot(x=x_values, y=benchmark_gains, label="Latency Improvement", marker=markers_per_config[config], color=sns.desaturate(colors_per_config[config], 0.7), dashes=[(2,2)])
            plt.xlabel("Scale Factor" if do_scales else "Candidates", fontsize=16)
            y_label = "Share of Benchmark Run-time [%]" if metric == "relative" else "Run-time [s]"
            plt.ylabel(y_label, fontsize=16)
            title = "Discovery Time vs. Latency Improvement"
            # plt.title(f"{title}")
            min_y = min(all_y)
            if min_y >= 0:
                plt.axis(ymin=0)
            plt.axis(xmin=0)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            plt.tight_layout(pad=0)
            file_indicator = "scales" if do_scales else "candidates"
            # print(os.path.join(output, f"{benchmark}_{file_indicator}_merged_{metric}.{extension}"))
            plt.savefig(os.path.join(output, f"all_{file_indicator}_merged_{metric}.{extension}"), dpi=300, bbox_inches="tight")
            plt.close()

        if got_scales:
            print("Benchmark Scales merged per Config", metric)
            my_configs = combined_data.Configuration.unique()
            # print(configs)

            for config in my_configs:
                print(indent(1, config))
                my_data = combined_data[combined_data.Configuration.eq(config)]
                sns.lineplot(data=my_data, x="x", y="val", style="Measurement", markers=markers[:2], hue="Benchmark", dashes=dashes, palette=palette, linewidth=2, markersize=9)
                plt.xlabel("Scale Factor" if do_scales else "Candidates", fontsize=16)
                y_label = "Share of Benchmark Run-time [%]" if metric == "relative" else "Run-time [s]"
                plt.ylabel(y_label, fontsize=16)
                title = "Discovery Time vs. Latency Improvement"
                # plt.title(f"{title} {config}")
                min_y = my_data.min().val
                if min_y >= 0:
                    plt.axis(ymin=0)
                plt.axis(xmin=0)
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=14)
                ax.tick_params(axis='both', which='minor', labelsize=14)
                if metric == "relative":
                    ax.get_legend().remove()
                plt.tight_layout(pad=0)
                # print(os.path.join(output, f"{configs[config]}_benchmarks_merged_{metric}.{extension}"))
                plt.savefig(os.path.join(output, f"{configs[config]}_benchmarks_merged_{metric}.{extension}"), dpi=300, bbox_inches="tight")
                plt.close()




    '''
    print("Baseline", end=" ")
    for benchmark in benchmarks:
        print(f"& \\numprint[s]{{{benchmark_stats[benchmark][1]}}}", end=" ")
        print("& -- " * 6, end=" ")
    print(" \\\\")

    for c in configs.keys():
        print(f"{c}", end=" ")
        for benchmark, benchmark_res in benchmark_results[c].items():
            # print("-----", benchmark, benchmark_res)
            print(f"& {'-' if benchmark_res.abs_change < 0 else ''}\\numprint[s]{{{abs(benchmark_res.abs_change)}}}", end=" ")
            print(f"& {'-' if benchmark_res.abs_change < 0 else ''}\\numprint[\\%]{{{abs(benchmark_res.rel_change)}}}", end=" ")
            print(f"& {benchmark_res.num_losses} & {benchmark_res.num_gains}", end=" ")
            mining_res = mining_results[c][benchmark]
            print(f"& {mining_res.num_candidates} & {mining_res.num_valid} & \\numprint[ms]{{{mining_res.time}}}", end=" ")
        print(" \\\\")
    '''


if __name__ == "__main__":
    args = parse_args()
    main(args.dir, args.output, args.extension, args.compare_benchmark_path)
