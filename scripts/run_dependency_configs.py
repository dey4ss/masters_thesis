#!/usr/bin/python3

import argparse
import os
import shutil
import json
from subprocess import Popen, PIPE

# Just holds the global state (**yeah**) of the benchmarks to run
class ExperimentSetup:
    def __init__(self):
        self.benchmarks = ["hyriseBenchmarkTPCDS", "hyriseBenchmarkJoinOrder"]
        self.benchmarks = ["hyriseBenchmarkTPCH", "hyriseBenchmarkTPCDS", "hyriseBenchmarkJoinOrder"]
        self.benchmarks = ["hyriseBenchmarkTPCDS"]

        self.configs = {
            "all_off": DependencyUsageConfig(False, False, False, False),
            # "only_dgr": DependencyUsageConfig(True, False, False, False),
            # "only_jts": DependencyUsageConfig(False, True, False, False),
            "only_join2pred": DependencyUsageConfig(False, False, True, False),
            # "only_join_elim": DependencyUsageConfig(False, False, False, True),
            #"jts_join2pred_join_elim": DependencyUsageConfig(False, True, True, True),
            #"all_on": DependencyUsageConfig(True, True, True, True),
            # "dgr_jts_join2pred": DependencyUsageConfig(True, True, True, False),
        }
        # self.scale_factors = [0.01, 1, 10, 20, 50, 100]
        self.scale_factors = [10]
        num_candidates = [1, 2, 3, 4, 5, 7, 10, 15, 20, 40, 60, 80, MiningConfig.default_config().max_validation_candidates]
        num_threads = list(range(2, 11))
        # self.mining_configs = [MiningConfig(max_validation_candidates=n) for n in num_candidates]
        # self.mining_configs = [MiningConfig(num_validators=n) for n in num_threads]
        self.mining_configs = [MiningConfig.default_config()]


class DependencyUsageConfig:
    def __init__(self, groupby_reduction, join_to_semi, join_to_predicate, join_elimination, preset_constraints=False):
        self.groupby_reduction = groupby_reduction
        self.join_to_semi = join_to_semi
        self.join_to_predicate = join_to_predicate
        self.join_elimination = join_elimination
        self.preset_constraints = preset_constraints

    def __eq__(self, other):
        return vars(self) == vars(other)

    def allows_nothing(self):
        return self == DependencyUsageConfig(False, False, False, False)

    def to_json(self, file_path):
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)


class MiningConfig:
    def __init__(self, num_validators=1, max_validation_candidates=-1, max_validation_time=-1):
        self.num_validators = num_validators
        self.max_validation_candidates = max_validation_candidates
        self.max_validation_time = max_validation_time

    def __eq__(self, other):
        return vars(self) == vars(other)

    @staticmethod
    def default_config():
        return MiningConfig()

    def is_default_config(self):
        return self == MiningConfig.default_config()

    def file_extension(self):
        default_config = MiningConfig.default_config()
        if self == default_config:
            return ""
        extension = ""
        if self.num_validators != default_config.num_validators:
            extension += f"_{self.num_validators}-validators"
        if self.max_validation_candidates != default_config.max_validation_candidates:
            extension += f"_max-cand-{self.max_validation_candidates}"
        if self.max_validation_time != default_config.max_validation_time:
            extension += f"_max-time-{self.max_validation_time}"
        return extension

    def to_json(self, file_path):
        with open(file_path, "w") as f:
            json.dump(vars(self), f, indent=4)


def parse_args():
    ap = argparse.ArgumentParser(description="Runs benchmarks for pre-defined dependency usage configurations")
    ap.add_argument("output_path", type=str, help="Path to output directory")
    ap.add_argument(
        "--build_dir",
        type=str,
        default="cmake-build-release",
        help="Build directory",
    )
    ap.add_argument(
        "--commit",
        "-c",
        type=str,
        default=None,
        help="Check out dedicated commit",
    )
    ap.add_argument(
        "--force-delete",
        "-d",
        action="store_true",
        help="Delete cached tables",
    )
    ap.add_argument(
        "--mining-only",
        "-m",
        action="store_true",
        help="Do not acually run benchmarks, only perform mining",
    )
    ap.add_argument(
        "--multi-threaded",
        "-mt",
        action="store_true",
        help="Run benchmarks multi-threaded",
    )

    return ap.parse_args()


def main(output_path, force_delete, build_dir, commit, no_runs, multi_threaded):
    pwd = os.getcwd()
    if not os.path.isdir(build_dir):
        print(f"Could not find build directory {build_dir}\nDid you call the script from project root?")
        return
    dep_mining_plugin_path = os.path.join(os.path.abspath(build_dir), "lib", "libhyriseDependencyMiningPlugin.so")
    config_file = "dependency_config.json"
    config_path = os.path.join(build_dir, config_file)
    mining_config_file = "mining_config.json"
    mining_config_path = os.path.join(build_dir, mining_config_file)
    setup = ExperimentSetup()
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    if force_delete:
        print("Clear cached tables")
        cached_table_dirs = ["imdb_data", "tpch_cached_tables", "tpcds_cached_tables"]
        cached_dir_prefix = ".."
        for cached_table_dir in cached_table_dirs:
            cached_table_path = os.path.join(cached_dir_prefix, cached_table_dir)
            if not os.path.isdir(cached_table_path):
                continue
            shutil.rmtree(cached_table_path)

    if commit:
        print(f"Checkout {commit}")
        with Popen(f"git rev-parse {commit} | head -n 1", shell=True, stdout=PIPE) as p:
            p.wait()
            commit_ref = p.stdout.read().decode().strip()
        with Popen(f"git checkout {commit_ref}", shell=True) as p:
            p.wait()

    print("Build executables")
    os.chdir(build_dir)
    all_benchmark_string = " ".join(setup.benchmarks)
    build_command = f"ninja {all_benchmark_string} hyriseDependencyMiningPlugin"
    with Popen(build_command, shell=True) as p:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError("Could not build executables")
    os.chdir(pwd)

    num_cores = -1
    if multi_threaded:
        print("Get hardware specs")
        num_core_command = "(lscpu -p | egrep -v '^#' | grep '^[0-9]*,[0-9]*,0,0' | sort -u -t, -k 2,4 | wc -l)"
        with Popen(num_core_command, shell=True, stdout=PIPE) as p:
            p.wait()
            if p.returncode != 0:
                raise RuntimeError("Could not access num cores")
            try:
                num_cores = int(p.stdout.read().decode().strip())
            except ValueError:
                raise RuntimeError("Invalid CPU core count")

    for config_name, config in setup.configs.items():
        print(f"\n{'=' * 20}\n{config_name.upper()}\n{'=' * 20}")
        config.to_json(config_path)

        for benchmark in setup.benchmarks:
            benchmark_path = os.path.join(build_dir, benchmark)

            for scale_factor in setup.scale_factors:
                if benchmark != "hyriseBenchmarkTPCH" and scale_factor == 0.01:
                    continue
                if benchmark == "hyriseBenchmarkJoinOrder" and scale_factor != 10:
                    continue

                sf_flag = f"-s {scale_factor}" if benchmark != "hyriseBenchmarkJoinOrder" else ""
                sf_printable = str(scale_factor).replace(".", "")
                sf_extension = f"_s-{sf_printable}" if benchmark != "hyriseBenchmarkJoinOrder" else ""
                run_time = max(round(scale_factor * 120), 120) if multi_threaded else max(round(scale_factor * 6), 60)

                for mining_config in setup.mining_configs:
                    is_default_config = mining_config.is_default_config()

                    if config.allows_nothing() and not is_default_config:
                        continue

                    if mining_config.max_validation_candidates > 20 and benchmark != "hyriseBenchmarkTPCDS":
                        continue

                    if not is_default_config:
                        mining_config.to_json(mining_config_path)
                        mining_flag = f"--mining_config {mining_config_path}"
                        mining_title = f" and {mining_config.file_extension()}"
                    else:
                        mining_flag = ""
                        mining_title = ""

                    print(f"\nRunning {benchmark} for {config_name} with SF {scale_factor}{mining_title}...")
                    base_file_name = f"{benchmark}_{config_name}{sf_extension}{mining_config.file_extension()}"
                    log_path = os.path.join(output_path, f"{base_file_name}.log")
                    results_path = os.path.join(output_path, f"{base_file_name}.json")
                    plugin_flag = "" if config.allows_nothing() else f"--dep_mining_plugin {dep_mining_plugin_path}"
                    runs = 0 if no_runs else 100
                    warmup_flag = "-w 1" if runs > 0 else ""
                    output_flag = f"-o {results_path}" if runs > 0 else ""
                    mt_flags = f"--scheduler --clients {num_cores} --cores {num_cores} -m Shuffled"
                    st_flags = f"-r {runs} {warmup_flag}"

                    executed = False
                    exec_command = (
                        f"({benchmark_path} -t {run_time} {mt_flags if multi_threaded else st_flags} {sf_flag} "
                        + f"{output_flag} {plugin_flag} --dep_config {config_path} {mining_flag} 2>&1 ) "
                        + f"| tee {log_path}"
                    )
                    # print(exec_command)
                    with Popen(exec_command, shell=True) as p:
                        p.wait()
                    if not is_default_config:
                        os.remove(mining_config_path)
        os.remove(config_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.output_path, args.force_delete, args.build_dir, args.commit, args.mining_only, args.multi_threaded)
