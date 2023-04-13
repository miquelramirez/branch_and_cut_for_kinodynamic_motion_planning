import json
import pandas as pd
import numpy as np
import glob
import os
import itertools

from typing import List, Tuple, Any
from json import JSONDecodeError


def match_from_path(path, values):
    """
    Returns sequence type from path
    :param path:
    :return:
    """
    tail = os.path.basename(path)
    for type in values:
        if tail.find(type) >= 0:
            return type
    return 'unknown'


# Class for loading results
class Result:

    def __init__(self, path):
        self.domain = None
        self.name = None
        self.seed = None
        self.num_no_goods = None
        self.rgg = {}
        self.planning_time = None
        self.verification_time = None
        self.exact_verification_time = None
        self.iterations = None
        self.dispersion = []
        self.step_size = None
        self.valid = True
        self.exact_verification_time = None
        self.plans = []
        self.holonomic_costs = []
        self.instance = None
        self.elapsed_time = None
        self.max_speed = None

        self.sequence = match_from_path(path, ['halton', 'uniform'])
        self.no_good_type = match_from_path(path, ['single_edge', 'multi_edge'])
        self.solver = match_from_path(path, ['a_star', 'cp_sat', 'pulse'])
        self.direction = match_from_path(path, ['bk', 'gammell'])
        self.constraint_check_type = match_from_path(path, ['approx', 'polytrace'])
        if 'lazy_prm_bc' in path:
            self.constraint_check_type = 'polytrace'

        with open(path) as instream:
            doc = json.load(instream)
            for key, value in doc.items():
                setattr(self, key, value)
            if len(self.holonomic_costs) == 0:
                print("Cost vector for instance {} was empty".format(self.instance))
                self.holonomic_costs = [1e20]
                self.smooth_costs = [1e20]
            #self.num_no_goods -= self.cusp_no_goods
            if len(self.plans) == 0:
                self.plan_length = None
            else:
                self.plan_length = len(self.plans[-1])
        if self.elapsed_time is None:
            #print("WARNING: Instance data file {} had missing elapsed_time field".format(path))
            self.elapsed_time = self.planning_time + self.verification_time


class InstanceMetadata(object):

    def __init__(self, path):
        with open(path) as instream:
            doc = json.load(instream)
            self.name = doc['name']
            #for key, value in doc["metrics"]["normalized"].items():
            for key, value in doc["metrics"].items():
                setattr(self, key, value)


def collect_instance_metadata(file_pattern: str):
    all_metadata_files = [f for f in glob.glob(file_pattern)]
    print("Found metadata for", len(all_metadata_files), "instances")
    all_instances_metadata = {}

    for filename in all_metadata_files:
        m = InstanceMetadata(filename)
        all_instances_metadata[m.name] = m
    return all_instances_metadata.values()


def tabulate_instance_metadata(metric_data: List[InstanceMetadata]):
    df = {'instance': [m.name for m in metric_data],
          'distance_closest_obstacle': [m.distance_closest_obstacle for m in metric_data],
          'average_visibility': [m.average_visibility for m in metric_data],
          'dispersion': [m.dispersion for m in metric_data],
          'characterisitic_dimensions': [m.characteristic_dimensions for m in metric_data],
          'tortuosity': [m.tortuosity for m in metric_data]}
    return pd.DataFrame(df)


def collect_deterministic_results(files: List[str], instances: List[int] = None, seeds: List[int] = None):

    all_result_files = []

    for file_pattern in files:
        all_result_files = [f for f in glob.glob(file_pattern)]

    print("Found", len(all_result_files), "result files")

    if instances is None:
        instances = [i for i in range(300)]

    if seeds is None:
        expected = {'instance_{:03d}'.format(i): 'instance_{:03d}.seed_{}'.format(i, '*') for i in instances}
    else:
        expected = {('instance_{:03d}'.format(i), s): 'instance_{:03d}.seed_{}'.format(i, s) for i in instances for s in seeds}

    all_results = []

    for filename in all_result_files:

        try:
            r = Result(filename)
            all_results += [r]
        except JSONDecodeError as e:
            print("Error decoding JSON file:", filename)
            print("Message:", str(e))

        try:
            if seeds is None:
                del expected[all_results[-1].name]
            else:
                del expected[all_results[-1].name, all_results[-1].seed]
        except KeyError:
            pass
            #print(filename)

    print("Missing results:", len(expected))
    for entry in expected:
        print("\t", entry)

    return all_results


def tabulate_deterministic_results(results: List[Result]):
    df = {'domain': [r.domain for r in results],
          'instance': [r.name for r in results],
          'seed': [r.seed for r in results],
          'sequence': [r.sequence for r in results],
          'max_speed': [r.max_speed for r in results],
          'no_good_type': [r.no_good_type for r in results],
          'solver': [r.solver for r in results],
          'direction': [r.direction for r in results],
          'check_type': [r.constraint_check_type for r in results],
          'num_no_goods': [r.num_no_goods for r in results],
          'rgg_V': [r.rgg['num_vertices'] for r in results],
          'rgg_E': [r.rgg['num_edges'] for r in results],
          'rgg_expansions': [r.rgg['expansions'] for r in results],
          'rgg_dispersion': [r.dispersion[-1] for r in results],
          'elapsed_time': [r.elapsed_time for r in results],
          'plan_time': [r.planning_time for r in results],
          'verif_time': [r.verification_time for r in results],
          'iterations': [r.iterations for r in results],
          'valid': [r.valid for r in results],
          'exact_verification_time': [r.exact_verification_time for r in results],
          'step_size': [r.step_size for r in results],
          'holonomic_cost_0': [r.holonomic_costs[0] for r in results],
          'holonomic_cost_k': [r.holonomic_costs[-1] for r in results],
          'smooth_cost_0': [r.smooth_costs[0] for r in results],
          'smooth_cost_k': [r.smooth_costs[-1] for r in results],
          'plan_length': [r.plan_length for r in results]}
    return pd.DataFrame(df)


def tabulate_results_by_step_size(table: pd.DataFrame):
    """
    Tabulates result by the values of 'step size' parameter
    :param table:
    :return:
    """
    step_sizes = np.unique(table['step_size'].values)
    step_size_tables = {}
    for v in step_sizes:
        step_size_tables[v] = table[table['step_size'] == v]
    return step_sizes, step_size_tables


def tabulate_results_by(table: pd.DataFrame, columns: Tuple[str]):
    """
    Tabulates results according to sequence of column names
    :param table:
    :param columns:
    :return:
    """
    key_domains = [np.unique(table[key_name]) for key_name in columns]
    print(key_domains)

    tables = {}
    for t in itertools.product(*key_domains):
        result = table[table[columns[0]] == t[0]]
        for key_idx in range(1, len(columns)):
            result = result[result[columns[key_idx]] == t[key_idx]]
        tables[t] = result
    print("Tables generated:", len(tables))
    return tables.keys(), tables


def tabulate_coverage(configs, tables):
    """
    Creates simple table listing coverages

    :param configs:
    :param tables:
    :return:
    """
    df = {
        'config': [],
        'total': [],
        'valid': [],
        'invalid': []
    }

    for v in configs:

        valid_set = select_valid_instances(tables[v])
        invalid_set = select_invalid_instances(tables[v])
        if len(tables[v]) == 0:
            print("Ignoring empty table for:", v)
            continue
        #print(v, "invalid:", len(invalid_set), "valid:", len(valid_set))
        df['config'] += [v]
        df['total'] += [len(tables[v])]
        df['valid'] += [len(valid_set)]
        df['invalid'] += [len(invalid_set)]

    return pd.DataFrame(df)

def select_invalid_instances(table: pd.DataFrame):
    return table[table['valid'] == False]


def select_valid_instances(table: pd.DataFrame):
    return table[table['valid'] == True]


def tabulate_reliability(step_sizes, tables):
    df = {
        'step_size': [],
        'total': [],
        'valid': [],
        'invalid': []
    }
    for v in step_sizes:
        valid_set = select_valid_instances(tables[v])
        invalid_set = select_invalid_instances(tables[v])
        print(v, "invalid:", len(invalid_set), "valid:", len(valid_set))
        df['step_size'] += [v]
        df['total'] += [len(tables[v])]
        df['valid'] += [len(valid_set)]
        df['invalid'] += [len(invalid_set)]

    return pd.DataFrame(df)


#default_time_breakpoints = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
default_time_breakpoints = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]


def coverage_over_time(table: pd.DataFrame, time_column='elapsed_time', time_points=default_time_breakpoints):
    """
    Produces coverage over time curves
    :param table:
    :param time_column:
    :param time_points:
    :return:
    """
    time_records = table[time_column].values
    coverage_at_breakpoint = np.zeros(len(time_points), dtype=int)

    for i in range(len(time_records)):
        for j in range(len(time_points)):
            if time_records[i] <= time_points[j]:
                coverage_at_breakpoint[j] += 1

    return coverage_at_breakpoint