import json
import pandas as pd
import numpy as np
import glob

from typing import List


# Class for loading results
class Result:

    def __init__(self, path):
        with open(path) as instream:
            doc = json.load(instream)
            for key, value in doc.items():
                setattr(self, key, value)
            if len(self.holonomic_costs) == 0:
                print("Cost vector for instance {} was empty".format(self.instance))
                self.holonomic_costs = [1e20]
                self.smooth_costs = [1e20]
            self.num_no_goods -= self.cusp_no_goods
            if len(self.plans) == 0:
                self.plan_length = None
            else:
                self.plan_length = len(self.plans[-1])


class NonDeterministicResult(object):

    def __init__(self, r):
        self.observations = [r]
        self.domain = self.observations[0].domain
        self.instance = self.observations[0].instance
        self.name = self.observations[0].name
        self.seed = self.observations[0].seed

        self.rgg_V = [None, None]
        self.rgg_E = [None, None]
        self.rgg_expansions = [None, None]
        self.rgg_dispersion = [None, None]

        self.num_no_goods = [None, None]
        self.num_cusp_no_goods = [None, None]

        self.planning_time = [None, None]
        self.verification_time = [None, None]

        self.iterations = [None, None]

        self.safe = 0.0

        self.holonomic_costs_0 = [None, None]
        self.holonomic_costs_k = [None, None]

        self.smooth_costs_0 = [None, None]
        self.smooth_costs_k = [None, None]

        self.plan_length = [None, None]

    def add(self, r):
        self.observations += [r]

    def compute_statistics(self):
        """
        Summarises information from observations
        :return:
        """
        def mean_and_std(attrib: str):
            return [np.mean([getattr(o, attrib) for o in self.observations]),
                    np.std([getattr(o, attrib) for o in self.observations])]

        def mean_and_std2(attrib: str, index: int):
            return [np.mean([getattr(o, attrib)[index] for o in self.observations]),
                    np.std([getattr(o, attrib)[index] for o in self.observations])]

        def mean_and_std3(attrib: str, key):
            return [np.mean([getattr(o, attrib)[key] for o in self.observations]),
                    np.std([getattr(o, attrib)[key] for o in self.observations])]

        def prob_event(attrib: str, v: bool):
            return np.sum([1 for o in self.observations if getattr(o, attrib) == v]) / len(self.observations)

        self.num_no_goods = mean_and_std('num_no_goods')
        self.num_cusp_no_goods = mean_and_std('cusp_no_goods')

        self.rgg_V = mean_and_std3('rgg', 'num_vertices')
        self.rgg_E = mean_and_std3('rgg', 'num_edges')
        self.rgg_expansions = mean_and_std3('rgg', 'expansions')
        self.rgg_dispersion = mean_and_std2('dispersion', -1)

        self.planning_time = mean_and_std('planning_time')
        self.verification_time = mean_and_std('verification_time')

        self.iterations = mean_and_std('iterations')

        self.safe = prob_event('safe', True)

        if self.safe > 0.0:
            self.holonomic_costs_0 = mean_and_std2('holonomic_costs', 0)
            self.holonomic_costs_k = mean_and_std2('holonomic_costs', -1)
            self.smooth_costs_0 = mean_and_std2('smooth_costs', 0)
            self.smooth_costs_k = mean_and_std2('smooth_costs', -1)
            self.plan_length = mean_and_std('plan_length')


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


def collect_non_deterministic_results(file_pattern: str):
    all_result_files = [f for f in glob.glob(file_pattern)]
    print("Found", len(all_result_files), "result files")
    all_results = {}

    expected = ['instance_{:03d}'.format(i) for i in range(300)]

    for filename in all_result_files:
        r = Result(filename)
        try:
            expected.remove(r.name)
        except ValueError:
            pass
        try:
            all_results[r.name].add(r)
        except KeyError:
            all_results[r.name] = NonDeterministicResult(r)

    print("Missing results:", len(expected))
    for entry in expected:
        print("\t", entry)

    for _, nd in all_results.items():
        nd.compute_statistics()

    return all_results.values()


def collect_deterministic_results(file_pattern: str):
    all_result_files = [f for f in glob.glob(file_pattern)]
    print("Found", len(all_result_files), "result files")

    expected = ['instance_{:03d}'.format(i) for i in range(300)]

    all_results = []

    for filename in all_result_files:
        all_results += [Result(filename)]
        try:
            expected.remove(all_results[-1].name)
        except ValueError:
            pass

    print("Missing results:", len(expected))
    for entry in expected:
        print("\t", entry)

    return all_results


def tabulate_deterministic_results(results: List[Result]):
    df = {'domain': [r.domain for r in results],
          'instance': [r.name for r in results],
          'seed': [r.seed for r in results],
          'num_no_goods': [r.num_no_goods for r in results],
          'num_cusp_no_goods': [r.cusp_no_goods for r in results],
          'rgg_V': [r.rgg['num_vertices'] for r in results],
          'rgg_E': [r.rgg['num_edges'] for r in results],
          'rgg_expansions': [r.rgg['expansions'] for r in results],
          'rgg_dispersion': [r.dispersion[-1] for r in results],
          'plan_time': [r.planning_time for r in results],
          'verif_time': [r.verification_time for r in results],
          'iterations': [r.iterations for r in results],
          'safe': [r.safe for r in results],
          'holonomic_cost_0': [r.holonomic_costs[0] for r in results],
          'holonomic_cost_k': [r.holonomic_costs[-1] for r in results],
          'smooth_cost_0': [r.smooth_costs[0] for r in results],
          'smooth_cost_k': [r.smooth_costs[-1] for r in results],
          'plan_length': [r.plan_length for r in results]}
    return pd.DataFrame(df)


def tabulate_non_deterministic_results(results: List[Result]):
    df = {'domain': [r.domain for r in results],
          'instance': [r.name for r in results],
          'N': [len(r.observations) for r in results],
          'mean(num_no_goods)': [r.num_no_goods[0] for r in results],
          'std(num_no_goods)': [r.num_no_goods[1] for r in results],
          'mean(num_cusp_no_goods)': [r.num_cusp_no_goods[0] for r in results],
          'std(num_cusp_no_goods)': [r.num_cusp_no_goods[1] for r in results],
          'mean(plan_time)': [r.planning_time[0] for r in results],
          'std(plan_time)': [r.planning_time[1] for r in results],
          'mean(verif_time)': [r.verification_time[0] for r in results],
          'std(verif_time)': [r.verification_time[1] for r in results],
          'mean(rgg_V)': [r.rgg_V[0] for r in results],
          'std(rgg_V)':  [r.rgg_V[1] for r in results],
          'mean(rgg_E)': [r.rgg_E[0] for r in results],
          'std(rgg_E)':  [r.rgg_E[1] for r in results],
          'mean(rgg_expansions)': [r.rgg_expansions[0] for r in results],
          'std(rgg_expansions)':  [r.rgg_expansions[1] for r in results],
          'mean(rgg_dispersion)': [r.rgg_dispersion[0] for r in results],
          'std(rgg_dispersion)':  [r.rgg_dispersion[1] for r in results],
          'mean(iterations)': [r.iterations[0] for r in results],
          'std(iterations)': [r.iterations[1] for r in results],
          'Pr(safe)': [r.safe for r in results],
          'mean(holonomic_cost_0)': [r.holonomic_costs_0[0] for r in results],
          'std(holonomic_cost_0)': [r.holonomic_costs_0[1] for r in results],
          'mean(holonomic_cost_k)': [r.holonomic_costs_k[0] for r in results],
          'std(holonomic_cost_k)': [r.holonomic_costs_k[1] for r in results],
          'mean(smooth_cost_0)': [r.smooth_costs_0[0] for r in results],
          'std(smooth_cost_0)': [r.smooth_costs_0[1] for r in results],
          'mean(smooth_cost_k)': [r.smooth_costs_k[0] for r in results],
          'std(smooth_cost_k)': [r.smooth_costs_k[1] for r in results],
          'mean(plan_length)': [r.plan_length[0] for r in results],
          'std(plan_length)': [r.plan_length[1] for r in results]}
    return pd.DataFrame(df)

