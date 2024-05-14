import multiprocessing
import pickle
import time
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from analyze import PylintAnalyzer, FeedbackAnalyzer, Analyzer
from feedback_model import FeedbackModel
from tester import test_all_files
from constants import COLORS, ROOT_DIR, ENGLISH_EXERCISE_NAMES_MAP, DUTCH_EXERCISE_NAMES_MAP

labels_list = ['Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5', 'Rank > 5']

colors = [
    COLORS["DARK GREEN"],
    COLORS["LIGHT GREEN"],
    COLORS["LIGHT YELLOW"],
    COLORS["LIGHT ORANGE"],
    COLORS["DARK ORANGE"],
    COLORS["DARK RED"]
]

timings = {}


def gather_data(analyzer: Analyzer, eid: str):
    train, test = analyzer.create_train_test_set()
    model = FeedbackModel()

    start_train = time.time()
    model.train(train, n_procs=N_PROCS)
    end_train = time.time()

    start_test = time.time()
    total_per_annotation, first_n_per_annotation, _, times = test_all_files(test, model, n=5, n_procs=N_PROCS)
    end_test = time.time()

    timings[eid] = (end_train - start_train, end_test - start_test, (min(times), sum(times) / len(times), max(times)))

    total = sum(total_per_annotation.values())
    percentages = [sum(value.values()) / total for value in first_n_per_annotation]
    percentages.append(1 - sum(percentages))

    return percentages, len(total_per_annotation.keys()), total


def plot_global_accuracies_stacked_bar(results: Dict[str, Tuple[List[float], int, int]], file_name="plot"):
    percentages_per_category = list([list(reversed([result[0][i] for result in results.values()])) for i in range(6)])

    eids = list(reversed(results.keys()))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    left = np.zeros(len(results))

    width = 0.5

    for color, percentages, label in zip(colors, percentages_per_category, labels_list):
        ax.barh(eids, percentages, width, label=label, left=left, color=color)
        left += percentages

    for bars, percentages in zip(ax.containers, percentages_per_category):
        labels = [f'{p * 100:.1f}%' if p > 0.05 else "" for p in percentages]
        ax.bar_label(bars, labels, label_type='center', color='white')

    for bar, eid in zip(ax.containers[0], eids):
        exercise_name = NAMES_MAP[eid] if eid in NAMES_MAP else eid
        plt.text(-0.02, bar.get_y() + bar.get_height() / 2, exercise_name, ha="right", va="center")
        plt.text(1.02, bar.get_y() + bar.get_height() / 2, f"{results[eid][1]};{results[eid][2]}", ha="left", va="center")

    ax.set_xlim((0, 1))
    ax.set_yticks([])

    plt.text(1.12, 1.4, "# test annotations ; instances", rotation=-90)

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), ncols=6)

    fig.tight_layout()

    plt.savefig(f'{ROOT_DIR}/output/plots/global/{file_name}.png', bbox_inches='tight', dpi=150)


def print_timings():
    print()
    print("Timings: ")
    for exercise_id, (train_time, test_time, (min_click, avg_click, max_click)) in timings.items():
        print(f"Exercise {NAMES_MAP[exercise_id] if exercise_id in NAMES_MAP else exercise_id} ({exercise_id}):")
        print(f"training time:  {train_time:.5f}")
        print(f"testing time:   {test_time:.5f}")
        print(f"min time/click: {min_click:.5f}")
        print(f"avg time/click: {avg_click:.5f}")
        print(f"max time/click: {max_click:.5f}")
        print()


def main_stacked_bars(exercise_ids: List[str], is_pylint: bool = False, load_data: bool = False, save_data: bool = False):
    stats_file = f"{STATS_FILE}{'_pylint' if is_pylint else ''}"
    analyzer = PylintAnalyzer() if is_pylint else FeedbackAnalyzer()
    results_per_exercise_id = {}

    if load_data:
        with open(f'{ROOT_DIR}/output/stats/{stats_file}', 'rb') as save_file:
            results_per_exercise_id = pickle.load(save_file)
    else:
        for eid in exercise_ids:
            print(f"Results for excercise with ID {eid}")

            analyzer.set_files(glob(f'{ROOT_DIR}/data/exercises/{eid}/*.py'))
            results_per_exercise_id[eid] = gather_data(analyzer, eid)

        if is_pylint:
            analyzer.set_files(glob(f'{ROOT_DIR}/data/exercises/*/*.py'))
            results_per_exercise_id["Combined"] = gather_data(analyzer, 'Combined')

        print_timings()

        if save_data:
            with open(f'{ROOT_DIR}/output/stats/{stats_file}', 'wb') as save_file:
                pickle.dump(results_per_exercise_id, save_file, protocol=pickle.HIGHEST_PROTOCOL)

    plot_global_accuracies_stacked_bar(results_per_exercise_id, file_name=f"{prefix}_plot{'_pylint' if is_pylint else ''}")


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    STATS_FILE = f"stacked_bars_global"
    NAMES_MAP = DUTCH_EXERCISE_NAMES_MAP
    N_PROCS = 8

    prefix = "no_identifying"
    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']

    main_stacked_bars(ids, is_pylint=False)
