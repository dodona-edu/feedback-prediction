import multiprocessing
import time
import pickle
from glob import glob
from typing import Dict, List, Set, Counter, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from analyze import FeedbackAnalyzer
from custom_types import AnnotatedTree
from feedback_model import FeedbackModel
from tester import test_all_files
from constants import ROOT_DIR, COLORS, ENGLISH_EXERCISE_NAMES_MAP, DUTCH_EXERCISE_NAMES_MAP


def get_annotation_ids(dataset: Dict[str, AnnotatedTree]) -> Set[int]:
    annotation_ids = set()
    for _, annotation_instances in dataset.values():
        annotation_ids.update(a_id for a_id, _ in annotation_instances)

    return annotation_ids


def get_not_seen_count(train_annotation_ids: Set[int], total_per_annotation: Counter) -> int:
    not_seen_count = 0
    for (a_id, count) in total_per_annotation.items():
        if a_id not in train_annotation_ids:
            not_seen_count += count

    return not_seen_count


def get_no_patterns_count(model: FeedbackModel, train_annotation_ids: Set[int], total_per_annotation: Counter) -> int:
    no_patterns_count = 0
    for (a_id, count) in total_per_annotation.items():
        if a_id in train_annotation_ids and a_id not in model.patterns:
            no_patterns_count += count

    return no_patterns_count


def simulate_online(analyzer: FeedbackAnalyzer) -> Tuple[List[Tuple[int, int]], List[int], Dict[str, List[int]], List[int], List[Tuple[float, Tuple[float, float, float]]]]:
    set_sizes = []  # Strings of amount of files in (train, test) set
    totals = []
    match_counts = {
        "First": [],
        "Top 5": [],
        "Out of top 5": [],
        "No patterns": [],
        "No training instances": []
    }
    unique_training_counts = []
    times_exercise = []

    model = FeedbackModel()

    test = analyzer.analyze_files()
    training = {}
    files = analyzer.get_sorted_files()

    while len(files) > 5:
        for file in files[0:5]:
            feedback_tree = test.pop(file)
            training[file] = feedback_tree
        files = files[5:]

        print(f"Training files: {len(training)}; Test files: {len(test)}")

        start = time.time()
        model.train(training)
        end = time.time()
        total_per_annotation, first_n_per_annotation, total_first_n_per_annotation, times = test_all_files(test, model)

        total = sum(total_per_annotation.values())
        train_annotation_ids = get_annotation_ids(training)

        first_count = sum(first_n_per_annotation[0].values())
        first_n_count = sum(total_first_n_per_annotation.values()) - first_count
        not_seen_count = get_not_seen_count(train_annotation_ids, total_per_annotation)
        no_patterns_count = get_no_patterns_count(model, train_annotation_ids, total_per_annotation)
        failed_count = total - first_count - first_n_count - not_seen_count - no_patterns_count

        match_counts["First"].append(first_count)
        match_counts["Top 5"].append(first_n_count)
        match_counts["Out of top 5"].append(failed_count)
        match_counts["No patterns"].append(no_patterns_count)
        match_counts["No training instances"].append(not_seen_count)

        unique_training_counts.append(len(train_annotation_ids))

        set_sizes.append((len(training), len(test)))
        totals.append(total)

        times_exercise.append((end - start, (min(times), sum(times) / len(times), max(times))))

    print(times_exercise)

    return set_sizes, totals, match_counts, unique_training_counts, times_exercise


def plot_simulation(e_id: str, stats: Tuple[List[Tuple[int, int]], List[int], Dict[str, List[int]], List[int], List[Tuple[float, Tuple[float, float, float]]]], file_name: str = "simulate"):
    bar_titles, totals, match_counts, unique_training_counts, _ = stats
    bar_titles = [str(bar_title) for bar_title in bar_titles]

    px = 1 / plt.rcParams['figure.dpi']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1280 * px, 960 * px), gridspec_kw={'width_ratios': [3, 1]})

    colors = [
        COLORS["DARK GREEN"],
        COLORS["LIGHT GREEN"],
        COLORS["DARK RED"],
        COLORS["GRAY"],
        COLORS["DARK GRAY"]
    ]

    left = np.zeros(len(bar_titles))
    for (bar_label, counts), color in zip(match_counts.items(), colors):
        percentages = [count / total for (count, total) in zip(counts, totals)]
        ax1.barh(bar_titles, percentages, label=bar_label, left=left, color=color)
        left += percentages

    for bars, counts in zip(ax1.containers, match_counts.values()):
        labels = [c if c > 0 else "" for c in counts]
        ax1.bar_label(bars, labels, label_type='center', color='white')

    ax1.set_title(f"Simulation of exercise {NAMES_MAP[e_id]}", pad=50)
    ax1.set_xlim([0, 1])

    ax1.set_xlabel("Percentage of annotation instances")
    ax1.set_ylabel("Files in (train, test) sets", rotation=0, horizontalalignment='left', y=1.02)

    ax1.invert_yaxis()
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1.05), ncols=5)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.barh(bar_titles, unique_training_counts)

    max_count = max(unique_training_counts)
    ax2.set_yticks([])
    ax2.set_xlim([0, max_count])
    tick_spacing = (10 if max_count >= 20 else 5) if e_id != '2146239081' else 8
    ticks = [i for i in range(0, max_count, tick_spacing)]
    ax2.set_xticks([*ticks, max_count])

    ax2.set_xlabel("# annotations encountered during training")

    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.savefig(f'{ROOT_DIR}/output/plots/simulation/{file_name}.png', bbox_inches='tight', dpi=150)


def plot_timings(e_id: str, stats: Tuple[List[Tuple[int, int]], List[Tuple[float, Tuple[float, float, float]]]], file_name: str = "timing"):
    bar_titles, timings = stats

    train_x = list(map(lambda x: str(x[0]), bar_titles))
    test_x = list(map(lambda x: str(x[1]), bar_titles))

    train_times = list(map(lambda x: x[0], timings))
    test_times = list(map(lambda x: x[1], timings))
    test_times_min = list(map(lambda x: x[0], test_times))
    test_times_avg = list(map(lambda x: x[1], test_times))
    test_times_max = list(map(lambda x: x[2], test_times))

    px = 1 / plt.rcParams['figure.dpi']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(1280 * px, 1280 * px))

    ax1.scatter(train_x, train_times, c="C1", zorder=10, clip_on=False)

    ax1.set_ylim(bottom=0)

    ax1.set_xlabel("Files in train set", loc="right")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Training times")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid()
    ax1.set_axisbelow(True)

    ax2.errorbar(test_x, test_times_avg, yerr=[test_times_min, test_times_max], fmt='o', capsize=6, ecolor="C0", c="C1", zorder=10, clip_on=False)

    ax2.set_ylim(bottom=0)

    ax2.set_xlabel("Files in test set", loc="right")
    ax2.set_ylabel("Time (s)")
    ax2.set_title(f"Testing times: (min, avg, max) per click")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid()
    ax2.set_axisbelow(True)

    plt.suptitle(f"Simulation timings of exercise {NAMES_MAP[e_id]}", fontsize=18)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{ROOT_DIR}/output/plots/simulation/{file_name}.png', bbox_inches='tight', dpi=150)


def plot_top5_evolution_line_graph(results_per_exercise: Dict[str, Tuple[List[int], List[float]]], file_name="evolution"):
    train_set_sizes = max(results_per_exercise.values(), key=lambda x: len(x[0]))[0]

    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(1, 1, figsize=(1280 * px, 640 * px))

    for e_id, (train_sizes, percentages) in results_per_exercise.items():
        ax.plot(train_sizes, percentages, label=NAMES_MAP[e_id])

    ax.set_xticks(range(0, train_set_sizes[-1] + 1, 10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlim(left=0, right=train_set_sizes[-1])
    ax.set_ylim(bottom=0, top=1)
    ax.grid(which='both')

    ax.set_title("Evolution of top 5 percentage during simulation")

    ax.set_xlabel("Files in train set")
    ax.set_ylabel("Top 5 percentage")

    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{ROOT_DIR}/output/plots/simulation/{file_name}.png', bbox_inches='tight', dpi=150)


def main_simulate(e_id: str, save_stats=False, load_stats=False):
    stats_file = f'{ROOT_DIR}/output/stats/{e_id}_simulation'

    if load_stats:
        with open(stats_file, 'rb') as stats_file:
            stats = pickle.load(stats_file)
    else:
        analyzer = FeedbackAnalyzer()
        analyzer.set_files(glob(f'{ROOT_DIR}/data/exercises/{e_id}/*.py'))

        stats = simulate_online(analyzer)
        if save_stats:
            with open(stats_file, 'wb') as stats_file:
                pickle.dump(stats, stats_file, pickle.HIGHEST_PROTOCOL)

    plot_simulation(e_id, stats, file_name=f"{prefix}_{e_id}")
    plot_timings(e_id, (stats[0], stats[4]), file_name=f"{prefix}_{e_id}_timings")

    return stats


def main_line_graph(e_ids: List[str], results):
    results_per_exercise = {}
    for e_id, result in zip(e_ids, results):
        set_sizes, totals, match_counts, _, _ = result
        train_sizes = [train_size for train_size, _ in set_sizes]
        percentages = [(first + top_n) / total for first, top_n, total in zip(match_counts["First"], match_counts["Top 5"], totals)]
        results_per_exercise[e_id] = ([0] + train_sizes, [0] + percentages)

    plot_top5_evolution_line_graph(results_per_exercise, file_name=f"{prefix}_line")


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    NAMES_MAP = DUTCH_EXERCISE_NAMES_MAP

    prefix = "simulation"
    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']

    all_results = []
    for eid in ids:
        print(f"Exercise {eid}")
        s = time.time()
        res = main_simulate(eid, load_stats=False, save_stats=False)
        all_results.append(res)
        print(f"Total time: {time.time() - s}")

    main_line_graph(ids, all_results)
