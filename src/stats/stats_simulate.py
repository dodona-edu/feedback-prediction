import pickle
from glob import glob
from typing import Dict, List, Set, Counter, Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.analyze import FeedbackAnalyzer
from src.custom_types import FeedbackTree
from src.feedback_model import FeedbackModel
from src.tester import test_all_files
from src.constants import ROOT_DIR, COLORS


def get_messages(dataset: Dict[str, FeedbackTree]) -> Set[str]:
    messages = set()
    for (_, annotations) in dataset.values():
        messages.update(annotation[0] for annotation in annotations)

    return messages


def get_not_seen_count(train_messages: Set[str], total_per_message: Counter) -> int:
    not_seen_count = 0
    for (m, count) in total_per_message.items():
        if m not in train_messages:
            not_seen_count += count

    return not_seen_count


def simulate_online(analyzer: FeedbackAnalyzer) -> Tuple[List[str], List[int], List[List[int]], List[List[int]]]:
    set_sizes = []  # Strings of amount of files in (train, test) set
    totals = []
    match_counts = [[], [], [], []]
    annotation_counts = [[], []]

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
        model.train(training)
        total_per_message, first_n_per_message, total_first_n_per_message = test_all_files(test, model)

        total = sum(total_per_message.values())
        train_messages = get_messages(training)
        test_messages = get_messages(test)

        first_count = sum(first_n_per_message[0].values())
        first_n_count = sum(total_first_n_per_message.values()) - first_count
        not_seen_count = get_not_seen_count(train_messages, total_per_message)
        failed_count = total - not_seen_count - first_count - first_n_count

        match_counts[0].insert(0, first_count)
        match_counts[1].insert(0, first_n_count)
        match_counts[2].insert(0, failed_count)
        match_counts[3].insert(0, not_seen_count)

        annotation_counts[0].insert(0, len(train_messages))
        annotation_counts[1].insert(0, len(test_messages))

        set_sizes.insert(0, str((len(training), len(test))))
        totals.insert(0, total)

    return set_sizes, totals, match_counts, annotation_counts


def plot_simulation(e_id: str, stats: Tuple[List[str], List[int], List[List[int]], List[List[int]]], file_name: str = "simulate"):
    bar_titles, totals, match_counts, annotation_counts = stats

    px = 1/plt.rcParams['figure.dpi']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1280 * px, 960 * px), gridspec_kw={'width_ratios': [3, 1]})

    match_stats = [[count / total for count, total in zip(match_counts[i], totals)] for i in range(len(match_counts))]

    left = np.zeros(len(bar_titles))
    ax1.barh(bar_titles, match_stats[0], label="First", left=left, color=COLORS[0])
    left += match_stats[0]
    ax1.barh(bar_titles, match_stats[1], label="Top 5", left=left, color=COLORS[1])
    left += match_stats[1]
    ax1.barh(bar_titles, match_stats[2], label="Out of top 5", left=left, color=COLORS[-1])
    left += match_stats[2]
    ax1.barh(bar_titles, match_stats[3], label="Not yet seen", left=left, color=COLORS[3])

    n = len(match_counts[0])
    for i, bar in enumerate(ax1.patches):
        match_count = match_counts[i // n][i % n]
        if match_count > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, match_count)

    fig.suptitle(f"Simulation of exercise {e_id}")
    ax1.set_xlim([0, 1])

    ax1.set_xlabel("Percentage of messages")
    ax1.set_ylabel("Amount of files in (train, test) sets", rotation=0, horizontalalignment='left', y=1.02)

    ax1.legend(bbox_to_anchor=(1, 1.13))

    left = np.zeros(len(bar_titles))
    ax2.barh(bar_titles, annotation_counts[0], label="Train messages", left=left)
    left += annotation_counts[0]
    ax2.barh(bar_titles, annotation_counts[1], label="Test messages", left=left)

    ax2.set_yticks([])
    max_count = 0
    for i in range(len(annotation_counts[0])):
        count = annotation_counts[0][i] + annotation_counts[1][i]
        if count > max_count:
            max_count = count
    ax2.set_xlim([0, max_count])

    ax2.set_xlabel("Amount of unique messages")

    ax2.legend(bbox_to_anchor=(1, 1.1))

    fig.tight_layout()
    plt.savefig(f'{ROOT_DIR}/output/plots/{file_name}.png', bbox_inches='tight')


def main_simulate(e_id: str, save_stats=False, load_stats=False):
    if load_stats:
        with open(f'{ROOT_DIR}/output/stats/{e_id}_simulation', 'rb') as stats_file:
            stats = pickle.load(stats_file)
    else:
        analyzer = FeedbackAnalyzer()
        analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{e_id}/*.py'))

        stats = simulate_online(analyzer)
        if save_stats:
            with open(f'{ROOT_DIR}/output/stats/{e_id}_simulation', 'wb') as stats_file:
                pickle.dump(stats, stats_file, pickle.HIGHEST_PROTOCOL)

    plot_simulation(e_id, stats, file_name=f"__simulation_{e_id}")


if __name__ == '__main__':
    # TODO probleem met ex 1875043169, rond 50 train files duurt heel lang?
    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']
    # for eid in ids:
    main_simulate('505886137', load_stats=False)
