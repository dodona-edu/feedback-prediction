import pickle
from glob import glob
from typing import Dict, List, Set, Counter, Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.analyze import FeedbackAnalyzer
from src.custom_types import AnnotatedCode
from src.feedback_model import FeedbackModel
from src.tester import test_all_files
from src.constants import ROOT_DIR, COLORS


def get_messages(dataset: Dict[str, AnnotatedCode]) -> Set[str]:
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


def get_no_patterns_count(model: FeedbackModel, train_messages, total_per_message: Counter) -> int:
    no_patterns_count = 0
    for (m, count) in total_per_message.items():
        if m in train_messages and m not in model.patterns:
            no_patterns_count += count

    return no_patterns_count


def simulate_online(analyzer: FeedbackAnalyzer) -> Tuple[List[str], List[int], Dict[str, List[int]], List[int]]:
    set_sizes = []  # Strings of amount of files in (train, test) set
    totals = []
    match_counts = {
        "First": [],
        "Top 5": [],
        "Out of top 5": [],
        "No patterns found": [],
        "Not yet seen": []
    }
    unique_training_counts = []

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

        first_count = sum(first_n_per_message[0].values())
        first_n_count = sum(total_first_n_per_message.values()) - first_count
        not_seen_count = get_not_seen_count(train_messages, total_per_message)
        no_patterns_count = get_no_patterns_count(model, train_messages, total_per_message)
        failed_count = total - first_count - first_n_count - not_seen_count - no_patterns_count

        match_counts["First"].append(first_count)
        match_counts["Top 5"].append(first_n_count)
        match_counts["Out of top 5"].append(failed_count)
        match_counts["No patterns found"].append(no_patterns_count)
        match_counts["Not yet seen"].append(not_seen_count)

        unique_training_counts.append(len(train_messages))

        set_sizes.append(str((len(training), len(test))))
        totals.append(total)

    return set_sizes, totals, match_counts, unique_training_counts


def plot_simulation(e_id: str, stats: Tuple[List[str], List[int], Dict[str, List[int]], List[int]], file_name: str = "simulate"):
    bar_titles, totals, match_counts, unique_training_counts = stats

    px = 1/plt.rcParams['figure.dpi']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1280 * px, 960 * px), gridspec_kw={'width_ratios': [3, 1]})

    colors = [
        COLORS["DARK GREEN"],
        COLORS["LIGHT GREEN"],
        COLORS["DARK RED"],
        COLORS["GRAY"],
        COLORS["LIGHT ORANGE"]
    ]

    left = np.zeros(len(bar_titles))
    for (bar_label, counts), color in zip(match_counts.items(), colors):
        percentages = [count / total for (count, total) in zip(counts, totals)]
        ax1.barh(bar_titles, percentages, label=bar_label, left=left, color=color)
        left += percentages

    for bars, counts in zip(ax1.containers, match_counts.values()):
        labels = [c if c > 0 else "" for c in counts]
        ax1.bar_label(bars, labels, label_type='center', color='white')

    ax1.set_title(f"Simulation of exercise {e_id}", pad=50)
    ax1.set_xlim([0, 1])

    ax1.set_xlabel("Percentage of messages")
    ax1.set_ylabel("Files in (train, test) sets", rotation=0, horizontalalignment='left', y=1.02)

    ax1.invert_yaxis()
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1.05), ncols=5)

    ax2.barh(bar_titles, unique_training_counts)

    max_count = max(unique_training_counts)
    ax2.set_yticks([])
    ax2.set_xlim([0, max_count])
    ticks = [i for i in range(0, max_count, 10 if max_count >= 20 else 5)]
    ax2.set_xticks([*ticks, max_count])

    ax2.set_xlabel("Unique messages in train")

    ax2.invert_yaxis()

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
