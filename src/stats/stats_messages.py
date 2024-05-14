import collections
import pickle
from glob import glob
from typing import List, Tuple, Counter, Dict

import numpy as np
from matplotlib import pyplot as plt

from analyze import Analyzer, PylintAnalyzer
from custom_types import AnnotatedTree
from feedback_model import FeedbackModel
from tester import test_all_files
from constants import COLORS, ROOT_DIR


def gather_stats_for_annotations(analyzer: Analyzer, annotations: List[str], load_stats, save_stats) -> Tuple[Counter, List[Counter], Counter, Counter, Dict[int, str]]:
    stats_file_name = f"messages_stats"

    if load_stats:
        with open(f'{ROOT_DIR}/output/stats/{stats_file_name}', 'rb') as stats_file:
            stats = pickle.load(stats_file)
    else:
        training, test = analyzer.create_train_test_set()
        model = FeedbackModel()
        model.train(training)

        filtered_test = {}
        for file in test:
            for (a_id, line) in test[file][1]:
                if a_id in analyzer.id_annotation_map and analyzer.id_annotation_map[a_id] in annotations:
                    if file not in filtered_test:
                        filtered_test[file] = (test[file][0], [])
                    filtered_test[file][1].append((a_id, line))

        total, first_n, total_first_n, _ = test_all_files(filtered_test, model, n=5)

        total_training = determine_training_totals(analyzer.train)

        stats = (total, first_n, total_first_n, total_training, analyzer.id_annotation_map)

        if save_stats:
            with open(f'{ROOT_DIR}/output/stats/{stats_file_name}', 'wb') as stats_file:
                pickle.dump(stats, stats_file, pickle.HIGHEST_PROTOCOL)

    return stats


def determine_training_totals(training: Dict[str, AnnotatedTree]):
    total = collections.Counter()
    for (_, annotation_instances) in training.values():
        for a_id, _ in annotation_instances:
            total[a_id] += 1

    return total


def plot_accuracies_for_annotations(analyzer: Analyzer, annotations: List[str], load_stats=False, save_stats=True) -> None:
    total, first_n, total_first_n, total_training, id_annotation_map = gather_stats_for_annotations(analyzer, annotations, load_stats, save_stats)

    annotation_ids = [a_id for a_id in total]

    def calculate_percentages(a_id):
        result = [first_n[pos][a_id] / total[a_id] for pos in range(5)]
        result.append((total[a_id] - total_first_n[a_id]) / total[a_id])
        return result

    percentages_per_annotation = [(a_id, calculate_percentages(a_id)) for a_id in annotation_ids]
    percentages_per_annotation.sort(key=lambda x: (sum(x[1][:-1]), x[1]))

    def get_percentages(pos):
        return list(map(lambda x: x[1][pos], percentages_per_annotation))

    positions = {
        "Rank 1": get_percentages(0),
        "Rank 2": get_percentages(1),
        "Rank 3": get_percentages(2),
        "Rank 4": get_percentages(3),
        "Rank 5": get_percentages(4),
        "Rank > 5": get_percentages(5)
    }

    width = 0.5
    fig, ax = plt.subplots(figsize=(10.7, 6))
    left = np.zeros(len(annotation_ids))

    bar_titles = [f"{id_annotation_map[a_id]} ({total_training[a_id]};{total[a_id]})" for a_id, _ in percentages_per_annotation]

    colors = [
        COLORS["DARK GREEN"],
        COLORS["LIGHT GREEN"],
        COLORS["LIGHT YELLOW"],
        COLORS["LIGHT ORANGE"],
        COLORS["DARK ORANGE"],
        COLORS["DARK RED"]
    ]

    for color, (position, counts) in zip(colors, positions.items()):
        ax.barh(bar_titles, counts, width, label=position, left=left, color=color)
        left += counts

    for bars, counts in zip(ax.containers, positions.values()):
        labels = [f'{p * 100:.1f}%' if p > 0.05 else "" for p in counts]
        ax.bar_label(bars, labels, label_type='center', color='white')

    for bar, label in zip(ax.containers[0], bar_titles):
        plt.text(-0.02, bar.get_y() + bar.get_height() / 2, label, ha="right", va="center")

    ax.set_title("Positions of pylint annotation instances in matches", pad=40)
    ax.set_yticks([])

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), ncols=6)

    fig.tight_layout()
    plt.savefig(f'{ROOT_DIR}/output/plots/messages/messages_plot.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    interesting_annotations = ['R1714-consider-using-in', 'R1728-consider-using-generator', 'R1720-no-else-raise', 'R1705-no-else-return',
                               'R1710-inconsistent-return-statements', 'R1732-consider-using-with', 'C0200-consider-using-enumerate',
                               'R0912-too-many-branches', 'W0612-unused-variable', 'C0206-consider-using-dict-items',
                               'R0911-too-many-return-statements']

    message_analyzer = PylintAnalyzer()
    message_analyzer.set_files(glob(f'{ROOT_DIR}/data/exercises/*/*.py'))

    plot_accuracies_for_annotations(message_analyzer, interesting_annotations, load_stats=False, save_stats=False)
