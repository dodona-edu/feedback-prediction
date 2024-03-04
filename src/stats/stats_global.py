import pickle
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from analyze import PylintAnalyzer, FeedbackAnalyzer, Analyzer
from feedback_model import FeedbackModel
from tester import test_all_files
from constants import COLORS, ROOT_DIR, ENGLISH_EXERCISE_NAMES_MAP

labels_list = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Other']

colors = [
    COLORS["DARK GREEN"],
    COLORS["LIGHT GREEN"],
    COLORS["LIGHT YELLOW"],
    COLORS["LIGHT ORANGE"],
    COLORS["DARK ORANGE"],
    COLORS["DARK RED"]
]


def gather_data(analyzer: Analyzer):
    train, test = analyzer.create_train_test_set()
    model = FeedbackModel()
    model.train(train)

    total_per_message, first_n_per_message, total_first_n_per_message = test_all_files(test, model, n=5, n_procs=8)

    total = sum(total_per_message.values())
    percentages = [sum(value.values()) / total for value in first_n_per_message]
    percentages.append(1 - sum(percentages))

    return percentages, len(total_per_message.keys())


def plot_global_accuracies_pie(analyzer: Analyzer, file_name="global_pie", eid=None):
    percentages, unique_test_message_count = gather_data(analyzer)

    patches, _ = plt.pie(percentages, colors=colors, pctdistance=1.17, startangle=180, counterclock=False,
                         wedgeprops={'linewidth': 1, 'edgecolor': 'k'} if 1.0 not in percentages else None)
    labels = [f'{label} - {percentage * 100:.2f}%' for label, percentage in zip(labels_list, percentages)]
    plt.legend(labels, loc="upper right", bbox_to_anchor=(1.4, 1))
    if eid is None:
        plt.suptitle('Positions of messages in matches')
    else:
        plt.suptitle(f'{ENGLISH_EXERCISE_NAMES_MAP[eid]}')
    plt.title(f'{unique_test_message_count} unique messages', fontsize='medium')
    plt.savefig(f'{ROOT_DIR}/output/plots/global/{file_name}.png', bbox_inches='tight', dpi=300)


def plot_global_accuracies_stacked_bar(results: Dict[str, Tuple[List[float], int]], file_name="plot"):
    percentages_per_category = list([list(reversed([result[0][i] for result in results.values()])) for i in range(6)])

    eids = list(reversed(results.keys()))

    fig, ax = plt.subplots(figsize=(10, 6))
    left = np.zeros(len(results))

    width = 0.5

    for color, percentages, label in zip(colors, percentages_per_category, labels_list):
        ax.barh(eids, percentages, width, label=label, left=left, color=color)
        left += percentages

    for bars, percentages in zip(ax.containers, percentages_per_category):
        labels = [f'{p * 100:.1f}%' if p > 0.05 else "" for p in percentages]
        ax.bar_label(bars, labels, label_type='center', color='white')

    for bar, eid in zip(ax.containers[0], eids):
        exercise_name = ENGLISH_EXERCISE_NAMES_MAP[eid] if eid in ENGLISH_EXERCISE_NAMES_MAP else eid
        plt.text(-0.02, bar.get_y() + bar.get_height() / 2, exercise_name, ha="right", va="center")
        plt.text(1.02, bar.get_y() + bar.get_height() / 2, f"{results[eid][1]}", ha="left", va="center")

    ax.set_xlim((0, 1))
    ax.set_yticks([])

    plt.text(1.06, 1, "Amount of unique test messages", rotation=-90)

    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.1), ncols=6)

    fig.tight_layout()

    plt.savefig(f'{ROOT_DIR}/output/plots/global/{file_name}.png', bbox_inches='tight', dpi=300)


def main_pie(analyzer: Analyzer, exercise_ids: List[str] = None):
    if exercise_ids is not None:
        for exercise_id in exercise_ids:
            print(f"Results for excercise with ID {exercise_id}")
            analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{exercise_id}/*.py'))
            plot_global_accuracies_pie(analyzer, file_name=f"_{exercise_id}", eid=exercise_id)
            plt.clf()

    else:
        plot_global_accuracies_pie(analyzer, file_name="plot_all_exercises")


def main_stacked_bars(exercise_ids: List[str], is_pylint: bool = False, load_data: bool = False, save_data: bool = False):
    file_name = f"stacked_bars_global{'_pylint' if is_pylint else ''}"
    analyzer = PylintAnalyzer() if is_pylint else FeedbackAnalyzer()
    results_per_exercise_id = {}

    if load_data:
        with open(f'{ROOT_DIR}/output/stats/{file_name}', 'rb') as save_file:
            results_per_exercise_id = pickle.load(save_file)
    else:
        for eid in exercise_ids:
            print(f"Results for excercise with ID {eid}")

            analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{eid}/*.py'))
            results_per_exercise_id[eid] = gather_data(analyzer)

        if is_pylint:
            analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/*/*.py'))
            results_per_exercise_id["Combined"] = gather_data(analyzer)

        if save_data:
            with open(f'{ROOT_DIR}/output/stats/{file_name}', 'wb') as save_file:
                pickle.dump(results_per_exercise_id, save_file, protocol=pickle.HIGHEST_PROTOCOL)

    plot_global_accuracies_stacked_bar(results_per_exercise_id, file_name=f"_plot")


if __name__ == '__main__':
    # message_analyzer = PylintAnalyzer()
    # for i in range(5):
    #     message_analyzer = PylintAnalyzer()
    #     message_analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/*/*.py'))
    #     plot_global_accuracies(message_analyzer, file_name=f"pylint_{i}_plot")

    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']
    # main_pie(message_analyzer, ids)
    main_stacked_bars(ids)
