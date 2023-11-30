import pickle
from collections import Counter
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from analyze import Analyzer, PylintAnalyzer, FeedbackAnalyzer
from matcher import test_all_files

colors = [(10, 118, 49), (35, 212, 23), (177, 212, 14), (212, 160, 26), (212, 88, 18), (125, 0, 27)]
colors = list(map(lambda x: (x[0] / 256, x[1] / 256, x[2] / 256), colors))


def gather_stats(messages, analyzer: Analyzer, load_stats=False, save_stats=True):
    if load_stats:
        with open(f'output/stats/{STATS_FILE}', 'rb') as stats_file:
            stats = pickle.load(stats_file)
    else:
        training, test, patterns, pattern_scores = analyzer.analyze()

        test_for_message = {}
        for file in test:
            for (m, line) in test[file][1]:
                if m in messages:
                    if file not in test_for_message:
                        test_for_message[file] = (test[file][0], [])
                    test_for_message[file][1].append((m, line))

        stats = list(test_all_files(test_for_message, patterns, pattern_scores, n=5))
        if save_stats:
            with open(f'output/stats/{STATS_FILE}', 'wb') as stats_file:
                pickle.dump(stats, stats_file, pickle.HIGHEST_PROTOCOL)

    return stats


def determine_training_totals(training):
    total = Counter()
    for (_, messages) in training.values():
        for m in messages:
            total[m[0]] += 1

    return total


def plot_accuracies(stats, total_training):
    total, first_n, total_first_n = stats

    messages = [m for m in total]

    def get_counts(pos):
        return [first_n[pos][m] / total[m] for m in messages]

    positions = {
        "first": get_counts(0),
        "second": get_counts(1),
        "third": get_counts(2),
        "fourth": get_counts(3),
        "fifth": get_counts(4),
        "other": [(total[m] - total_first_n[m]) / total[m] for m in messages]
    }

    width = 0.5
    fig, ax = plt.subplots()
    bottom = np.zeros(len(messages))

    bar_titles = [f"{m} ({total_training[m]};{total[m]})" for m in messages]

    for color, (position, counts) in zip(colors, positions.items()):
        ax.bar(bar_titles, counts, width, label=position, bottom=bottom, color=color)
        bottom += counts

    ax.set_title("Positions of messages in matches", pad=20)
    # plt.suptitle(STATS_FILE)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right', clip_on=True)
    handles, labels = ax.get_legend_handles_labels()

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])

    ax.legend(handles[::-1], labels[::-1], loc="center right", bbox_to_anchor=(1.5, 0.7))

    fig.tight_layout()
    plt.savefig('output/plots/saved_plot.png', bbox_inches='tight')
    # fig.show()


def plot_global_accuracies(stats, file_name="plot"):
    total_per_message, first_n_per_message, total_first_n_per_message = stats

    total = sum(total_per_message.values())
    percentages = [sum(value.values()) / total for value in first_n_per_message]
    percentages.append(1 - sum(percentages))

    labels = ['first', 'second', 'third', 'fourth', 'fifth', 'other']

    plt.pie(percentages, autopct='%1.1f%%', colors=colors, pctdistance=1.17, startangle=180, counterclock=False)
    plt.legend(labels, loc="upper right", bbox_to_anchor=(1.2, 1))
    plt.title(f'Positions of messages in matches (#tm={len(total_per_message.keys())})', pad=10)  # #tm = amount of different test messages
    plt.savefig(f'output/plots/{file_name}.png', bbox_inches='tight')



def main_pylint(interesting_messages):
    message_analyzer = PylintAnalyzer(load_analysis=True, load_patterns=False, save_analysis=False, save_patterns=False)
    results = gather_stats(message_analyzer, messages=interesting_messages, load_stats=False, save_stats=False)
    if len(interesting_messages) > 0:
        training_totals = determine_training_totals(message_analyzer.perform_analysis()[0])
        plot_accuracies(results, training_totals)
    else:
        plot_global_accuracies(results)


def main_feedback(exercise_ids=None):
    message_analyzer = FeedbackAnalyzer(load_analysis=False, save_analysis=False, load_patterns=False, save_patterns=False)
    if exercise_ids is not None:
        for exercise_id in exercise_ids:
            print(f"Results for excercise with ID {exercise_id}")
            message_analyzer.load_files(glob(f'data/excercises/{exercise_id}/*.py'))
            stats = gather_stats(message_analyzer, save_stats=False)
            plot_global_accuracies(stats, file_name=exercise_id)
            plt.clf()

    else:
        stats = gather_stats(message_analyzer, load_stats=False, save_stats=False)
        plot_global_accuracies(stats, file_name="plot2")


if __name__ == '__main__':
    interesting_messages = ['R1714-consider-using-in', 'R1728-consider-using-generator', 'R1720-no-else-raise', 'R1705-no-else-return']
    # interesting_messages = ['R1710-inconsistent-return-statements', 'R1732-consider-using-with', 'C0200-consider-using-enumerate',
    #                         'R0912-too-many-branches']

    message_analyzer = PylintAnalyzer(load_analysis=True, load_patterns=False, save_patterns=False)
    STATS_FILE = f"stats_{message_analyzer.PATTERNS_FILE}"
    results = gather_stats(interesting_messages, message_analyzer, load_stats=False, save_stats=False)
    plot_accuracies(results)
