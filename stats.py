import pickle
import matplotlib.pyplot as plt
import numpy as np

from analyze import Analyzer, PylintAnalyzer, FeedbackAnalyzer
from matcher import test_all_files

STATS_FILE = ""


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
    plt.show()
    plt.savefig('output/plots/saved_plot.png', bbox_inches='tight')
    # fig.show()


if __name__ == '__main__':
    interesting_messages = ['R1714-consider-using-in', 'R1728-consider-using-generator', 'R1720-no-else-raise', 'R1705-no-else-return']
    # interesting_messages = ['R1710-inconsistent-return-statements', 'R1732-consider-using-with', 'C0200-consider-using-enumerate',
    #                         'R0912-too-many-branches']

    message_analyzer = PylintAnalyzer(load_analysis=True, load_patterns=False, save_patterns=False)
    STATS_FILE = f"stats_{message_analyzer.PATTERNS_FILE}"
    results = gather_stats(interesting_messages, message_analyzer, load_stats=False, save_stats=False)
    plot_accuracies(results)
