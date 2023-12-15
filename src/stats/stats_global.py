from glob import glob
from typing import List

from matplotlib import pyplot as plt

from src.analyze import PylintAnalyzer, FeedbackAnalyzer, Analyzer
from src.feedback_model import FeedbackModel
from src.tester import test_all_files
from src.constants import COLORS, ROOT_DIR, EXERCISE_NAMES_MAP


def plot_global_accuracies(analyzer: Analyzer, file_name="plot", eid=None):
    train, test = analyzer.create_train_test_set()
    model = FeedbackModel()
    model.train(train)

    total_per_message, first_n_per_message, total_first_n_per_message = test_all_files(test, model, n=5)

    total = sum(total_per_message.values())
    percentages = [sum(value.values()) / total for value in first_n_per_message]
    percentages.append(1 - sum(percentages))

    labels_list = ['first', 'second', 'third', 'fourth', 'fifth', 'other']

    patches, _ = plt.pie(percentages, colors=COLORS, pctdistance=1.17, startangle=180, counterclock=False,
                         wedgeprops={'linewidth': 1, 'edgecolor': 'k'} if 1.0 not in percentages else None)
    labels = [f'{label} - {percentage * 100:.2f}%' for label, percentage in zip(labels_list, percentages)]
    plt.legend(labels, loc="upper right", bbox_to_anchor=(1.4, 1))
    if eid is None:
        plt.suptitle('Positions of messages in matches')
    else:
        plt.suptitle(f'{EXERCISE_NAMES_MAP[eid]}')
    plt.title(f'{len(total_per_message.keys())} unique messages', fontsize='medium')
    plt.savefig(f'{ROOT_DIR}/output/plots/{file_name}.png', bbox_inches='tight', dpi=300)


def main_feedback(exercise_ids: List[str] = None):
    message_analyzer = FeedbackAnalyzer()
    message_analyzer.filter_test = False
    if exercise_ids is not None:
        for exercise_id in exercise_ids:
            print(f"Results for excercise with ID {exercise_id}")
            message_analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{exercise_id}/*.py'))
            plot_global_accuracies(message_analyzer, file_name=f"_{exercise_id}", eid=exercise_id)
            plt.clf()

    else:
        plot_global_accuracies(message_analyzer, file_name="plot_all_exercises")


if __name__ == '__main__':
    # message_analyzer = PylintAnalyzer()
    # plot_global_accuracies(message_analyzer)

    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']
    main_feedback(ids)
