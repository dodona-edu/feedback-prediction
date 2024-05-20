import csv
import json
from collections import Counter
from typing import List, Set, Tuple
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

from constants import ROOT_DIR, DUTCH_EXERCISE_NAMES_MAP


def manual_statistics(annotations: List[int], predictions: List[int], seen_annotations: Set[int], annotations_in_file: Counter, stats: Counter):
    if not annotations and not predictions:
        stats["TN"] += 1
    else:
        if annotations:
            annotations_in_file.update(annotations)

        correct_annotations = set(predictions).intersection(annotations)
        wrong_annotations = set(predictions).difference(annotations)
        unpredicted_annotations = set(annotations).difference(predictions)

        stats["TP"] += len(correct_annotations)

        if wrong_annotations:
            stats["FP"] += len(wrong_annotations)
        if unpredicted_annotations:
            for a_id in unpredicted_annotations:
                if a_id in seen_annotations:
                    stats["FN"] += 1


def calculate_classification_report(results: List[Tuple[List[int], List[int]]]) -> dict:
    binarizer = MultiLabelBinarizer().fit([r[0] for r in results])
    annotations = binarizer.transform([r[0] for r in results])
    predictions = binarizer.transform([r[1] for r in results])

    print(classification_report(annotations, predictions, target_names=list(map(str, binarizer.classes_))))
    return classification_report(annotations, predictions, target_names=list(map(str, binarizer.classes_)), output_dict=True)


def predictions_within_k_lines(results_per_submission: List[List[Tuple[int, List[int], List[int]]]], k):
    result = []
    for submission in results_per_submission:
        max_line = submission[-1][0]
        res = [([], []) for _ in range(max_line + 1)]
        for line, annotations, predictions in submission:
            res[line][0].extend(annotations)
            if predictions:
                start = max(0, line - k)
                stop = min(max_line + 1, line + k)
                for i in range(start, stop):
                    res[i][1].extend(predictions)

        result.extend(res)

    return result


def annotations_within_k_lines(results_per_submission: List[List[Tuple[int, List[int], List[int]]]], k):
    result = []
    for submission in results_per_submission:
        max_line = submission[-1][0]
        res = [([], []) for _ in range(max_line + 1)]
        for line, annotations, predictions in submission:
            res[line][1].extend(predictions)
            if annotations:
                start = max(0, line - k)
                stop = min(max_line + 1, line + k)
                for i in range(start, stop):
                    res[i][0].extend(annotations)

        result.extend(res)

    return result


def statistics_for_exercise(e_id: str):
    with open(f'{ROOT_DIR}/output/predictor/test/{stats_file_prefix}_{e_id}.csv') as csv_file:
        rows = list(csv.reader(csv_file, delimiter='|'))

        seen_annotations = set()
        annotations_in_file = Counter()

        current_file = rows[0][1]

        stats = Counter()

        submission_result = []
        results_per_submission = []
        results = []
        for _, file, line, annotations, predicted in rows:
            if file != current_file:
                seen_annotations.update(annotations_in_file.keys())
                annotations_in_file.clear()
                current_file = file
                results_per_submission.append(submission_result)
                submission_result = []

            annotation_ids = [x[0] for x in json.loads(annotations)]
            prediction_ids = [x[0] for x in json.loads(predicted)]

            # The results should only take into account annotations that have already been seen at least once
            results.append(([a_id for a_id in annotation_ids if a_id in seen_annotations], prediction_ids))
            manual_statistics(annotation_ids, prediction_ids, seen_annotations, annotations_in_file, stats)

            submission_result.append((int(line), [a_id for a_id in annotation_ids if a_id in seen_annotations], prediction_ids))

        if submission_result:
            results_per_submission.append(submission_result)

        print(f"Exercise: {NAMES_MAP[e_id]}")
        print("Manual statistics:")
        print(stats)
        print()

        print("Classification report:")
        regular_report = calculate_classification_report(results)

        print(f"Classification report with k={k} line margin for predictions (only look at recall):")
        within_k_line_results_precision = annotations_within_k_lines(results_per_submission, k)
        within_k_line_results_recall = predictions_within_k_lines(results_per_submission, k)

        within_k_precision = calculate_classification_report(within_k_line_results_precision)["weighted avg"]["precision"]
        within_k_recall = calculate_classification_report(within_k_line_results_recall)["weighted avg"]["recall"]

        return regular_report, (within_k_precision, within_k_recall)


def plot_precision_recall(eids):
    results = []
    for eid in eids:
        report, (within_k_precision, within_k_recall) = statistics_for_exercise(eid)
        precision, recall = report["weighted avg"]["precision"], report["weighted avg"]["recall"]
        results.append((eid, precision, recall, within_k_precision, within_k_recall))

    fig, ax = plt.subplots(figsize=(10.5, 6))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    leg_handles = []
    leg_labels = []
    for (eid, precision, recall, within_k_precision, within_k_recall), color in zip(results, colors):
        leg_handle1 = ax.scatter(recall, precision, label=NAMES_MAP[eid], color=color)
        leg_handle2 = ax.scatter(within_k_recall, within_k_precision, label=NAMES_MAP[eid], color=color, edgecolors='black', linewidth=1.5)
        leg_handles.append((leg_handle2, leg_handle1))
        leg_labels.append(NAMES_MAP[eid])

    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(which='both')

    ax.legend(handles=leg_handles, labels=leg_labels, handler_map={tuple: HandlerTuple(ndivide=1)})

    fig.tight_layout()

    plt.savefig(f'{ROOT_DIR}/output/plots/predictor/pr_{stats_file_prefix}.png', bbox_inches='tight', dpi=150)


if __name__ == '__main__':
    NAMES_MAP = DUTCH_EXERCISE_NAMES_MAP
    stats_file_prefix = 'negatives'
    k = 5

    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']

    plot_precision_recall(ids)
    # statistics_for_exercise("505886137")
    # statistics_for_exercise("933265977")
    # statistics_for_exercise("1730686412")
    # statistics_for_exercise("1875043169")
    # statistics_for_exercise("2046492002")
    # statistics_for_exercise("2146239081")


# https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
# The precision is intuitively the ability of the classifier not to label a negative sample as positive.

# The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
# The recall is intuitively the ability of the classifier to find all the positive samples.

# The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
# The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
