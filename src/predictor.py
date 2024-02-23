import math
import csv
from glob import glob
from typing import List, Tuple
from collections import defaultdict, Counter

from src.analyze import FeedbackAnalyzer
from src.constants import ROOT_DIR
from src.feedback_model import FeedbackModel
from src.custom_types import AnnotatedTree
from src.tree_algorithms.subtree_on_line import find_subtree_on_line


message_occurrence_counter = Counter()


def save_results(results: List[Tuple[str, List[Tuple[int, List[str], str | None]]]]):
    with open(f'{ROOT_DIR}/output/predictor/results.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        writer.writerow(['Filename', 'Line', 'Actual messages', 'Predicted message'])
        for file, line_results in results:
            for line, actual_messages, predicted_message in line_results:
                writer.writerow([file, line, actual_messages, predicted_message])


def most_fitting_predictions_for_file(model: FeedbackModel, annotated_tree: AnnotatedTree, stats: Counter):
    """
    Determine for each feedback message the most fitting line to assign it to in the file
    """
    # TODO average occurrences per file bijhouden, en dan per message de x beste posities aanraden (met x het aantal average occurrences)
    tree, annotations = annotated_tree

    messages_per_line = defaultdict(set)
    for (message, line) in annotations:
        messages_per_line[line].add(message)

    subtrees_per_line = {}

    matching_scores_per_line_per_message = defaultdict(dict)
    highest_score_and_line_per_message = defaultdict(lambda: (-math.inf, -1))
    for line in tree['lines']:
        subtree = find_subtree_on_line(tree, line)
        subtrees_per_line[line] = subtree
        if subtree is not None:
            matching_scores = model.calculate_matching_scores(subtree, identifying_only=True)
            matching_scores_per_line_per_message[line] = matching_scores
            over_threshold_scores = {m: s for m, s in matching_scores.items() if s >= model.score_thresholds[m]}
            for message, score in over_threshold_scores.items():
                if score > highest_score_and_line_per_message[message][0]:
                    highest_score_and_line_per_message[message] = (score, line)

    predicted_messages_and_score_per_line = defaultdict(dict)
    for message, (score, line) in highest_score_and_line_per_message.items():
        predicted_messages_and_score_per_line[line][message] = score

    for line, subtree in subtrees_per_line.items():
        if subtree is not None:
            messages = messages_per_line[line]
            if messages:
                print(f"Actual messages on line {line}: {list(messages)}")

            predicted_messages = set(predicted_messages_and_score_per_line[line].keys())
            if predicted_messages:
                print(f"Predicted messages on line {line}: {predicted_messages}")

            # TODO als meerdere messages predicted/on_line zijn: voor allemaal de stats aanpassen, eerste false negative niet meerekenen
            if predicted_messages and predicted_messages.intersection(messages):
                stats["TP"] += 1
            elif predicted_messages and messages:
                stats["different_message"] += 1
            elif predicted_messages:
                for message in predicted_messages:
                    model.score_thresholds[message] = 1.01 * predicted_messages_and_score_per_line[line][message]
                stats["FP"] += 1
            elif messages:
                for message in messages:
                    matching_scores = matching_scores_per_line_per_message[line]
                    if message in matching_scores:
                        model.update_score_threshold(message, matching_scores[message])
                stats["FN"] += 1
                if any(m in model.patterns for m in messages):
                    stats["filtered_FN"] += 1
            else:
                stats["TN"] += 1


def predictions_for_file(model: FeedbackModel, annotated_tree: AnnotatedTree, stats: Counter) -> List[Tuple[int, List[str], str | None]]:
    """
    Predict in the file for each line the highest matching feedback message, if the matching score is above a threshold
    """
    tree, annotations = annotated_tree

    messages_per_line = defaultdict(list)
    for (message, line) in annotations:
        messages_per_line[line].append(message)

    results = []
    new_score_thresholds = {}

    for line in tree['lines']:
        predicted_message = None
        matching_scores = []
        subtree = find_subtree_on_line(tree, line)
        messages_on_line = messages_per_line[line]

        if subtree is not None:
            matching_scores = model.calculate_matching_scores(subtree, identifying_only=True)
            messages_sorted = sorted(matching_scores.keys(), key=lambda ms: matching_scores[ms], reverse=True)
            predicted_message = next((m for m in messages_sorted if matching_scores[m] >= model.score_thresholds[m]), None)

        if messages_on_line:
            print(f"Actual messages on line {line}: {messages_on_line}")
            for message in messages_on_line:
                message_occurrence_counter[message] += 1
                if message in matching_scores:
                    old_threshold = new_score_thresholds[message] if message in new_score_thresholds else model.score_thresholds[message]
                    new_score_thresholds[message] = old_threshold * 0.8 + matching_scores[message] * 0.2 * 0.75

        if predicted_message is not None:
            print(f"Predicted message on line {line}: {predicted_message}")
            # If the predicted message was "rejected" (so it was not actually placed there), raise the threshold
            if predicted_message not in messages_on_line:
                # TODO refine
                new_score_thresholds[predicted_message] = 1.01 * matching_scores[predicted_message]

        if predicted_message is None and not messages_on_line:
            stats["TN"] += 1
        else:
            if predicted_message is not None:
                if predicted_message in messages_on_line:
                    stats["TP"] += 1
                elif messages_on_line:
                    stats["different_message"] += 1
                else:
                    stats["FP"] += 1

            for m in messages_on_line:
                if m != predicted_message and message_occurrence_counter[m] > 1:
                    stats["FN"] += 1
                    if any(m in model.patterns for m in messages_on_line):
                        stats["filtered_FN"] += 1

        if messages_on_line or predicted_message is not None:
            results.append((line, messages_on_line, predicted_message))

    model.score_thresholds.update(new_score_thresholds)

    return results


def test_predictions(exercise_id: str):
    analyzer = FeedbackAnalyzer()
    analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{exercise_id}/*.py'))

    train, test = analyzer.create_train_test_set()
    model = FeedbackModel()
    model.train(train)

    # FN zijn alle false negatives, dus ook van messages waarvoor er geen patronen zijn.
    # Filtered FN zijn enkel de FN's van messages waarvoor er wel patronen zijn
    stats = Counter({"FP": 0, "FN": 0, "TP": 0, "TN": 0, "filtered_FN": 0})
    for file, tree in list(test.items()):
        print(f"Results for file: {file}")
        predictions_for_file(model, tree, stats)
        # most_fitting_predictions_for_file(model, tree, stats)
        print()

    print(stats)


def test_simulate(exercise_id: str):
    stats = Counter({"FP": 0, "FN": 0, "TP": 0, "TN": 0, "filtered_FN": 0, "different_message": 0})

    analyzer = FeedbackAnalyzer()
    analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{exercise_id}/*.py'))

    model = FeedbackModel()

    results = []

    train = {}
    test = analyzer.analyze_files()
    files = analyzer.get_sorted_files()
    for i, file in enumerate(files):
        print(f"Results for file: {file}")
        annotated_code = test.pop(file)

        result = predictions_for_file(model, annotated_code, stats)
        results.append((file, result))
        # most_fitting_predictions_for_file(model, annotated_code, stats)

        train[file] = annotated_code
        if i % 5 == 0:
            # Only train again every 5 files
            print(f"Retraining model, train-set size: {len(train)}")
            model.train(train, thresholds=True)

    save_results(results)

    print(stats)
    print(f'False positive rate: {stats["FP"] / (stats["FP"] + stats["TN"])}')
    print(f'True positive rate: {stats["TP"] / (stats["TP"] + stats["FN"])}')
    print(f'Filtered true positive rate: {stats["TP"] / (stats["TP"] + stats["filtered_FN"])}')


if __name__ == '__main__':
    # test_predictions("505886137")
    test_simulate("505886137")
