import csv
import json
import math
import time
from collections import Counter, defaultdict
from glob import glob
from typing import List, Tuple, Dict, Set

from pqdm.processes import pqdm
import multiprocessing.pool

from analyze import FeedbackAnalyzer
from constants import ROOT_DIR
from custom_types import LineTree, Annotation, AnnotatedTree, Tree
from feedback_model import FeedbackModel
from tree_algorithms.subtree_on_line import find_subtree_on_line


class Predictor:

    def __init__(self):
        self.retrain_every_n = 5
        self.train = {}
        self.model = FeedbackModel()
        self.average_occurrences_per_file = {}

    def _calculate_matching_scores(self, line, tree):
        subtree = find_subtree_on_line(tree, line)
        if subtree is not None:
            return line, self.model.calculate_matching_scores(subtree, identifying_only=True)

        return line, None

    def predict(self, tree: LineTree) -> Tuple[Dict[int, Set[str]], Dict[int, Dict[str, float]], List[int]]:
        # Keep track of which lines had a subtree on them, only used for calculating statistics
        lines_with_subtree = []

        matching_scores_per_line_per_message = defaultdict(dict)
        scores_and_lines_per_message = defaultdict(list)

        pool = multiprocessing.pool.Pool(8)
        results = pool.starmap(self._calculate_matching_scores, [(line, tree) for line in tree['lines']], chunksize=math.ceil(len(tree['lines']) / 8))
        pool.close()
        # results = pqdm([(line, tree) for line in tree['lines']], self._calculate_over_threshold_scores, n_jobs=8, argument_type='args', exception_behaviour='immediate')

        for line, matching_scores in results:
            if matching_scores is not None:
                lines_with_subtree.append(line)
                matching_scores_per_line_per_message[line] = matching_scores
                over_threshold_scores = {m: s for m, s in matching_scores.items() if s >= self.model.score_thresholds[m]}
                for message, score in over_threshold_scores.items():
                    scores_and_lines_per_message[message].append((score, line))

        predicted_messages_per_line = defaultdict(set)
        for message, possible_annotations in scores_and_lines_per_message.items():
            possible_annotations.sort(key=lambda an: an[0], reverse=True)
            average_occurrences = max(round(self.average_occurrences_per_file.get(message, (0.0, 0))[0]), 1)
            for (score, line) in possible_annotations[:average_occurrences]:
                predicted_messages_per_line[line].add(message)

        return predicted_messages_per_line, matching_scores_per_line_per_message, lines_with_subtree

    def reject_suggestion(self, suggestion: Annotation, subtree: Tree):
        # TODO negative patterns
        pass

    def process_actual_feedback(self, annotations: List[Annotation], predicted_messages_per_line: Dict[int, Set[str]],
                                matching_scores_per_line_per_message: Dict[int, Dict[str, float]]) -> Dict[int, Set[str]]:
        message_occurrences_in_this_file = Counter()
        messages_per_line = defaultdict(set)
        for (message, line) in annotations:
            message_occurrences_in_this_file[message] += 1
            messages_per_line[line].add(message)

            # If the message was not predicted because it's threshold was too high, update the threshold
            if message not in predicted_messages_per_line[line]:
                matching_scores = matching_scores_per_line_per_message[line]
                if message in matching_scores:
                    # print(f"Matching score {message}: {matching_scores[message]}")
                    self.model.update_score_threshold(message, matching_scores[message])

        for message, count in message_occurrences_in_this_file.items():
            old_average, old_file_count = self.average_occurrences_per_file.get(message, (0.0, 0))
            new_file_count = old_file_count + 1
            new_average = old_average + (count - old_average) / new_file_count
            self.average_occurrences_per_file[message] = (new_average, new_file_count)

        return messages_per_line

    def update_train(self, file: str, annotated_tree: AnnotatedTree):
        self.train[file] = annotated_tree

        if len(self.train) % self.retrain_every_n == 0:
            print(f"Retraining model, train-set size: {len(self.train)}")
            self.model.train(self.train, n_procs=8, thresholds=True)


def calculate_statistics(result: List[Tuple[int, List[str], List[str]]], lines_with_subtree: List[int],
                         stats: Counter, predictor: Predictor, seen_messages: Set[str]):
    # TODO betere manier van metrics, zie classification report of confusion matrix https://medium.com/apprentice-journal/evaluating-multi-class-classifiers-12b2946e755b
    # TODO multi-label classification evaluation: https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea
    messages_in_file = Counter()
    predictions_in_file = Counter()

    non_empty_lines = [line for line, _, _ in result]
    stats["TN"] += len(list(filter(lambda line: line not in non_empty_lines, lines_with_subtree)))

    for line, messages, predicted_messages in result:
        if messages:
            messages_in_file.update(messages)

        if predicted_messages:
            predictions_in_file.update(predicted_messages)

        correct_messages = set(predicted_messages).intersection(messages)
        wrong_messages = set(predicted_messages).difference(messages)
        unpredicted_messages = set(messages).difference(predicted_messages)

        stats["TP"] += len(correct_messages)

        if not messages:
            stats["FP"] += len(wrong_messages)
        elif unpredicted_messages:
            for m in unpredicted_messages:
                if m in seen_messages:
                    stats["FN"] += 1
                    if m in predictor.model.patterns:
                        stats["filtered_FN"] += 1

    seen_messages.update(messages_in_file.keys())


def save_results(results: List[Tuple[str, List[Tuple[int, List[str], List[str]]]]], e_id: str):
    with open(f'{ROOT_DIR}/output/predictor/fitting_results_{e_id}_2.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        for file, line_results in results:
            file_parts = file.split('/')
            for line, actual_messages, predicted_messages in line_results:
                writer.writerow([file_parts[-2], file_parts[-1], line, json.dumps(actual_messages), json.dumps(predicted_messages)])


def test_simulate(exercise_id: str):
    stats = Counter({"FP": 0, "FN": 0, "TP": 0, "TN": 0, "filtered_FN": 0, "different_message": 0})

    analyzer = FeedbackAnalyzer()
    analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/{exercise_id}/*.py'))

    results = []

    seen_messages = set()
    predictor = Predictor()

    test = analyzer.analyze_files()
    files = analyzer.get_sorted_files()
    for i, file in enumerate(files):
        print(f"Results for file: {file}")
        annotated_tree = test.pop(file)

        predictions_per_line, scores, lines_with_subtree = predictor.predict(annotated_tree[0])
        messages_per_line = predictor.process_actual_feedback(annotated_tree[1], predictions_per_line, scores)

        result = []
        for line in lines_with_subtree:
            messages = list(messages_per_line[line])
            if messages:
                print(f"Actual messages on line {line}: {messages}")
            predictions = predictions_per_line[line]
            if predictions:
                print(f"Predicted messages on line {line}: {predictions}")
            if messages or predictions:
                result.append((line, messages, list(predictions)))

        calculate_statistics(result, lines_with_subtree, stats, predictor, seen_messages)
        results.append((file, result))

        predictor.update_train(file, annotated_tree)

        print()
        # print(f"Current score thresholds: {predictor.model.score_thresholds}")
        # print()

    save_results(results, exercise_id)

    print(stats)
    print(f'False positive rate: {stats["FP"] / (stats["FP"] + stats["TN"])}')
    print(f'True positive rate: {stats["TP"] / (stats["TP"] + stats["FN"])}')
    print(f'Filtered true positive rate: {stats["TP"] / (stats["TP"] + stats["filtered_FN"])}')


if __name__ == '__main__':
    start = time.time()
    test_simulate("505886137")
    print()
    print(f"Total time: {time.time() - start}")

