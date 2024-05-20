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
from custom_types import LineTree, AnnotationInstance, AnnotatedTree, Tree
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
            return line, self.model.calculate_matching_scores(subtree, use_negatives=True)

        return line, None

    def predict(self, tree: LineTree) -> Tuple[Dict[int, Set[int]], Dict[int, Dict[int, float]], List[int]]:
        # Keep track of which lines had a subtree on them, only used for calculating statistics
        lines_with_subtree = []

        matching_scores_per_line_per_annotation = defaultdict(dict)
        scores_and_lines_per_annotation = defaultdict(list)

        pool = multiprocessing.pool.Pool(8)
        results = pool.starmap(self._calculate_matching_scores, [(line, tree) for line in tree['lines']], chunksize=math.ceil(len(tree['lines']) / 8))
        pool.close()
        # results = pqdm([(line, tree) for line in tree['lines']], self._calculate_matching_scores, n_jobs=1, argument_type='args', exception_behaviour='immediate')

        for line, matching_scores in results:
            if matching_scores is not None:
                lines_with_subtree.append(line)
                matching_scores_per_line_per_annotation[line] = matching_scores
                over_threshold_scores = {a_id: s for a_id, s in matching_scores.items() if s >= self.model.score_thresholds[a_id]}
                for a_id, score in over_threshold_scores.items():
                    scores_and_lines_per_annotation[a_id].append((score, line))

        predicted_annotations_per_line = defaultdict(set)
        for a_id, possible_annotations in scores_and_lines_per_annotation.items():
            possible_annotations.sort(key=lambda an: an[0], reverse=True)
            average_occurrences = max(round(self.average_occurrences_per_file.get(a_id, (0.0, 0))[0]), 1)
            for (score, line) in possible_annotations[:average_occurrences]:
                predicted_annotations_per_line[line].add(a_id)

        return predicted_annotations_per_line, matching_scores_per_line_per_annotation, lines_with_subtree

    def reject_suggestion(self, suggestion: AnnotationInstance, tree: LineTree):
        print(f"Rejection of suggestion: {suggestion}")
        annotation_id, line = suggestion
        subtree = find_subtree_on_line(tree, line)
        self.model.update_negatives(annotation_id, subtree)

    def process_actual_feedback(self, annotation_instances: List[AnnotationInstance], predicted_annotations_per_line: Dict[int, Set[int]],
                                matching_scores_per_line_per_annotation: Dict[int, Dict[int, float]]) -> Dict[int, Set[int]]:
        annotation_occurrences_in_this_file = Counter()
        annotations_per_line = defaultdict(set)
        for (a_id, line) in annotation_instances:
            annotation_occurrences_in_this_file[a_id] += 1
            annotations_per_line[line].add(a_id)

            # If the annotation was not predicted because it's threshold was too high, update the threshold
            if a_id not in predicted_annotations_per_line[line]:
                matching_scores = matching_scores_per_line_per_annotation[line]
                if a_id in matching_scores:
                    # print(f"Matching score {message}: {matching_scores[message]}")
                    self.model.update_score_threshold(a_id, matching_scores[a_id])

        for a_id, count in annotation_occurrences_in_this_file.items():
            old_average, old_file_count = self.average_occurrences_per_file.get(a_id, (0.0, 0))
            new_file_count = old_file_count + 1
            new_average = old_average + (count - old_average) / new_file_count
            self.average_occurrences_per_file[a_id] = (new_average, new_file_count)

        return annotations_per_line

    def update_train(self, file: str, annotated_tree: AnnotatedTree):
        self.train[file] = annotated_tree

        if len(self.train) % self.retrain_every_n == 0:
            print(f"Retraining model, train-set size: {len(self.train)}")
            self.model.train(self.train, n_procs=8, thresholds=True)


def calculate_statistics(result: List[Tuple[int, List[int], List[int]]], lines_with_subtree: List[int],
                         stats: Counter, predictor: Predictor, seen_annotations: Set[int]):
    annotations_in_file = Counter()

    non_empty_lines = [line for line, _, _ in result]
    stats["TN"] += len(list(filter(lambda line: line not in non_empty_lines, lines_with_subtree)))

    for line, annotations, predicted_annotations in result:
        if annotations:
            annotations_in_file.update(annotations)

        correct_annotations = set(predicted_annotations).intersection(annotations)
        wrong_annotations = set(predicted_annotations).difference(annotations)
        unpredicted_annotations = set(annotations).difference(predicted_annotations)

        stats["TP"] += len(correct_annotations)

        if wrong_annotations:
            stats["FP"] += len(wrong_annotations)
        elif unpredicted_annotations:
            for a_id in unpredicted_annotations:
                if a_id in seen_annotations:
                    stats["FN"] += 1
                    if a_id in predictor.model.patterns:
                        stats["filtered_FN"] += 1

    seen_annotations.update(annotations_in_file.keys())


def save_results(results: List[Tuple[str, List[Tuple[int, List[Tuple[int, str]], List[Tuple[int, str]]]]]], e_id: str):
    with open(f'{ROOT_DIR}/output/predictor/{e_id}.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='|')
        for file, line_results in results:
            file_parts = file.split('/')
            for line, actual_annotations, predicted_annotations in line_results:
                writer.writerow([file_parts[-2], file_parts[-1], line, json.dumps(actual_annotations), json.dumps(predicted_annotations)])


def test_simulate(exercise_id: str):
    stats = Counter({"FP": 0, "FN": 0, "TP": 0, "TN": 0, "filtered_FN": 0, "different_message": 0})

    analyzer = FeedbackAnalyzer()
    analyzer.set_files(glob(f'{ROOT_DIR}/data/exercises/{exercise_id}/*.py'))

    results = []

    seen_annotations = set()
    predictor = Predictor()

    test = analyzer.analyze_files()
    files = analyzer.get_sorted_files()
    for i, file in enumerate(files):
        print(f"Results for file: {file}")
        submission_id = file.split('/')[-1].split('.')[0]
        annotated_tree = test.pop(file)

        predictions_per_line, scores, lines_with_subtree = predictor.predict(annotated_tree[0])
        annotations_per_line = predictor.process_actual_feedback(annotated_tree[1], predictions_per_line, scores)

        result = []
        result_with_string = []
        for line in lines_with_subtree:
            annotations_with_string, predictions_with_string = [], []

            annotations = list(annotations_per_line[line])
            if annotations:
                annotations_with_string = [(a_id, analyzer.annotations_per_submission_per_line[submission_id][line][a_id]) for a_id in annotations]
                print(f"Actual messages on line {line}: {annotations}")

            predictions = predictions_per_line[line]
            if predictions:
                predictions_with_string = [(a_id, analyzer.id_annotation_map[a_id]) for a_id in predictions]
                print(f"Predicted messages on line {line}: {predictions}")

            if annotations or predictions:
                result.append((line, annotations, list(predictions)))
            result_with_string.append((line, annotations_with_string, predictions_with_string))

            # Reject wrong predictions:
            wrong_predictions = predictions.difference(annotations)
            for a_id in wrong_predictions:
                predictor.reject_suggestion((a_id, line), annotated_tree[0])

        results.append((file, result_with_string))

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

