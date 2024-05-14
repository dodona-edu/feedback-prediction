import datetime
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple

from pqdm.processes import pqdm

from config import USE_IDENTIFYING_NODES, MIN_PATTERN_SUBTREES, MAX_IDENTIFYING_SUBTREES
from constants import ROOT_DIR
from custom_types import AnnotatedTree, Tree, HorizontalTree, PatternCollection
from util import to_string_encoding
from tree_algorithms.treeminer import mine_patterns
from tree_algorithms.subtree_matches import subtree_matches
from tree_algorithms.subtree_on_line import find_subtree_on_line


class FeedbackModel:
    MODEL_DIR = f'{ROOT_DIR}/output/models'

    def __init__(self):
        self.patterns: Dict[int, PatternCollection] = {}
        self.pattern_weights = {}
        self.score_thresholds = {}

    def load_model(self, model_file: str) -> None:
        print("Loading patterns data")
        with open(f'{self.MODEL_DIR}/{model_file}', 'rb') as patterns_file:
            self.patterns, self.pattern_weights, self.score_thresholds = pickle.load(patterns_file)

    def save_model(self, model_file: str) -> None:
        print("Saving patterns data")
        with open(f'{self.MODEL_DIR}/{model_file}', 'wb') as patterns_file:
            pickle.dump((self.patterns, self.pattern_weights, self.score_thresholds), patterns_file, pickle.HIGHEST_PROTOCOL)

    def get_annotation_subtrees(self, dataset: Dict[str, AnnotatedTree]) -> Dict[int, List[HorizontalTree]]:
        result = defaultdict(list)
        for tree, annotation_instances in dataset.values():
            for a_id, line in annotation_instances:
                subtree = find_subtree_on_line(tree, line)
                if subtree is not None:
                    result[a_id].append(list(to_string_encoding(subtree)))
        return result

    @staticmethod
    def _find_patterns(annotation_id: int, subtrees: List[HorizontalTree]) -> Tuple[int, PatternCollection]:
        """
        Find the patterns present in the given subtrees.
        Also collect all nodes of the subtrees in a set.
        """
        annotation_patterns = set()
        frequent_nodes = set()
        node_set = set()

        if len(subtrees) >= MIN_PATTERN_SUBTREES:
            annotation_patterns, frequent_nodes = mine_patterns(subtrees)

        if USE_IDENTIFYING_NODES:
            for subtree in subtrees:
                node_set.update(subtree)

        return annotation_id, (annotation_patterns, frequent_nodes, node_set)

    def train(self, training: Dict[str, AnnotatedTree], n_procs=8, thresholds=False) -> None:
        """
        Train the feedback model.
        """
        start = datetime.datetime.now()

        annotation_subtrees = self.get_annotation_subtrees(training)

        print("Determining patterns for training data")
        patterns = {}

        results: List[Tuple[int, PatternCollection]] = pqdm(list(annotation_subtrees.items()), self._find_patterns, n_jobs=n_procs, argument_type='args', exception_behaviour='immediate')

        node_counts = defaultdict(int)
        for _, (_, _, nodes) in results:
            for node in nodes:
                node_counts[node] += 1
        nodes_to_remove = {n for n, c in node_counts.items() if c > MAX_IDENTIFYING_SUBTREES}

        for a_id, (pattern_set, frequent_nodes, node_set) in results:
            identifying_nodes = node_set.difference(nodes_to_remove)
            if pattern_set or identifying_nodes:
                patterns[a_id] = (pattern_set, frequent_nodes, identifying_nodes)

        print("Calculating pattern weights")
        pattern_weights = defaultdict(float)
        for annotation_patterns, _, _ in patterns.values():
            for pattern in annotation_patterns:
                pattern_weights[pattern] += 1

        for pattern in pattern_weights.keys():
            # pattern_scores[pattern] = len(pattern) / math.log10(len(pattern_weights) / pattern_scores[pattern])
            pattern_weights[pattern] = len(pattern) / pattern_weights[pattern]

        self.patterns = patterns
        self.pattern_weights = pattern_weights

        if thresholds:
            print("Calculating score thresholds")
            self.calculate_score_thresholds(annotation_subtrees, n_procs)

        print(f"Total training time: {datetime.datetime.now() - start}")

    def _calculate_score_threshold(self, a_id: int, subtrees: List[HorizontalTree]) -> Tuple[int, float]:
        matching_scores = []
        for subtree in subtrees:
            matching_scores.append(self.calculate_matching_score(a_id, subtree))
        score_threshold = 0.95 * (min(matching_scores) + max(matching_scores)) / 2

        return a_id, score_threshold

    def calculate_score_thresholds(self, training_subtrees_per_annotation: Dict[int, List[HorizontalTree]], n_procs: int) -> None:
        annotation_subtrees = []
        for a_id, subtrees in training_subtrees_per_annotation.items():
            if a_id in self.patterns:
                annotation_subtrees.append((a_id, subtrees))

        score_thresholds = pqdm(annotation_subtrees, self._calculate_score_threshold, n_jobs=n_procs, argument_type='args')

        for a_id, score_threshold in score_thresholds:
            if a_id in self.score_thresholds:
                self.score_thresholds[a_id] = 0.2 * score_threshold + 0.8 * self.score_thresholds[a_id]
            else:
                self.score_thresholds[a_id] = score_threshold

    def update_score_threshold(self, annotation_id: int, score: float) -> None:
        if score > 0.2 * self.score_thresholds[annotation_id]:
            self.score_thresholds[annotation_id] = score * 0.95
        else:
            self.score_thresholds[annotation_id] = self.score_thresholds[annotation_id] * 0.8 + score * 0.2 * 0.95

    def calculate_matching_score(self, annotation_id: int, subtree: HorizontalTree, use_negatives=False) -> float:
        pattern_set = self.patterns[annotation_id][0]
        matches = list(filter(lambda pattern: subtree_matches(subtree, pattern), pattern_set))
        matches_score = 0
        if pattern_set:
            matches_score = sum(self.pattern_weights[match] for match in matches) / len(pattern_set)

        node_set = self.patterns[annotation_id][2]
        nodes = set(subtree).intersection(node_set)
        nodes_score = 0
        if node_set:
            nodes_score = len(nodes)

        return matches_score + nodes_score

    def calculate_matching_scores(self, subtree: Tree, use_negatives=False) -> Dict[int, float]:
        horizontal_subtree = list(to_string_encoding(subtree))
        nodes = set(horizontal_subtree)
        nodes.discard(-1)

        matching_scores = {}
        for a_id, (_, frequent_nodes, identifying_nodes) in self.patterns.items() :
            if nodes.intersection(identifying_nodes) or nodes.intersection(frequent_nodes):
                matching_scores[a_id] = self.calculate_matching_score(a_id, horizontal_subtree)
            else:
                matching_scores[a_id] = 0
        return matching_scores


class DummyModel(FeedbackModel):
    """
    This heuristic model gives a score to all annotations based on how frequently they occur.
    It does not use the given context and just returns this static score when asked to calculate the scores.
    """

    def __init__(self):
        super().__init__()
        self.annotation_scores = {}

    def train(self, training: Dict[str, AnnotatedTree], n_procs=8, thresholds=False) -> None:
        self.annotation_scores.clear()
        annotation_subtrees = self.get_annotation_subtrees(training)
        for a_id, subtrees in annotation_subtrees.items():
            self.annotation_scores[a_id] = len(subtrees)
            self.patterns[a_id] = ()

    def calculate_matching_scores(self, subtree: Tree, use_negatives=False) -> Dict[int, float]:
        return self.annotation_scores
