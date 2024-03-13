import datetime
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple

from pqdm.processes import pqdm

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

    def _annotation_subtrees(self, dataset: Dict[str, AnnotatedTree]) -> Dict[int, List[HorizontalTree]]:
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
        Also determines the identifying nodes of the subtrees.
        """
        annotation_patterns = set()
        identifying_nodes = set()

        if len(subtrees) >= 3:
            annotation_patterns = mine_patterns(subtrees)

            for subtree in subtrees:
                identifying_nodes.update(subtree)
            identifying_nodes.remove(-1)

        return annotation_id, (annotation_patterns, identifying_nodes)

    def train(self, training: Dict[str, AnnotatedTree], n_procs=8, thresholds=False) -> None:
        """
        Train the feedback model.
        """
        start = datetime.datetime.now()

        subtrees = self._annotation_subtrees(training)
        print("Determining patterns for training data")
        patterns = {}

        results: List[Tuple[int, PatternCollection]] = pqdm(list(subtrees.items()), self._find_patterns, n_jobs=n_procs, argument_type='args')

        node_counts = defaultdict(int)
        for _, (_, nodes) in results:
            for node in nodes:
                node_counts[node] += 1
        nodes_to_remove = {n for n, c in node_counts.items() if c > 3}

        for m, (pattern_set, node_set) in results:
            node_set.difference_update(nodes_to_remove)
            if pattern_set or node_set:
                patterns[a_id] = (pattern_set, node_set)

        print("Calculating pattern weights")
        pattern_weights = defaultdict(float)
        for annotation_patterns, _ in patterns.values():
            for pattern in annotation_patterns:
                pattern_weights[pattern] += 1

        for pattern in pattern_weights.keys():
            # pattern_scores[pattern] = len(pattern) / math.log10(len(pattern_weights) / pattern_scores[pattern])
            pattern_weights[pattern] = len(pattern) / pattern_weights[pattern]

        self.patterns = patterns
        self.pattern_weights = pattern_weights

        if thresholds:
            print("Calculating score thresholds")
            self.calculate_score_thresholds(subtrees)

        print(f"Total training time: {datetime.datetime.now() - start}")

    def calculate_score_thresholds(self, training_subtrees_per_message: Dict[str, List[HorizontalTree]]) -> None:
        for m, subtrees in training_subtrees_per_message.items():
            if m in self.patterns:
                score_threshold = sum(pqdm([(m, subtree) for subtree in subtrees], self.calculate_matching_score, n_jobs=8, argument_type='args'))
                self.score_thresholds[m] = 0.75 * score_threshold / len(subtrees)

    def update_score_threshold(self, message: str, score: float):
        self.score_thresholds[message] = self.score_thresholds[message] * 0.8 + score * 0.2 * 0.75

    def calculate_matching_score(self, annotation_id: int, subtree: HorizontalTree) -> float:
        pattern_set = self.patterns[annotation_id][0]
        matches = list(filter(lambda pattern: subtree_matches(subtree, pattern), pattern_set))
        matches_score = 0
        if pattern_set:
            matches_score = sum(self.pattern_weights[match] for match in matches) / len(pattern_set)

        node_set = self.patterns[annotation_id][1]
        nodes = set(subtree).intersection(node_set)
        nodes_score = 0
        if node_set:
            nodes_score = len(nodes) / len(node_set)

        return matches_score + nodes_score

    def calculate_matching_scores(self, subtree: Tree, identifying_only=False) -> Dict[int, float]:
        horizontal_subtree = list(to_string_encoding(subtree))
        if identifying_only:
            nodes = set(horizontal_subtree)
            matching_scores = {message: self.calculate_matching_score(message, horizontal_subtree) for message, (_, identifying_nodes) in self.patterns.items() if nodes.intersection(identifying_nodes)}
        else:
            matching_scores = {message: self.calculate_matching_score(message, horizontal_subtree) for message in self.patterns.keys()}
        return matching_scores
