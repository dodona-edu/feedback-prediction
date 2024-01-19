import datetime
import multiprocessing
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Set

from pqdm.processes import pqdm
from tqdm import tqdm

from src.analyze import FeedbackAnalyzer
from src.custom_types import AnnotatedCode, Tree, HorizontalTree
from src.util import to_string_encoding
from src.treeminer import Treeminerd
from src.util import sequence_fully_contains_other_sequence


analyzer = FeedbackAnalyzer()


class FeedbackModel:
    PATTERNS_DIR = ""

    def __init__(self):
        self.patterns: Dict[str, Set[HorizontalTree]] = {}
        self.pattern_weights = {}

    def load_model(self, model_file: str) -> None:
        print("Loading patterns data")
        with open(f'{self.PATTERNS_DIR}/{model_file}', 'rb') as patterns_file:
            self.patterns, self.pattern_weights = pickle.load(patterns_file)

    def save_model(self, model_file: str) -> None:
        print("Saving patterns data")
        with open(f'{self.PATTERNS_DIR}/{model_file}', 'wb') as patterns_file:
            pickle.dump((self.patterns, self.pattern_weights), patterns_file, pickle.HIGHEST_PROTOCOL)

    def tree_on_line(self, code: List[bytes], line: int) -> Tree | None:
        code = b''.join(code[line:line + 1])
        root_on_line = analyzer.parser.parse(code).root_node
        root_as_tree = analyzer.map_tree(root_on_line)
        if len(root_as_tree['children']):
            return root_as_tree['children'][0]
        else:
            return None

    def _message_subtrees(self, dataset: Dict[str, AnnotatedCode]) -> Dict[str, List[Tree]]:
        result = defaultdict(list)
        for key, item in dataset.items():
            for m, line in item[1]:
                subtree = self.tree_on_line(item[0], line)
                if subtree is not None:
                    result[m].append(subtree)
        return result

    @staticmethod
    def _find_patterns(message: str, ts: List[Tree]) -> Tuple[str, Set[HorizontalTree]]:
        message_patterns = []
        if len(ts) >= 3:
            miner = Treeminerd(ts, support=0.8)
            if miner.early_stopping:
                p = multiprocessing.Process(target=miner.get_patterns)
                p.start()
                p.join(30)
                if p.is_alive():
                    p.terminate()
                message_patterns = set(miner.result)
            else:
                message_patterns = miner.get_patterns()


        return message, message_patterns

    def train(self, training: Dict[str, AnnotatedCode], n_procs=8) -> None:
        """
        Determine the patterns present in the trees in the training set.
        """
        start = datetime.datetime.now()

        subtrees = self._message_subtrees(training)
        print("Determining patterns for training data")
        patterns = {}

        results: List[Tuple[str, Set[HorizontalTree]]] = []
        if n_procs > 1:
            results = pqdm(list(subtrees.items()), self._find_patterns, n_jobs=8, argument_type='args')
        else:
            for m, ts in tqdm(subtrees.items()):
                results.append(self._find_patterns(m, ts))

        for m, res in results:
            if len(res) > 0:
                patterns[m] = res

        print("Calculating pattern weights")
        pattern_weights = defaultdict(float)
        for message_patterns in patterns.values():
            for pattern in message_patterns:
                pattern_weights[pattern] += 1

        for pattern in pattern_weights.keys():
            # pattern_scores[pattern] = len(pattern) / math.log10(len(pattern_weights) / pattern_scores[pattern])
            pattern_weights[pattern] = len(pattern) / pattern_weights[pattern]

        self.patterns = patterns
        self.pattern_weights = pattern_weights

        print(f"Total training time: {datetime.datetime.now() - start}")

    @staticmethod
    def subtree_matches(subtree: HorizontalTree, pattern: HorizontalTree) -> bool:
        """
        >>> model = FeedbackModel()
        >>> model.subtree_matches((0, 1, 3, 7, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2 modified
        False
        >>> model.subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 2, -1, 1, -1)) # Fig 2
        False
        >>> model.subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 1, -1), (1, 2, -1, 1, -1)) # Fig 2 modified
        False
        >>> model.subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2
        True
        >>> model.subtree_matches((0, 1, 3, 1, -1, 4, -1, -1, 4, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2 modified
        False
        >>> model.subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2
        True
        >>> model.subtree_matches((0, 1, 3, 1, 2, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1), (1, 1, -1, 2, -1))
        False
        >>> model.subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1))
        True
        >>> model.subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2, -1, -1))
        True
        >>> model.subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2, -1))
        False
        >>> model.subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2, -1, -1))
        False
        >>> model.subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
        False
        >>> model.subtree_matches((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T0
        True
        >>> model.subtree_matches((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4, -1)) # Fig 5 T1
        True
        >>> model.subtree_matches((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T2
        True
        >>> model.subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
        True
        >>> model.subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
        False
        """
        if not set(pattern).issubset(subtree):
            return False

        pattern_length = len(pattern)

        start = 0
        p_i = 0
        pattern_depth = 0
        depth = 0
        depth_stack = []
        history = []

        def find_in_subtree() -> bool:
            nonlocal start, p_i, pattern_depth, depth, depth_stack

            for i, item in enumerate(subtree[start:]):
                if item == -1:
                    if depth_stack and depth - 1 == depth_stack[-1]:
                        last_depth = depth_stack.pop()
                        if pattern[p_i] != -1 and (last_depth < pattern_depth or not depth_stack):
                            p_i = 0
                        if pattern[p_i] == -1:
                            pattern_depth -= 1
                            p_i += 1
                    depth -= 1
                else:
                    if pattern[p_i] == item:
                        history.append((start + i + 1, depth + 1, depth_stack[:], pattern_depth, p_i))
                        depth_stack.append(depth)
                        pattern_depth += 1
                        p_i += 1

                    depth += 1

                if p_i == pattern_length:
                    return True

            return False

        result = find_in_subtree()
        while not result and history:
            start, depth, depth_stack, pattern_depth, p_i = history.pop()
            # Subtree needs to contain all items from pattern
            if len(pattern) - p_i <= len(subtree) - start and sequence_fully_contains_other_sequence(subtree[start:], pattern[p_i:]):
                result = find_in_subtree()

        return result

    def predict_most_likely_messages(self, tree: Tree, line: int) -> List[str] | None:
        subtree = self.find_subtree_on_line(tree, line)
        if subtree is None:
            return None

    def calculate_matching_score(self, m: str, subtree: HorizontalTree) -> float:
        pattern_set = self.patterns[m]
        matches = list(filter(lambda pattern: self.subtree_matches(subtree, pattern), pattern_set))
        matches_score = 0
        if pattern_set:
            matches_score = sum(self.pattern_weights[match] for match in matches) / len(pattern_set)

        return matches_score

    def calculate_matching_scores(self, subtree: Tree) -> Dict[str, float]:
        horizontal_subtree = list(to_string_encoding(subtree))
        matching_scores = {message: self.calculate_matching_score(message, horizontal_subtree) for message in self.patterns.keys()}
        return matching_scores
