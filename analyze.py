"""
Module for matching annotations to a syntax tree
"""
import csv
from abc import ABC, abstractmethod
from io import StringIO
import json
from collections import defaultdict
from glob import glob
from typing import List

from treeminer import Treeminerd
import pickle
import datetime
import random

from pylint import lint
from pylint.reporters.json_reporter import JSONReporter
from pqdm.processes import pqdm
from tqdm import tqdm
from tree_sitter import Language, Parser


def keep_only_nodes_of_line(tree, line: int):
    return {"name": tree["name"], "children": list(map(lambda t: keep_only_nodes_of_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}


def find_subtree_on_line(tree, line):
    """
    Find the subtree that has a root such that 'line' is its first child
    """
    root = None
    while root is None:
        i = 0
        children = tree["children"]
        subtree = None
        while i < len(children) and tree != subtree:
            subtree = children[i]
            lines = subtree["lines"]
            if line in lines:
                tree = subtree
                if sorted(lines)[0] == line:
                    root = subtree
            i += 1

        if tree != subtree:
            return None

    if root is not None:
        # result = {"name": root["name"], "children": list(filter(lambda t: line in t["lines"], root["children"]))}
        result = {"name": root["name"], "children": list(map(lambda t: keep_only_nodes_of_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}

        return result

    return None


def subtree_on_line(tree, line):
    # return {"name": tree["name"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}
    # return {"name": tree["name"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: len({line - 1, line, line + 1} & t["lines"]) > 0, tree["children"])))}
    return find_subtree_on_line(tree, line)


def find_patterns(message: str, ts):
    message_patterns = []
    if len(ts) >= 3:
        message_patterns = Treeminerd(ts, support=0.8).get_patterns()

    return message, message_patterns


class Analyzer(ABC):

    ANALYSIS_DIR: str
    PATTERNS_DIR: str

    TRAINING_FILE: str
    TEST_FILE: str
    PATTERNS_FILE: str

    def __init__(self, save_analysis=True, load_analysis=False, save_patterns=True, load_patterns=False):
        PYTHON = Language("build/languages.so", "python")
        parser = Parser()
        parser.set_language(PYTHON)
        self.parser = parser

        self.save_analysis = save_analysis
        self.load_analysis = load_analysis
        self.save_patterns = save_patterns
        self.load_patterns = load_patterns

        self.filter_test = True  # Whether to filter the test set or not. This will remove all messages from the test set for which no pattern was found.

        self.files: List[str] = []

    def map_tree(self, node):
        children = [self.map_tree(child) for child in node.children if node.type != "comment"]
        name = node.text.decode("utf-8") if node.type == "identifier" else node.type
        lines = set(range(node.start_point[0], node.end_point[0] + 1))
        # TODO hier sorted(lines) ipv in find_subtree_on_line() als beslist zou worden om sowieso subtree_on_line te gebruiken
        return {"name": name, "lines": lines, "children": children}

    def parse_file(self, file: str):
        with open(file, "rb") as f:
            return self.map_tree(self.parser.parse(f.read()).root_node)

    @abstractmethod
    def messages_for_file(self, file: str):
        pass

    def analyze_file(self, file: str):
        tree = self.parse_file(file)

        return tree, [(message, line) for line, message in self.messages_for_file(file)]

    def get_messages_set(self, analysis_data):
        messages_set = set()
        for (_, items) in analysis_data.values():
            messages_set.update(item[0] for item in items)

        return messages_set

    def perform_analysis(self, files: List[str]):
        """
        Split the files into training and testing files,
        and analyze them by parsing them and adding messages.
        """
        training = {}
        test = {}
        if not self.load_analysis:
            random.shuffle(files)

            print("Analyzing training data")
            for filename in tqdm(files[:len(files) // 2]):
                training[filename] = self.analyze_file(filename)

            training_messages = self.get_messages_set(training)

            print("Analyzing test data")
            for filename in tqdm(files[len(files) // 2:]):
                tree, messages = self.analyze_file(filename)
                if self.filter_test:
                    # If there are messages in test files that are not present in any training files, remove them from the test set
                    messages = list(filter(lambda x: x[0] in training_messages, messages))
                test[filename] = (tree, messages)

            if self.save_analysis:
                with open(f'{self.ANALYSIS_DIR}/{self.TRAINING_FILE}', 'wb') as training_analysis_file:
                    pickle.dump(training, training_analysis_file, pickle.HIGHEST_PROTOCOL)
                with open(f'{self.ANALYSIS_DIR}/{self.TEST_FILE}', 'wb') as testing_analysis_file:
                    pickle.dump(test, testing_analysis_file, pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading analysis data")
            with open(f'{self.ANALYSIS_DIR}/{self.TRAINING_FILE}', 'rb') as training_analysis_file:
                training = pickle.load(training_analysis_file)
            with open(f'{self.ANALYSIS_DIR}/{self.TEST_FILE}', 'rb') as testing_analysis_file:
                test = pickle.load(testing_analysis_file)

        return training, test

    def message_subtrees(self, trees):
        result = defaultdict(list)
        for key, item in trees.items():
            for m, line in item[1]:
                subtree = subtree_on_line(item[0], line)
                if subtree is not None:
                    result[m].append(subtree)
        return result

    def determine_patterns(self, training):
        """
        Determine the patterns present in the trees in the training set.
        """
        if not self.load_patterns:
            start = datetime.datetime.now()

            subtrees = self.message_subtrees(training)
            print("Determining patterns for training data")
            patterns = {}

            results = pqdm(list(subtrees.items()), find_patterns, n_jobs=8, argument_type='args')
            # results = []
            # for m, ts in tqdm(subtrees.items()):
            #     results.append(find_patterns(m, ts))

            for m, res in results:
                if len(res) > 0:
                    patterns[m] = res

            print("Calculating pattern scores")
            pattern_scores = defaultdict(float)
            for message_patterns in patterns.values():
                for pattern in message_patterns:
                    pattern_scores[pattern] += 1

            for pattern in pattern_scores.keys():
                pattern_scores[pattern] = len(pattern) / pattern_scores[pattern]

            print(f"Total training time: {datetime.datetime.now() - start}")

            if self.save_patterns:
                with open(f'{self.PATTERNS_DIR}/{self.PATTERNS_FILE}', 'wb') as patterns_file:
                    pickle.dump((patterns, pattern_scores), patterns_file, pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading patterns data")
            with open(f'{self.PATTERNS_DIR}/{self.PATTERNS_FILE}', 'rb') as patterns_file:
                patterns, pattern_scores = pickle.load(patterns_file)

        return patterns, pattern_scores

    def analyze(self):
        random.seed(314159)

        training, test = self.perform_analysis(self.files)
        patterns, pattern_scores = self.determine_patterns(training)

        if self.filter_test:
            # Filter our the messages for which no pattern was found
            for file in test.keys():
                test[file] = (test[file][0], [(m, line) for m, line in test[file][1] if m in patterns.keys()])

        return training, test, patterns, pattern_scores


class PylintAnalyzer(Analyzer):

    def __init__(self, save_analysis=True, load_analysis=False, save_patterns=True, load_patterns=False):
        super().__init__(save_analysis, load_analysis, save_patterns, load_patterns)

        self.ANALYSIS_DIR = "pylint/output/analysis"
        self.PATTERNS_DIR = "pylint/output/patterns"

        self.TRAINING_FILE = "training"
        self.TEST_FILE = "test"
        self.PATTERNS_FILE = "patterns_smallsub_pos_neg"

        self.files = glob('pylint/submissions/*/*/*.py')

    def messages_for_file(self, file: str):
        pylint_output = StringIO()
        reporter = JSONReporter(pylint_output)
        lint.Run(["--module-naming-style=any", "--disable=C0304", file], reporter=reporter, exit=False)
        return list(map(lambda m: (m["line"] - 1, f"{m['message-id']}-{m['symbol']}"), json.loads(pylint_output.getvalue())))


class FeedbackAnalyzer(Analyzer):

    def __init__(self, save_analysis=True, load_analysis=False, save_patterns=True, load_patterns=False):
        super().__init__(save_analysis, load_analysis, save_patterns, load_patterns)

        self.ANALYSIS_DIR = "output/analysis"
        self.PATTERNS_DIR = "output/patterns"

        self.TRAINING_FILE = "training"
        self.TEST_FILE = "test"
        self.PATTERNS_FILE = "patterns_smallsub_pos_neg"

        self.files = glob('data/excercises/*/*.py')
        self.annotations_file = "data/annotations.tsv"

        self.submission_annotations_map = defaultdict(list)
        self.load_submission_annotations_map()

        self.files = [file for file in self.files if file.split('/')[-1].split('.')[0] in self.submission_annotations_map.keys()]

    def load_submission_annotations_map(self):
        with open(self.annotations_file) as annotations_file:
            rows = csv.reader(annotations_file, delimiter='\t')
            next(rows)
            for row in rows:
                if row[1] != 'NULL':
                    self.submission_annotations_map[row[2]].append((int(row[1]), row[4]))

    def messages_for_file(self, file: str):
        submission_id = file.split('/')[-1].split('.')[0]
        return self.submission_annotations_map[submission_id]


if __name__ == '__main__':
    # analyzer = PylintAnalyzer(save_patterns=False)
    analyzer = FeedbackAnalyzer(save_analysis=False)
    analyzer.analyze()
