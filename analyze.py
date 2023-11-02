"""
Module for matching pylint comments to a syntax tree
"""
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

PYTHON = Language("build/languages.so", "python")

parser = Parser()
parser.set_language(PYTHON)

TRAINING_FILE = "training_filtered_string"
TEST_FILE = "test_filtered_string"
PATTERNS_FILE = "patterns_extended_filtered_string"


def messages_for_file(filename: str):
    pylint_output = StringIO()
    reporter = JSONReporter(pylint_output)
    lint.Run(["--module-naming-style=any", "--disable=C0304", filename], reporter=reporter, exit=False)
    return list(map(lambda m: (m["line"] - 1, f"{m['message-id']}-{m['symbol']}"), json.loads(pylint_output.getvalue())))


def map_tree(node):
    children = [map_tree(child) for child in node.children]
    name = node.text.decode("utf-8") if node.type == "identifier" else node.type
    lines = set(range(node.start_point[0], node.end_point[0] + 1))
    return {"name": name, "lines": lines, "children": children}


def parse_file(filename):
    with open(filename, "rb") as f:
        return map_tree(parser.parse(f.read()).root_node)


def analyze_file(filename):
    tree = parse_file(filename)

    # to_remove = ['C0303-trailing-whitespace', 'C0114-missing-module-docstring', 'C0305-trailing-newlines', 'E0001-syntax-error',
    #              'C0301-line-too-long', 'E0602-undefined-variable', 'W0622-redefined-builtin', 'W0621-redefined-outer-name', 'W0612-unused-variable',
    #              'W0311-bad-indentation', 'C0103-invalid-name', 'E0601-used-before-assignment']
    to_remove = []

    return tree, [(message, line) for line, message in messages_for_file(filename) if message not in to_remove]


def find_subtree_on_line(tree, line):
    parent = None
    search_stack = [tree]
    while parent is None and len(search_stack) > 0:
        tree = search_stack.pop()
        for subtree in tree["children"]:
            lines = subtree["lines"]
            if line in lines:
                if sorted(lines)[0] == line:
                    parent = subtree
                search_stack.append(subtree)
                break

    if parent is not None:
        result = {"name": parent["name"], "children": list(filter(lambda t: line in t["lines"], parent["children"]))}
        return result

    return None


def subtree_on_line(tree, line):
    return {"name": tree["name"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}
    # return {"name": tree["name"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: len({line - 1, line, line + 1} & t["lines"]) > 0, tree["children"])))}
    # return find_subtree_on_line(tree, line)


def message_subtrees(trees):
    result = defaultdict(list)
    for item in trees.values():
        for m, line in item[1]:
            subtree = subtree_on_line(item[0], line)
            if subtree is not None:
                result[m].append(subtree)
    return result


def timed(func):
    start = datetime.datetime.now()
    res = func()
    print(datetime.datetime.now() - start)
    return res


def get_messages_set(analysis_data):
    messages_set = set()
    for (_, items) in analysis_data.values():
        messages_set.update(item[0] for item in items)

    return messages_set


def perform_analysis(files: List[str], save_analysis: bool, load_analysis: bool):
    """
    Split the files into training and testing files,
    and analyze them by parsing them and adding linter messages.
    """
    training = {}
    test = {}
    if not load_analysis:
        random.shuffle(files)

        print("Analyzing training data")
        for filename in tqdm(files[:len(files) // 2]):
            training[filename] = analyze_file(filename)

        training_messages = get_messages_set(training)

        print("Analyzing test data")
        for filename in tqdm(files[len(files) // 2:]):
            tree, messages = analyze_file(filename)
            # If there are messages in test files that are not present in any training files, remove them from the test set
            messages = list(filter(lambda x: x[0] in training_messages, messages))
            test[filename] = (tree, messages)

        if save_analysis:
            with open(f'output/analysis/{TRAINING_FILE}', 'wb') as training_analysis_file:
                pickle.dump(training, training_analysis_file, pickle.HIGHEST_PROTOCOL)
            with open(f'output/analysis/{TEST_FILE}', 'wb') as testing_analysis_file:
                pickle.dump(test, testing_analysis_file, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading analysis data")
        with open(f'output/analysis/{TRAINING_FILE}', 'rb') as training_analysis_file:
            training = pickle.load(training_analysis_file)
        with open(f'output/analysis/{TEST_FILE}', 'rb') as testing_analysis_file:
            test = pickle.load(testing_analysis_file)

    return training, test


def find_patterns(message, ts):
    message_patterns = []
    if len(ts) >= 3:
        message_patterns = Treeminerd(ts, support=0.8).get_patterns()

    return message, message_patterns


def determine_patterns(training, save_patterns: bool, load_patterns: bool):
    """
    Determine the patterns present in the trees in the training set.
    """
    if not load_patterns:
        start = datetime.datetime.now()

        subtrees = message_subtrees(training)
        print("Determining patterns for training data")
        patterns = {}

        results = pqdm(list(subtrees.items()), find_patterns, n_jobs=8, argument_type='args')
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

        if save_patterns:
            with open(f'output/patterns/{PATTERNS_FILE}', 'wb') as patterns_file:
                pickle.dump((patterns, pattern_scores), patterns_file, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading patterns data")
        with open(f'output/patterns/{PATTERNS_FILE}', 'rb') as patterns_file:
            patterns, pattern_scores = pickle.load(patterns_file)

    return patterns, pattern_scores


def analyze(save_analysis=True, load_analysis=False, save_patterns=True, load_patterns=False):
    files = glob('submissions/*/*/*.py')

    random.seed(314159)

    training, test = perform_analysis(files, save_analysis, load_analysis)
    patterns, pattern_scores = determine_patterns(training, save_patterns, load_patterns)

    return training, test, patterns, pattern_scores


if __name__ == '__main__':
    analyze(load_analysis=True, load_patterns=False)
