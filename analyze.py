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


def subtree_on_line(tree, line):
    # return {"name": tree["name"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}
    return {"name": tree["name"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: len({line - 1, line, line + 1} & t["lines"]) > 0, tree["children"])))}


def message_subtrees(trees):
    result = defaultdict(list)
    for item in trees.values():
        for m, line in item[1]:
            result[m].append(subtree_on_line(item[0], line))
    return result


def timed(func):
    start = datetime.datetime.now()
    res = func()
    print(datetime.datetime.now() - start)
    return res


def subtree_matches_2(subtree, pattern):
    """
    >>> subtree_matches_2((0, 1, 3, 7, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 2, -1, 1, -1)) # Fig 2
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 1, -1), (1, 2, -1, 1, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_2((0, 1, 3, 1, -1, 4, -1, -1, 4, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_2((0, 1, 3, 1, 2, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1), (1, 1, -1, 2, -1))
    False
    >>> subtree_matches_2((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1))
    True
    >>> subtree_matches_2((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2, -1, -1))
    True
    >>> subtree_matches_2((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2, -1))
    False
    >>> subtree_matches_2((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2, -1, -1))
    False
    >>> subtree_matches_2((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches_2((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T0
    True
    >>> subtree_matches_2((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4, -1)) # Fig 5 T1
    True
    >>> subtree_matches_2((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T2
    True
    >>> subtree_matches_2((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches_2((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    """
    p_i = 0
    pattern_depth = 0
    depth = 0
    depth_stack = []
    for i, item in enumerate(subtree):
        if item == -1:
            if len(depth_stack) > 0 and depth - 1 == depth_stack[-1]:
                last_depth = depth_stack.pop()
                if pattern[p_i] != -1 and (last_depth < pattern_depth or len(depth_stack) == 0):
                    p_i = 0
            depth -= 1
        else:
            if pattern[p_i] == item and item != -1:
                depth_stack.append(depth)
            depth += 1

        if pattern[p_i] == item:
            if item == -1:
                pattern_depth -= 1
            else:
                pattern_depth += 1
            p_i += 1

        if p_i == len(pattern):
            return True

    return False


def subtree_matches(subtree, pattern):
    """
    >>> subtree_matches((0, 1, 3, 7, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 2, -1, 1, -1)) # Fig 2
    False
    >>> subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 1, -1), (1, 2, -1, 1, -1)) # Fig 2 modified
    False
    >>> subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches((0, 1, 3, 1, -1, 4, -1, -1, 4, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches((0, 1, 3, 1, 2, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1), (1, 1, -1, 2, -1))
    False
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1))
    True
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2))
    True
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2))
    False
    >>> subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2))
    False
    >>> subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4)) # Fig 5 T0
    True
    >>> subtree_matches((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4)) # Fig 5 T1
    True
    >>> subtree_matches((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4)) # Fig 5 T2
    True
    >>> subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    """
    index = 0
    depth = 0
    depth_stack = [0]
    for atom in pattern:
        if atom == -1:
            while index < len(subtree) and depth > depth_stack[-1]:
                if subtree[index] == -1:
                    depth -= 1
                else:
                    depth += 1
                index += 1
            if index == len(subtree):
                return False
            del depth_stack[-1]
        else:
            while index < len(subtree) and depth > depth_stack[-1] and subtree[index] != atom:
                if subtree[index] == -1:
                    depth -= 1
                else:
                    depth += 1
                index += 1
            if index == len(subtree) or depth < depth_stack[-1]:
                return False
            depth_stack.append(depth)
            index += 1
            depth += 1
    return True


def most_likely_messages(pattern_collection, subtree):
    messages = list(pattern_collection.keys())
    messages.sort(
        key=lambda m: -len(list(filter(lambda pattern: subtree_matches(subtree, pattern), pattern_collection[m]))) / len(pattern_collection[m]))
    return messages


def result_for_file(patterns, f, messages):
    total = Counter()
    first = Counter()
    first_three = Counter()
    tree = messages[0]
    for m, line in messages[1]:
        subtree = list(to_string_encoding(subtree_on_line(tree, line)))
        total[m] += 1
        matched = most_likely_messages(patterns, subtree)
        if m == matched[0]:
            first[m] += 1
        if m in matched[:3]:
            first_three[m] += 1
    return total, first, first_three


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
        print("Analyzing test data")
        for filename in tqdm(files[len(files) // 2:]):
            test[filename] = analyze_file(filename)
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

        # for m, ts in tqdm(subtrees.items()):
        #     if len(ts) >= 3:
        #         message_patterns = Treeminerd(ts, support=0.8).get_patterns()
        #         if len(message_patterns) != 0:
        #             patterns[m] = message_patterns
        print(f"Total training time: {datetime.datetime.now() - start}")

        if save_patterns:
            with open(f'output/patterns/{PATTERNS_FILE}', 'wb') as patterns_file:
                pickle.dump(patterns, patterns_file, pickle.HIGHEST_PROTOCOL)
    else:
        print("Loading patterns data")
        with open(f'output/patterns/{PATTERNS_FILE}', 'rb') as patterns_file:
            patterns = pickle.load(patterns_file)

    return patterns


def analyze(save_analysis=True, load_analysis=False, save_patterns=True, load_patterns=False):
    files = glob('submissions/*/*/*.py')

    random.seed(314159)

    training, test = perform_analysis(files, save_analysis, load_analysis)
    patterns = determine_patterns(training, save_patterns, load_patterns)

    return training, test, patterns


if __name__ == '__main__':
    analyze(load_analysis=True, load_patterns=False)
