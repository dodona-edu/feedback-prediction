#!/usr/bin/env -S python -i
"""
Module for matching pylint comments to a syntax tree
"""
from io import StringIO
import json
from functools import reduce
from glob import glob
from freqt import treeminer, to_string_encoding
import pickle

from pylint import lint
from pylint.reporters.json_reporter import JSONReporter
from tqdm import tqdm
from tree_sitter import Language, Parser

PYTHON = Language("build/languages.so", "python")

parser = Parser()
parser.set_language(PYTHON)

def messages_for_file(filename):
    pylint_output = StringIO()
    reporter = JSONReporter(pylint_output)
    lint.Run(["--module-naming-style=any", "--disable=C0304", filename], reporter=reporter, do_exit=False)
    return list(map(lambda m: (m["line"] - 1, f"{m['message-id']}-{m['symbol']}"), json.loads(pylint_output.getvalue())))

def map_tree(node):
    return { "name": node.text.decode("utf-8") if node.type == "identifier" else node.type, "lines": set(range(node.start_point[0], node.end_point[0] + 1)), "children": [map_tree(child) for child in node.children] }

def parse_file(filename):
    with open(filename, "rb") as f:
        return map_tree(parser.parse(f.read()).root_node)
    
def nodes_for_line(tree, n):
    """
    Get all nodes that are fully on the given line.
    """
    queue = [tree]
    result = []
    while queue:
        node = queue.pop(0)
        if n in node["lines"]:
            result.append(node["name"])
            queue = node["children"] + queue

    return result
    
def analyze_file(filename):
    tree = parse_file(filename)

    return (tree, [(message, line, nodes_for_line(tree, line)) for line, message in messages_for_file(filename)])

def paths_for_message(analysis, message):
    result = []
    for item in analysis.values():
        item = item[1]
        result += [x[2] for x in item if x[0] == message]
    return result

def edit_distance(path1, path2):
    dist = [[0] * (len(path2) + 1) for _ in range(len(path1) + 1)]
    for i in range(1, len(path1) + 1):
        dist[i][0] = i
    for j in range(1, len(path2) + 1):
        dist[0][j] = j

    for j in range(len(path2)):
        for i in range(len(path1)):
            if path1[i] == path2[j]:
                cost = 0
            else:
                cost = 1
            dist[i + 1][j + 1] = min(dist[i][j + 1] + 1, dist[i + 1][j] + 1, dist[i][j] + cost)

    return dist[len(path1)][len(path2)]
    

if __name__ == '__main__':
    # result = {}
    # for filename in tqdm(glob('submissions/exam-group-1-examen-groep-1/*/*/*.py')):
    #     result[filename] = analyze_file(filename)
    # with open('result.pickle', 'wb') as f:
    #     pickle.dump(result, f)
    with open('result.pickle', 'rb') as f:
        result = pickle.load(f)
    # messages = reduce(lambda x, y: x | y, [{x for x in map(lambda x: x[0], item[1])} for item in result.values()])
    # paths = paths_for_message(result, 'R1714-consider-using-in')
    # matrix = [[0] * len(paths) for _ in paths]
    # for i, p1 in enumerate(paths):
    #     for j, p2 in enumerate(paths):
    #         matrix[i][j] = edit_distance(p1, p2)

    # print(max(reduce(lambda x, y: x + y, matrix))) # In centrifuge: 57 (59 exam 1)
    # print(sum(reduce(lambda x, y: x + y, matrix)) / (len(matrix) ** 2)) # In centrifuge: 13.23 (13.40 exam 2)
