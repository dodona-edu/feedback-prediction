"""
Module for matching pylint comments to a syntax tree
"""
from io import StringIO
import json
from functools import reduce
from glob import glob
from treeminer import Treeminer, to_string_encoding
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

def split_with_message(trees, message):
    result = [], []
    for item in trees.values():
        if any(x[0] == message for x in item[1]):
            result[0].append(item[0])
        else:
            result[1].append(item[0])
    return result

if __name__ == '__main__':
    result = {}
    # for filename in tqdm(glob('submissions/exam-group-1-examen-groep-1/*/*.py')):
    #     result[filename] = analyze_file(filename)
    # with open('result.pickle', 'wb') as f:
    #     pickle.dump(result, f)
    with open('result.pickle', 'rb') as f:
        result = pickle.load(f)

    haves, have_nots = split_with_message(result, 'R1714-consider-using-in')
