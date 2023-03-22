"""
Module for matching pylint comments to a syntax tree
"""
from io import StringIO
import json
from functools import reduce
from glob import glob
from treeminer import Treeminerd, to_string_encoding
import pickle
import datetime

from pylint import lint
from pylint.reporters.json_reporter import JSONReporter
from tqdm import tqdm
from tree_sitter import Language, Parser

PYTHON = Language("build/languages.so", "python")
# FORBIDDEN_LEAFS = {'.', ':', '(', ')', ',', 'def', '"', "'", '!=', '-', '==', '='}
FORBIDDEN_LEAFS = {'(', ')', ':', '.', '"', ','}

parser = Parser()
parser.set_language(PYTHON)

def messages_for_file(filename):
    pylint_output = StringIO()
    reporter = JSONReporter(pylint_output)
    lint.Run(["--module-naming-style=any", "--disable=C0304", filename], reporter=reporter, do_exit=False)
    return list(map(lambda m: (m["line"] - 1, f"{m['message-id']}-{m['symbol']}"), json.loads(pylint_output.getvalue())))

def map_tree(node):
    children = [map_tree(child) for child in node.children if child.type not in FORBIDDEN_LEAFS]
    name = node.text.decode("utf-8") if node.type == "identifier" else node.type
    lines = set(range(node.start_point[0], node.end_point[0] + 1))
    if len(children) == 1:
        name += " " + children[0]["name"]
        children = children[0]["children"]
    return { "name": name, "lines": lines, "children": children}

def parse_file(filename):
    with open(filename, "rb") as f:
        return map_tree(parser.parse(f.read()).root_node)
    
def analyze_file(filename):
    tree = parse_file(filename)

    return (tree, [(message, line) for line, message in messages_for_file(filename)])

def subtree_on_line(tree, line):
    return {"name": tree["name"], "lines": tree["lines"], "children": list(map(lambda t: subtree_on_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}
    pass

def message_subtrees(trees, message):
    result = []
    for item in trees.values():
        for m, line in item[1]:
            if m == message:
                result.append(subtree_on_line(item[0], line))
    return result

def timed(func):
    start = datetime.datetime.now()
    res = func()
    print(datetime.datetime.now() - start)
    return res


if __name__ == '__main__':
    # result = {}
    # for filename in tqdm(glob('submissions/exam-group-1-examen-groep-1/*/*.py')):
    #     result[filename] = analyze_file(filename)
    # with open('result.pickle', 'wb') as f:
    #     pickle.dump(result, f)
    with open('result.pickle', 'rb') as f:
        result = pickle.load(f)

    subtrees = message_subtrees(result, 'R1714-consider-using-in')
    patterns = timed(lambda: Treeminerd(subtrees).get_patterns())
