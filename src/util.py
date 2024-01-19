from collections import Counter
from typing import List, Iterator, Sequence

import pydot

from src.constants import ROOT_DIR
from src.custom_types import Tree, HorizontalTree


def to_string_encoding(tree: Tree) -> Iterator[str | int]:
    """
    Convert a tree into it's "string" encoding. Note that this is actually a
    generator of "atoms", to be collected in a list or other sequential type.
    """
    yield tree["name"]
    for child in tree["children"]:
        for elem in to_string_encoding(child):
            yield elem
        yield -1


def sequence_fully_contains_other_sequence(list1: Sequence, list2: Sequence) -> bool:
    """
    Return True if list1 fully contains all items (with duplicates) form list2
    Source: https://stackoverflow.com/questions/68390939/how-to-check-if-a-list-contains-all-the-elements-of-another-list-including-dupli
    """
    return not (Counter(list2) - Counter(list1))


# tree.png from data/excercises/505886137/12989287.py

def visualise_parse_tree(tree: Tree | HorizontalTree, is_string_encoding=False, file_name="tree") -> None:
    """
    Create a dot file of the parse tree that can be visualised by Graphviz
    The dot file can be turned into e.g. a png using:
    dot -Tpng tree.dot -o tree.png
    """
    graph = pydot.Dot("tree", graph_type="graph")
    graph.set_splines(False)
    if is_string_encoding:
        visualise_string_subtree(graph, tree)
    else:
        visualise_subtree(graph, tree, [0])
    output_graphviz_dot = graph.to_string()
    with open(f'{ROOT_DIR}/output/dots/{file_name}.dot', 'w') as file:
        file.write(output_graphviz_dot)


def visualise_subtree(graph: pydot.Dot, tree: Tree, next_node_id: List[int]) -> pydot.Node:
    label = f"'{tree['name']}'"
    node = pydot.Node(next_node_id[0], label=label)
    next_node_id[0] += 1
    graph.add_node(node)
    for child in tree['children']:
        child_node = visualise_subtree(graph, child, next_node_id)
        graph.add_edge(pydot.Edge(node.get_name(), child_node.get_name()))
    return node


def visualise_string_subtree(graph: pydot.Dot, tree: HorizontalTree) -> None:
    node_id = 0
    parent_ids = [0]
    graph.add_node(pydot.Node(node_id, label=f"'{tree[0]}'"))
    for item in tree[1:]:
        if item != -1:
            node_id += 1
            graph.add_node(pydot.Node(node_id, label=f"'{item}'"))
            graph.add_edge(pydot.Edge(parent_ids[-1], node_id))
            parent_ids.append(node_id)
        else:
            parent_ids.pop()


if __name__ == '__main__':
    visualise_parse_tree(["a", "b", "d", "a", -1, "c", -1, -1, "c", -1, "c"], is_string_encoding=True, file_name="examples/not_matching_subtree")
