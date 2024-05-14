import math
from collections import Counter
from glob import glob
from typing import List, Iterator, Sequence

import pydot

from analyze import FeedbackAnalyzer
from constants import ROOT_DIR, ENGLISH_EXERCISE_NAMES_MAP
from custom_types import Tree, HorizontalTree


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


def get_dataset_stats(print_annotations=False):
    ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']

    for e_id in ids:
        analyzer = FeedbackAnalyzer()
        analyzer.set_files(glob(f'{ROOT_DIR}/data/exercises/{e_id}/*.py'), annotations_only=False)
        test = analyzer.analyze_files()

        annotation_ids = set()

        min_annotation_instances = math.inf
        max_annotation_instances = 0

        annotation_instances_counter = Counter()

        for _, annotation_instances in test.values():
            for a_id, _ in annotation_instances:
                annotation_ids.add(a_id)
                annotation_instances_counter[a_id] += 1
            if len(annotation_instances) < min_annotation_instances:
                min_annotation_instances = len(annotation_instances)
            if len(annotation_instances) > max_annotation_instances:
                max_annotation_instances = len(annotation_instances)

        print(f"Exercise '{ENGLISH_EXERCISE_NAMES_MAP[e_id]}' ({e_id}): ")
        print(f"# solutions: {len(test)}")
        print(f"# annotations: {len(annotation_ids)}")
        print(f"# annotation instances: {annotation_instances_counter.total()}")
        print(f"min instances/file: {min_annotation_instances}")
        print(f"max instances/file: {max_annotation_instances}")
        print(f"average instances/file: {annotation_instances_counter.total() / len(test)}")
        print(f"min instances/annotation: {min(annotation_instances_counter.values())}")
        print(f"max instances/annotation: {max(annotation_instances_counter.values())}")
        print(f"average instances/annotation: {annotation_instances_counter.total() / len(annotation_instances_counter)}")
        if print_annotations:
            print("Annotations: ")
            for a_id in annotation_ids:
                print(f"{a_id}: {analyzer.id_annotation_map[a_id]}")
        print()


if __name__ == '__main__':
    visualise_parse_tree(["a", "b", "d", "a", -1, "c", -1, -1, "c", -1, "c"], is_string_encoding=True, file_name="examples/not_matching_subtree")
