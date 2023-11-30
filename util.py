import datetime
from collections import Counter

import pydot


def timed(func):
    start = datetime.datetime.now()
    res = func()
    print(datetime.datetime.now() - start)
    return res


def list_fully_contains_other_list(list1, list2):
    """
    Return True if list1 fully contains all items (with duplicates) form list2
    Source: https://stackoverflow.com/questions/68390939/how-to-check-if-a-list-contains-all-the-elements-of-another-list-including-dupli
    """
    return not (Counter(list2) - Counter(list1))


# tree.png from data/excercises/505886137/12989287.py

def visualise_parse_tree(tree, is_string_encoding=False, file_name="tree", with_lines=False):
    """
    Create a dot file of the parse tree that can be visualised by Graphviz
    The dot file can be turned into e.g. a png using:
    dot -Tpng tree.dot -o tree.png
    """
    graph = pydot.Dot("tree", graph_type="graph")
    if is_string_encoding:
        visualise_string_subtree(graph, tree)
    else:
        visualise_subtree(graph, tree, [0], with_lines)
    output_graphviz_dot = graph.to_string()
    with open(f'{file_name}.dot', 'w') as file:
        file.write(output_graphviz_dot)


def visualise_subtree(graph, tree, next_node_id, with_lines):
    if with_lines:
        label = f"'{tree['name']} - {str(tree['lines'])}'"
    else:
        label = f"'{tree['name']}'"
    node = pydot.Node(next_node_id[0], label=label)
    next_node_id[0] += 1
    graph.add_node(node)
    for child in tree['children']:
        child_node = visualise_subtree(graph, child, next_node_id, with_lines)
        graph.add_edge(pydot.Edge(node.get_name(), child_node.get_name()))
    return node


def visualise_string_subtree(graph, tree):
    node_id = 0
    parent_ids = [0]
    graph.add_node(pydot.Node(node_id, label=tree[0]))
    for item in tree[1:]:
        if item != -1:
            node_id += 1
            graph.add_node(pydot.Node(node_id, label=f"'{item}'"))
            graph.add_edge(pydot.Edge(parent_ids[-1], node_id))
            parent_ids.append(node_id)
        else:
            parent_ids.pop()
