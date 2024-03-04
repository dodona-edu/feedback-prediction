from custom_types import Tree, LineTree


def _keep_only_nodes_of_line(tree: LineTree, line: int):
    return {"name": tree["name"],
            "children": list(map(lambda t: _keep_only_nodes_of_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}


def find_subtree_on_line(tree: LineTree, line: int) -> Tree | None:
    """
    Find the subtree corresponding to the code on the specified line
    """
    subtree_on_line = None
    while subtree_on_line is None:
        children = tree["children"]
        parent_tree = tree
        i = 0
        tree = None
        lines = []
        # Go through all child nodes, until we are sure we have passed the required line
        while i < len(children) and (not lines or lines[0] <= line):
            subtree = children[i]
            lines = subtree["lines"]
            # If the line is in the current subtree
            if line in lines:
                if tree is None:
                    # Set up the next tree to search in
                    tree = subtree
                    # If the first number in lines is the required line, we may have found our subtree
                    if lines[0] == line:
                        subtree_on_line = subtree
                else:
                    # If we already found a next tree to search in a previous iteration, but the line is also in this iteration's lines (can happen if e.g. an expression is split over multiple lines)
                    # Set the subtree_on_line to the parent of the current subtree
                    subtree_on_line = parent_tree
            i += 1

        if tree is None:
            return None

    if subtree_on_line is not None:
        result = {"name": subtree_on_line["name"],
                  "children": list(map(lambda t: _keep_only_nodes_of_line(t, line), filter(lambda t: line in t["lines"], subtree_on_line["children"])))
                  }

        return result

    return None
