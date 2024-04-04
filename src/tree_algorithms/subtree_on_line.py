from custom_types import Tree, LineTree


def _keep_only_nodes_of_line(tree: LineTree, line: int) -> Tree:
    return {"name": tree["name"],
            "children": list(map(lambda t: _keep_only_nodes_of_line(t, line), filter(lambda t: line in t["lines"], tree["children"])))}


def find_subtree_on_line(tree: LineTree, line: int) -> Tree | None:
    """
    Find the subtree corresponding to the code on the specified line
    """
    context_nodes = ["if_statement", "for_statement", "while_statement", "with_statement", "try_statement", "else_clause", "elif_clause", "except_clause", "finally_clause"]

    subtree_on_line = None
    context_root = None
    context = None
    next_child_is_function_name = False

    while subtree_on_line is None:
        children = tree["children"]
        parent_tree = tree
        i = 0
        tree = None
        lines = []
        next_node_is_function_name = next_child_is_function_name
        # Go through all child nodes, until we are sure we have passed the required line
        while i < len(children) and (not lines or lines[0] <= line):
            subtree = children[i]
            lines = subtree["lines"]
            if context_root is None and next_node_is_function_name:
                context = []
                context_root = {"name": subtree["name"], "children": context}
            # If the line is in the current subtree
            if line in lines:
                next_child_is_function_name = subtree["name"] == "function_definition"
                if tree is None:
                    # Set up the next tree to search in
                    tree = subtree
                    # If the first number in lines is the required line, we may have found our subtree
                    if lines[0] == line:
                        subtree_on_line = subtree
                    elif context is not None and subtree["name"] in context_nodes:
                        next_context = []
                        context.append({"name": subtree["name"], "children": next_context})
                        context = next_context
                else:
                    # If we already found a next tree to search in a previous iteration, but the line is also in this iteration's lines (can happen if e.g. an expression is split over multiple lines)
                    # Set the subtree_on_line to the parent of the current subtree
                    subtree_on_line = parent_tree

            i += 1

        if tree is None:
            return None

    result = _keep_only_nodes_of_line(subtree_on_line, line)

    if context_root is not None:
        context.append(result)
        result = context_root

    return result

