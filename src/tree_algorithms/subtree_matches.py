from typing import List

from custom_types import HorizontalTree
from util import sequence_fully_contains_other_sequence


def find_in_subtree(current_subtree: HorizontalTree, pattern: HorizontalTree, pattern_length: int, history: List,
                    p_i: int, depth: int, depth_stack: List[int]) -> bool:
    local_history = []
    for i, item in enumerate(current_subtree):
        # We go up in the tree
        if item == -1:
            # If the depth of the last matched node is equal to what the depth will be after this iteration
            if depth_stack and depth - 1 == depth_stack[-1]:
                # We go up past the last node
                depth_stack.pop()
                # If in the pattern, we are not supposed to go up
                if pattern[p_i] != -1:
                    # Reset the pattern index
                    p_i = 0
                    # If we are past the first matched node, we can limit our future backtracking in local_history to a new subtree
                    if not depth_stack:
                        history.append((local_history, current_subtree[:i + 1]))
                        local_history = []
                    depth_stack = []
                else:
                    p_i += 1
            depth -= 1
        else:
            if pattern[p_i] == item:
                local_history.append((i + 1, depth + 1, depth_stack[:], p_i))
                depth_stack.append(depth)
                p_i += 1

            depth += 1

        if p_i == pattern_length:
            return True

    if local_history:
        history.append((local_history, current_subtree))

    return False


def subtree_matches(subtree: HorizontalTree, pattern: HorizontalTree) -> bool:
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
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1))
    True
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2, -1, -1))
    True
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2, -1))
    False
    >>> subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2, -1, -1))
    False
    >>> subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T0
    True
    >>> subtree_matches((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4, -1)) # Fig 5 T1
    True
    >>> subtree_matches((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T2
    True
    >>> subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    >>> subtree_matches((0, 1, 3, 1, -1, 4, -1, -1, 5, -1, -1, 6, -1, 1, 2, -1, -1), (0, 1, 2, -1, -1))
    True
    >>> subtree_matches((4, 0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1, -1, 5, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1))
    True
    >>> subtree_matches(('expression_statement', 'assignment', 'uitgang', -1, '=', -1, 'list', '[', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ']', -1, -1, -1), ('list', 'call', 'range', -1, '(', -1, '('))
    False
    >>> subtree_matches(('expression_statement', 'augmented_assignment', 'output', -1, '+=', -1, 'subscript', 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'attribute', 'attribute', 'self', -1, '.', -1, 'codec', -1, -1, '.', -1, 'keys', -1, -1, 'argument_list', '(', -1, ')', -1, -1, -1, ')', -1, -1, -1, '[', -1, 'call', 'attribute', 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'attribute', 'attribute', 'self', -1, '.', -1, 'codec', -1, -1, '.', -1, 'values', -1, -1, 'argument_list', '(', -1, ')', -1, -1, -1, ')', -1, -1, -1, '.', -1, 'index', -1, -1, 'argument_list', '(', -1, 'element', -1, ')', -1, -1, -1, ']', -1, -1, -1), ('call', 'attribute', '.', -1, -1, '(', -1, 'self'))
    False
    >>> subtree_matches(('function_definition', 'def', -1, 'verborgen_coderen', -1, 'parameters', '(', -1, 'self', -1, ',', -1, 'klare_tekst', -1, ',', -1, 'default_parameter', 'a', -1, '=', -1, 'none', -1, -1, ',', -1, 'something', 'b', -1, '=', -1, 'none', -1, -1, ')', -1, -1, ':', -1, 'block', -1), ('function_definition', 'def', -1, 'parameters', '(', -1, 'default_parameter', 'b', -1, '=', -1, 'none', -1, -1, -1, 'block'))
    False
    >>> subtree_matches(('function_definition', 'def', -1, 'verborgen_coderen', -1, 'parameters', '(', -1, 'self', -1, ',', -1, 'klare_tekst', -1, ',', -1, 'default_parameter', 'a', -1, '=', -1, 'none', -1, -1, ',', -1, 'default_parameter', 'b', -1, '=', -1, 'none', -1, -1, ')', -1, -1, ':', -1, 'block', -1), ('parameters', '(', -1, 'self', -1, ',', -1, 'default_parameter', 'b', -1, -1, ')'))
    True
    >>> subtree_matches(('function_definition', 'def', -1, 'verborgen_coderen', -1, 'parameters', '(', -1, 'self', -1, ',', -1, 'klare_tekst', -1, ',', -1, 'default_parameter', 'a', -1, '=', -1, 'none', -1, -1, ',', -1, 'default_parameter', 'b', -1, '=', -1, 'none', -1, -1, ')', -1, -1, ':', -1, 'block', -1), ('function_definition', 'def', -1, 'parameters', '(', -1, 'default_parameter', 'b', -1, '=', -1, 'none', -1, -1, -1, 'block'))
    True
    >>> subtree_matches(('function_definition', 'def', -1, 'verborgen_coderen', -1, 'parameters', '(', -1, 'self', -1, ',', -1, 'klare_tekst', -1, ',', -1, 'default_parameter', 'a', -1, '=', -1, 'none', -1, -1, ',', -1, 'default_parameter', 'b', -1, '=', -1, 'none', -1, -1, ')', -1, -1, ':', -1, 'block', -1), ('function_definition', 'def', -1, 'parameters', '(', -1, ',', -1, 'default_parameter', 'b', -1, '=', -1, 'none', -1, -1, -1, ':'))
    True
    >>> subtree_matches(('expression_statement', 'assignment', 'lijst', -1, '=', -1, 'list_comprehension', '[', -1, 'call', 'read_cocktail', -1, 'argument_list', '(', -1, 'line', -1, ')', -1, -1, -1, 'for_in_clause', 'for', -1, 'line', -1, 'in', -1, 'call', 'map', -1, 'argument_list', '(', -1, 'attribute', 'str', -1, '.', -1, 'strip', -1, -1, ',', -1, 'call', 'open', -1, 'argument_list', '(', -1, 'file', -1, ',', -1, 'string', '"', -1, '"', -1, -1, ',', -1, 'keyword_argument', 'encoding', -1, '=', -1, 'string', '"', -1, '"', -1, -1, -1, ')', -1, -1, -1, ')', -1, -1, -1, -1, ']', -1, -1, -1), ('expression_statement', 'assignment', '=', -1, 'call', 'argument_list', '(', -1, ',', -1, '"', -1, '"'))
    True
    >>> subtree_matches(('expression_statement', 'assignment', 'lijst', -1, '=', -1, 'list_comprehension', '[', -1, 'call', 'read_cocktail', -1, 'argument_list', '(', -1, 'line', -1, ')', -1, -1, -1, 'for_in_clause', 'for', -1, 'line', -1, 'in', -1, 'call', 'map', -1, 'argument_list', '(', -1, 'attribute', 'str', -1, '.', -1, 'strip', -1, -1, ',', -1, 'call', 'open', -1, 'argument_list', '(', -1, 'file', -1, ',', -1, 'string', '"', -1, '"', -1, -1, ',', -1, 'keyword_argument', 'encoding', -1, '=', -1, 'string', '"', -1, '"', -1, -1, -1, ')', -1, -1, -1, ')', -1, -1, -1, -1, ']', -1, -1, -1), ('expression_statement', 'assignment', 'call', 'argument_list', 'string'))
    True
    >>> subtree_matches(('expression_statement', 'assignment', 'lijst', -1, '=', -1, 'list_comprehension', '[', -1, 'call', 'read_cocktail', -1, 'argument_list', '(', -1, 'line', -1, ')', -1, -1, -1, 'for_in_clause', 'for', -1, 'line', -1, 'in', -1, 'call', 'map', -1, 'argument_list', '(', -1, 'attribute', 'str', -1, '.', -1, 'strip', -1, -1, ',', -1, 'call', 'open', -1, 'argument_list', '(', -1, 'file', -1, ',', -1, 'string', '"', -1, '"', -1, -1, ',', -1, 'keyword_argument', 'encoding', -1, '=', -1, 'string', '"', -1, '"', -1, -1, -1, ')', -1, -1, -1, ')', -1, -1, -1, -1, ']', -1, -1, -1), ('expression_statement', 'assignment', 'call', 'argument_list', '(', -1, ','))
    True
    >>> subtree_matches(('expression_statement', 'assignment', 'uitgang', -1, '=', -1, 'list', '[', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ',', -1, 'call', 'list', -1, 'argument_list', '(', -1, 'call', 'range', -1, 'argument_list', '(', -1, 'integer', -1, ',', -1, 'integer', -1, ')', -1, -1, -1, ')', -1, -1, -1, ']', -1, -1, -1), ('call', 'range', -1, '(', -1, '('))
    False
    """
    if not set(pattern).issubset(subtree):
        return False

    pattern_length = len(pattern)
    history = []

    result = find_in_subtree(subtree, pattern, pattern_length, history, 0, 0, [])
    while not result and history:
        to_explore, to_explore_subtree = history.pop()
        while not result and to_explore:
            start, depth, depth_stack, p_i = to_explore.pop()
            new_subtree = to_explore_subtree[start:]
            # Subtree needs to contain all items from pattern
            if pattern_length - p_i <= len(new_subtree) and sequence_fully_contains_other_sequence(new_subtree, pattern[p_i:]):
                result = find_in_subtree(new_subtree, pattern, pattern_length, history, p_i, depth, depth_stack)

    return result
