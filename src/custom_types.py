from typing import TypedDict, List, Tuple, Sequence, Set


class Tree(TypedDict):
    name: str
    children: List['Tree']


# A horizontal version of a tree, where a string means going down one level and a -1 means going up
type HorizontalTree = Sequence[str | int]

# A combination of patterns and 'identifying' nodes. Identifying means that these nodes might be important/useful and related to the annotation
type PatternCollection = Tuple[Set[HorizontalTree], Set[str]]

# A combination of a feedback comment and a line number on which the feedback was given
type Annotation = Tuple[str, int]

# A combination of a list of lines of code and a list of (message, line) corresponding to feedback provided
type AnnotatedCode = Tuple[List[bytes], List[Annotation]]

type Scope = Tuple[int, int]


class ScopeElement(TypedDict):
    """
    The elements inside a scope list
    """
    tree_id: int
    scope: Scope
