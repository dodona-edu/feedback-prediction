from typing import TypedDict, List, Tuple, Sequence


class Tree(TypedDict):
    name: str
    children: List['Tree']


type HorizontalTree = Sequence[str | int]

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
