from typing import TypedDict, List, Tuple, Sequence


class Tree(TypedDict):
    name: str
    lines: List[int]
    children: List['Tree']


class FilteredTree(TypedDict):
    name: str
    children: List['FilteredTree']


type HorizontalTree = Sequence[str | int]

type Annotation = Tuple[str, int]

# A combination of a parse tree with a list of (message, line) corresponding to feedback provided
type FeedbackTree = Tuple[Tree, List[Annotation]]
