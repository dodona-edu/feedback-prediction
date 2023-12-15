from typing import TypedDict, List, Tuple, Collection


class Tree(TypedDict):
    name: str
    lines: List[int]
    children: List['Tree']


class FilteredTree(TypedDict):
    name: str
    children: List['FilteredTree']


HorizontalTree = Collection[str | int]

# TODO mss naar 3.12 updaten zodat je type keyword kan gebruiken (test eerst of 3.12 trager is dan 3.10)
Annotation = Tuple[str, int]

# A combination of a parse tree with a list of (message, line) corresponding to feedback provided
FeedbackTree = Tuple[Tree, List[Annotation]]
