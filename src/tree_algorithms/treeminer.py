"""
Module containing treeminer implementation.
"""
import datetime
import multiprocessing
from collections import defaultdict
from typing import List, Set, Dict, Tuple

from src.custom_types import HorizontalTree, ScopeElement, Scope
from src.util import to_string_encoding


def mine_patterns(subtrees: List[HorizontalTree]) -> Set[HorizontalTree]:
    miner = Treeminerd(subtrees, support=0.8)
    if miner.early_stopping:
        p = multiprocessing.Process(target=miner.get_patterns)
        p.start()
        p.join(30)
        if p.is_alive():
            p.terminate()
        patterns = set(miner.result)
    else:
        patterns = miner.get_patterns()

    return patterns


class MinerAlgorithm:
    """
    Abstract class to keep the generic parts of treeminer analyses the same
    """

    def __init__(self, database: List[HorizontalTree], support=0.9):
        """
        database: list of trees
        support: minimum frequency of patterns in final solution
        """
        self.database: List[HorizontalTree] = database
        self.frequent_patterns: Set[HorizontalTree] = set()

        # if there is a small amount of trees, and the trees are reasonably large, we may need to stop execution early
        self.early_stopping = len(self.database) < 6 and len(max(self.database, key=len)) > 60
        if self.early_stopping:
            manager = multiprocessing.Manager()
            self.result = manager.list()

        self.tree_count = len(database)
        atom_occurences = defaultdict(int)
        for tree in self.database:
            for elem in set(tree):
                if elem != -1:
                    atom_occurences[elem] += 1
        self.f_1 = sorted(
            k for k, v in atom_occurences.items() if v / self.tree_count >= support
        )
        self.label2id = {el: i for i, el in enumerate(self.f_1)}
        self.support = support

        # f_2_support[i][j] = a set of tree_ids that contain the embedded pattern [i, j, -1]
        self.f_2_support: List[List[Set[int]]] = [
            [set() for _ in range(len(self.f_1))] for _ in range(len(self.f_1))
        ]

        # f_1_scope_list[i] = scope list for element i, containing (tree_id, scope) pairs
        self.scope_lists_f_1: Dict[int, List[ScopeElement]] = {}

        self.calculate_f_2_support_and_f_1_scope_list()

    def calculate_f_2_support_and_f_1_scope_list(self):
        """
        f_2_support[i][j] = a set of tree_ids that contain the embedded pattern [i, j, -1]
        f_1_scope_list[i] = scope list for element i, containing (tree_id, scope) pairs
        """
        for tid, tree in enumerate(self.database):
            node_count = -1
            for i, elem1 in enumerate(tree[:-1]):
                if elem1 != -1:
                    node_count += 1
                if elem1 in self.f_1:
                    orig_node_count = node_count
                    depth_stack = 1
                    j = i + 1
                    while depth_stack > 0 and j < len(tree):
                        elem2 = tree[j]
                        if elem2 != -1:
                            node_count += 1
                            depth_stack += 1
                            if elem2 in self.f_1:
                                self.f_2_support[self.label2id[elem1]][self.label2id[elem2]].add(tid)
                        else:
                            depth_stack -= 1
                        j += 1
                    if self.label2id[elem1] not in self.scope_lists_f_1:
                        self.scope_lists_f_1[self.label2id[elem1]] = []
                    self.scope_lists_f_1[self.label2id[elem1]].append({
                        "tree_id": tid,
                        "scope": (orig_node_count, node_count)
                    })
                    node_count = orig_node_count


class Treeminerd(MinerAlgorithm):
    """
    Class representing a treeminerd analysis
    """

    def enumerate_frequent_subtrees(self, prefix: Tuple[int,...], clazz: list[tuple[int, int, dict[int, list[list[Scope]]]]]):
        """
        Recursively enumerate frequent subtrees until no more subtrees of larger size can be added
        """
        if self.early_stopping:
            self.result.append(prefix)
        else:
            self.frequent_patterns.add(prefix)
        prefix_size = len(prefix)
        for x, i, scope_lists_x in clazz:
            subclass = []
            for y, j, scope_lists_y in clazz:
                if i == j:
                    scope_list = self.in_scope_test(
                        x, i, scope_lists_x, y, j, scope_lists_y
                    )
                    if len(scope_list) / self.tree_count >= self.support:
                        subclass.append((y, j + 1, scope_list))
                scope_list = self.out_scope_test(x, i, scope_lists_x, y, j, scope_lists_y)
                if len(scope_list) / self.tree_count >= self.support:
                    subclass.append((y, j, scope_list))
            self.enumerate_frequent_subtrees(
                prefix
                + (-1,) * (prefix_size - (2 * prefix.count(-1)) - i - 1)
                + (self.f_1[x],),
                subclass,
            )

    def in_scope_test(self, x, i, scope_list_x, y, j, scope_list_y):
        in_scope_dict = defaultdict(list)
        for tid in scope_list_x:
            for elem1 in scope_list_x[tid]:
                for elem2 in scope_list_y[tid]:
                    if (
                        elem1 != elem2
                        and elem1[-1] != elem2[-1]
                        and elem1[-1][0] <= elem2[-1][0]
                        and elem1[-1][1] >= elem2[-1][1]
                    ):
                        should_add = True
                        for elem3 in scope_list_x[tid]:
                            if (
                                elem1 != elem3
                                and elem2 != elem3
                                and elem1[-1][0] <= elem3[-1][0] <= elem2[-1][0]
                                and elem1[-1][1] >= elem3[-1][1] >= elem2[-1][1]
                            ):
                                should_add = False
                        if should_add:
                            path = [e for e in elem1 if e[1] >= elem2[-1][1]] + [
                                e for e in elem2 if e[1] >= elem2[-1][1]
                            ]
                            in_scope_dict[tid].append(sorted(set(path)))
        return in_scope_dict

    def out_scope_test(self, x, i, scope_list_x, y, j, scope_list_y):
        scope_dict = defaultdict(list)
        for tid in scope_list_x:
            for elem1 in scope_list_x[tid]:
                elem1_size = len(elem1)
                for elem2 in scope_list_y[tid]:
                    if elem1 != elem2:
                        to_be_added = None
                        if (
                            j + 1 < elem1_size
                            and elem1[j][0] <= elem2[-1][0]
                            and elem1[j][1] >= elem2[-1][1]
                            and elem1[j + 1][1] < elem2[-1][0]
                        ):
                            to_be_added = elem1[: j + 1] + [elem2[-1]]
                        elif (
                            j < elem1_size
                            and elem1[j][1] < elem2[-1][0]
                            and elem1[j][0] <= elem2[j][0]
                            and elem1[j][1] >= elem2[j][1]
                        ):
                            to_be_added = elem2[: j + 1] + [elem2[-1]]
                        if to_be_added is not None:
                            scope_list = scope_dict[tid]
                            should_add = True
                            for k in range(len(scope_list) - 1, -1, -1):
                                if (
                                    scope_list[k][-1] == to_be_added[-1]
                                    and scope_list[k][-2][0] > to_be_added[-2][0]
                                ):
                                    should_add = False
                                if (
                                    scope_list[k][-1] == to_be_added[-1]
                                    and scope_list[k][-2][0] < to_be_added[-2][0]
                                ):
                                    del scope_list[k]
                            if should_add and to_be_added not in scope_list:
                                scope_list.append(to_be_added)
        return scope_dict

    def get_patterns(self) -> Set[HorizontalTree]:
        """
        Calculate f_2 and start recursive search for f_n
        """
        f_1_ids = [self.label2id[el] for el in self.f_1]
        for x, x_id in zip(self.f_1, f_1_ids):
            subclass = []
            for y_id in f_1_ids:
                if (
                    len(self.f_2_support[x_id][y_id])
                    / self.tree_count
                    >= self.support
                ):  # If element x with element y attached (tree [x, y, -1]) is frequent
                    scope_dict: Dict[int, List[List[Scope]]] = defaultdict(list)
                    for elem1 in self.scope_lists_f_1[x_id]:
                        for elem2 in self.scope_lists_f_1[y_id]:
                            scope_x = elem1["scope"]
                            scope_y = elem2["scope"]
                            if (
                                    elem1 != elem2
                                    and elem1["tree_id"] == elem2["tree_id"]
                                    and scope_x[0] <= scope_y[0]
                                    and scope_x[1] >= scope_y[1]
                            ):
                                scope_dict[elem1["tree_id"]].append([scope_x, scope_y])
                    subclass.append((y_id, 0, scope_dict))  # Subclass = (label_id, position, scope_dict)
            self.enumerate_frequent_subtrees((x,), subclass)
        return self.frequent_patterns


class Treeminer(MinerAlgorithm):
    """
    Class representing a treeminer analysis
    """

    def enumerate_frequent_subtrees(self, prefix, clazz):
        """
        Recursively enumerate frequent subtrees until no more subtrees of larger size can be added
        """
        self.frequent_patterns.add(prefix)
        for x, i, scope_list_x in clazz:
            subclass = []
            for y, j, scope_list_y in clazz:
                if i == j:
                    scope_list = self.in_scope_test(
                        x, i, scope_list_x, y, j, scope_list_y
                    )
                    if (
                        len(set(elem[0] for elem in scope_list)) / self.tree_count
                        >= self.support
                    ):
                        subclass.append((y, j + 1, scope_list))
                scope_list = self.out_scope_test(x, i, scope_list_x, y, j, scope_list_y)
                if (
                    len(set(elem[0] for elem in scope_list)) / self.tree_count
                    >= self.support
                ):
                    subclass.append((y, j, scope_list))
            self.enumerate_frequent_subtrees(
                prefix
                + (-1,) * (len(prefix) - (2 * prefix.count(-1)) - i - 1)
                + (self.f_1[x],),
                subclass,
            )

    def in_scope_test(self, x, i, scope_list_x, y, j, scope_list_y):
        scope_list = []
        for elem1 in scope_list_x:
            for elem2 in scope_list_y:
                if (
                    elem1 != elem2
                    and elem1[0] == elem2[0]
                    and elem1[1] == elem2[1]
                    and elem1[2][0] <= elem2[2][0]
                    and elem1[2][1] >= elem2[2][1]
                ) and (
                    elem1[0],
                    elem1[1] + [elem1[2][1]],
                    elem2[2],
                ) not in scope_list:
                    scope_list.append((elem1[0], elem1[1] + [elem1[2][1]], elem2[2]))
        return scope_list

    def out_scope_test(self, x, i, scope_list_x, y, j, scope_list_y):
        scope_list = []
        for elem1 in scope_list_x:
            for elem2 in scope_list_y:
                if (
                    elem1 != elem2
                    and elem1[0] == elem2[0]
                    and elem1[1] == elem2[1]
                    and elem1[2][1] < elem2[2][0]
                ) and (
                    elem1[0],
                    elem1[1] + [elem1[2][1]],
                    elem2[2],
                ) not in scope_list:
                    scope_list.append((elem1[0], elem1[1] + [elem1[2][1]], elem2[2]))
        return scope_list

    def get_patterns(self):
        f_2 = {}
        for x in self.f_1:
            f_2[(x,)] = []
            for y in self.f_1:
                if (
                    len(self.f_2_support[self.label2id[x]][self.label2id[y]])
                    / self.tree_count
                    >= self.support
                ):
                    scope_list = []
                    for elem1 in self.scope_list_f_1[self.label2id[x]]:
                        for elem2 in self.scope_list_f_1[self.label2id[y]]:
                            if (
                                elem1 != elem2
                                and elem1[0] == elem2[0]
                                and elem1[1][0] <= elem2[1][0]
                                and elem1[1][1] >= elem2[1][1]
                            ):
                                scope_list.append((elem1[0], [elem1[1][0]], elem2[1]))
                    f_2[(x,)].append((self.label2id[y], 0, scope_list))
        for prefix, clazz in f_2.items():
            self.enumerate_frequent_subtrees(prefix, clazz)

        return self.frequent_patterns


if __name__ == "__main__":
    def timed(func):
        start = datetime.datetime.now()
        res = func()
        print(datetime.datetime.now() - start)
        return res


    def analyze_trees(trees):
        trees = list(map(lambda t: list(to_string_encoding(t)), trees))
        # t_patterns = timed(Treeminer(trees).get_patterns)
        # print(t_patterns)
        td_patterns = timed(Treeminerd(trees).get_patterns)
        print(td_patterns)
        # print()
        # assert (
        #     t_patterns == td_patterns
        # ), f"difference in patterns for {trees}: {td_patterns - t_patterns} {t_patterns - td_patterns} {len(td_patterns)} {len(t_patterns)}"


    # This is the forest of trees from figure 5 of the paper
    analyze_trees(
        [
            {
                "name": "1",
                "children": [
                    {"name": "2", "children": []},
                    {
                        "name": "3",
                        "children": [{"name": "4", "children": []}]
                    }
                ]
            },
            {
                "name": "2",
                "children": [
                    {
                        "name": "1",
                        "children": [
                            {"name": "2", "children": []},
                            {"name": "4", "children": []}
                        ]
                    },
                    {"name": "2", "children": []},
                    {"name": "3", "children": []}
                ]
            },
            {
                "name": "1",
                "children": [
                    {
                        "name": "3",
                        "children": [{"name": "2", "children": []}]
                    },
                    {
                        "name": "5",
                        "children": [
                            {
                                "name": "1",
                                "children": [
                                    {"name": "2", "children": []},
                                    {
                                        "name": "3",
                                        "children": [
                                            {"name": "4", "children": []}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    )

    analyze_trees(
        [
            {
                "name": "1",
                "children": [
                    {
                        "name": "1",
                        "children": [{"name": "2", "children": []}],
                    },
                    {
                        "name": "1",
                        "children": [{"name": "2", "children": []}],
                    },
                ],
            }
        ]
    )

    analyze_trees(
        [
            {
                "name": "1",
                "children": [
                    {"name": "2", "children": []},
                    {
                        "name": "3",
                        "children": [
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                        ],
                    },
                    {"name": "2", "children": []},
                    {
                        "name": "3",
                        "children": [
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                        ],
                    },
                    {"name": "2", "children": []},
                    {
                        "name": "3",
                        "children": [
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                        ],
                    },
                    {"name": "2", "children": []},
                    {
                        "name": "3",
                        "children": [],
                    },
                ],
            },
        ]
    )

    analyze_trees(
        [
            {
                "name": "1",
                "children": [
                    {
                        "name": "2",
                        "children": [
                            {
                                "name": "3",
                                "children": [
                                    {
                                        "name": "4",
                                        "children": [
                                            {
                                                "name": "3",
                                                "children": [
                                                    {
                                                        "name": "5",
                                                        "children": [
                                                            {
                                                                "name": "6",
                                                                "children": [],
                                                            },
                                                            {
                                                                "name": "7",
                                                                "children": [
                                                                    {
                                                                        "name": "8",
                                                                        "children": [
                                                                            {
                                                                                "name": "9",
                                                                                "children": [
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ".",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "10",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                    {
                                                                        "name": "11",
                                                                        "children": [
                                                                            {
                                                                                "name": "(",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "n",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ",",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "13",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ")",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "name": "1",
                "children": [
                    {
                        "name": "2",
                        "children": [
                            {
                                "name": "3",
                                "children": [
                                    {
                                        "name": "4",
                                        "children": [
                                            {
                                                "name": "3",
                                                "children": [
                                                    {
                                                        "name": "5",
                                                        "children": [
                                                            {
                                                                "name": "6",
                                                                "children": [],
                                                            },
                                                            {
                                                                "name": "7",
                                                                "children": [
                                                                    {
                                                                        "name": "8",
                                                                        "children": [
                                                                            {
                                                                                "name": "9",
                                                                                "children": [
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ".",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "10",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                    {
                                                                        "name": "11",
                                                                        "children": [
                                                                            {
                                                                                "name": "(",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "aantal",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ",",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "14",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ")",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "name": "1",
                "children": [
                    {
                        "name": "2",
                        "children": [
                            {
                                "name": "3",
                                "children": [
                                    {
                                        "name": "4",
                                        "children": [
                                            {
                                                "name": "3",
                                                "children": [
                                                    {
                                                        "name": "5",
                                                        "children": [
                                                            {
                                                                "name": "6",
                                                                "children": [],
                                                            },
                                                            {
                                                                "name": "7",
                                                                "children": [
                                                                    {
                                                                        "name": "8",
                                                                        "children": [
                                                                            {
                                                                                "name": "9",
                                                                                "children": [
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ".",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "10",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                    {
                                                                        "name": "11",
                                                                        "children": [
                                                                            {
                                                                                "name": "(",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "15",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ",",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "16",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ")",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "name": "1",
                "children": [
                    {
                        "name": "2",
                        "children": [
                            {
                                "name": "3",
                                "children": [
                                    {
                                        "name": "4",
                                        "children": [
                                            {
                                                "name": "3",
                                                "children": [
                                                    {
                                                        "name": "5",
                                                        "children": [
                                                            {
                                                                "name": "6",
                                                                "children": [],
                                                            },
                                                            {
                                                                "name": "7",
                                                                "children": [
                                                                    {
                                                                        "name": "8",
                                                                        "children": [
                                                                            {
                                                                                "name": "9",
                                                                                "children": [
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": '"',
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ".",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "10",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                    {
                                                                        "name": "11",
                                                                        "children": [
                                                                            {
                                                                                "name": "(",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "15",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ",",
                                                                                "children": [],
                                                                            },
                                                                            {
                                                                                "name": "8",
                                                                                "children": [
                                                                                    {
                                                                                        "name": "12",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": ".",
                                                                                        "children": [],
                                                                                    },
                                                                                    {
                                                                                        "name": "13",
                                                                                        "children": [],
                                                                                    },
                                                                                ],
                                                                            },
                                                                            {
                                                                                "name": ")",
                                                                                "children": [],
                                                                            },
                                                                        ],
                                                                    },
                                                                ],
                                                            },
                                                        ],
                                                    }
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        ]
    )
