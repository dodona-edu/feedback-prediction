"""
Module containing treeminer implementation.
"""
import datetime
from collections import defaultdict


def to_string_encoding(tree):
    """
    Convert a tree into it's "string" encoding. Note that this is actually a
    generator of "atoms", to be collected in a list or other sequential type.
    """
    yield tree["name"]
    for child in tree["children"]:
        for elem in to_string_encoding(child):
            yield elem
        yield -1


def numbers_to_string(numbers):
    result = []
    for item in numbers:
        if item == -1:
            result.append(-1)
        else:
            result.append(str(item))

    return result


class MinerAlgorithm:
    """
    Abstract class to keep the generic parts of treeminer analyses the same
    """

    def __init__(self, database, support=0.9, to_string=True):
        """
        database: list of trees
        support: minimum frequency of patterns in final solution
        """
        if to_string:
            self.database = list(map(lambda t: list(to_string_encoding(t)), database))
        else:
            self.database = database
        self.frequent_patterns = set()
        self.tree_count = len(database)
        atom_occurences = {}
        for tree in self.database:
            for elem in set(tree):
                if elem != -1:
                    if elem not in atom_occurences:
                        atom_occurences[elem] = 0
                    atom_occurences[elem] += 1
        self.f_1 = sorted(
            k for k, v in atom_occurences.items() if v / self.tree_count >= support
        )
        self.label2id = {el: i for i, el in enumerate(self.f_1)}
        self.support = support

        # f_2_support[i][j] = a set of tree_ids that contain the embedded pattern [i, j, -1]
        self.f_2_support = [
            [set() for _ in range(len(self.f_1))] for _ in range(len(self.f_1))
        ]

        # f_1_scope_list[i] = scope list for element i, containing (tree_id, scope) pairs
        self.scope_list_f_1 = {}

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
                    if self.label2id[elem1] not in self.scope_list_f_1:
                        self.scope_list_f_1[self.label2id[elem1]] = []
                    self.scope_list_f_1[self.label2id[elem1]].append(
                        (tid, (orig_node_count, node_count))
                    )
                    node_count = orig_node_count

    def save_id_db_to_file(self):
        """
        # TODO improve method, better way to handle filtered out labels?
        Transform the tree database to it's corresponding list of ids
        """
        label2id = self.label2id
        max_len = 0
        with open("output/id_db", "w") as file:
            for i, tree in enumerate(self.database):
                id_tree = []
                for item in tree:
                    if item != -1:
                        if item not in label2id.keys():
                            new_id = max(label2id.values()) + 1
                            label2id[item] = new_id
                        id_tree.append(str(label2id[item]))
                    else:
                        id_tree.append(str(item))
                if len(id_tree) > max_len:
                    max_len = len(id_tree)
                encoding = ' '.join(id_tree)
                file.write(f"{i} {i} {len(tree)} {encoding}\n")

            print(f"Maxium id-tree length: {max_len}")


class Treeminerd(MinerAlgorithm):
    """
    Class representing a treeminerd analysis
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
                    if len(scope_list) / self.tree_count >= self.support:
                        subclass.append((y, j + 1, scope_list))
                scope_list = self.out_scope_test(x, i, scope_list_x, y, j, scope_list_y)
                if len(scope_list) / self.tree_count >= self.support:
                    subclass.append((y, j, scope_list))
            self.enumerate_frequent_subtrees(
                prefix
                + (-1,) * (len(prefix) - (2 * prefix.count(-1)) - i - 1)
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
                                and elem1[-1][1] >= elem3[-1][1] >= elem3[-1][1]
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
                for elem2 in scope_list_y[tid]:
                    if elem1 != elem2:
                        to_be_added = None
                        if (
                            j + 1 < len(elem1)
                            and elem1[j][0] <= elem2[-1][0]
                            and elem1[j][1] >= elem2[-1][1]
                            and elem1[j + 1][1] < elem2[-1][0]
                        ):
                            to_be_added = elem1[: j + 1] + [elem2[-1]]
                        elif (
                            j < len(elem1)
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

    def get_patterns(self):
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
                    scope_dict = defaultdict(list)
                    for elem1 in self.scope_list_f_1[x_id]:
                        for elem2 in self.scope_list_f_1[y_id]:
                            if (
                                elem1 != elem2
                                and elem1[0] == elem2[0]
                                and elem1[1][0] <= elem2[1][0]
                                and elem1[1][1] >= elem2[1][1]
                            ):
                                scope_dict[elem1[0]].append([elem1[1], elem2[1]])
                    subclass.append((y_id, 0, scope_dict))
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
