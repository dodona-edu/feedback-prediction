"""
Module containing treeminer implementation.
"""
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

class MinerAlgorithm:
    """
    Abstract class to keep the generic parts of treeminer analyses the same
    """
    def __init__(self, database, support=0.9):
        """
        database: list of trees
        support: minimum frequency of patterns in final solution
        """
        self.database = list(map(lambda t: list(to_string_encoding(t)), database))
        self.frequent_patterns = set()
        self.tree_count = len(database)
        atom_occurences = {}
        for tree in self.database:
            for elem in set(tree):
                if elem != -1:
                    if elem not in atom_occurences:
                        atom_occurences[elem] = 0
                    atom_occurences[elem] += 1
        self.f_1 = sorted(k for k, v in atom_occurences.items() if v / self.tree_count >= support)
        self.label2id = {el: i for i, el in enumerate(self.f_1)}
        self.support = support

    def enumerate_frequent_subtrees(self, prefix, clazz):
        """
        Recursively enumerate frequent subtrees until no more subtrees of larger size can be added
        """
        self.frequent_patterns.add(prefix)
        for x, i, scope_list_x in clazz:
            subclass = []
            for y, j, scope_list_y in clazz:
                if i == j:
                    scope_list = self.in_scope_test(x, i, scope_list_x, y, j, scope_list_y)
                    if len(set(elem[0] for elem in scope_list)) / self.tree_count >= self.support:
                        subclass.append((y, j + 1, scope_list))
                scope_list = self.out_scope_test(x, i, scope_list_x, y, j, scope_list_y)
                if len(set(elem[0] for elem in scope_list)) / self.tree_count >= self.support:
                    subclass.append((y, j, scope_list))
            self.enumerate_frequent_subtrees(prefix + (-1,) * (len(prefix) - (2 * prefix.count(-1)) - i - 1) + (self.f_1[x],), subclass)


class Treeminerd(MinerAlgorithm):
    """
    Class representing a treeminerd analysis
    """

    def in_scope_test(self, x, i, scope_list_x, y, j, scope_list_y):
        in_scope_dict = defaultdict(list)
        for elem1 in scope_list_x:
            for elem2 in scope_list_y:
                if elem1 != elem2 and elem1[0] == elem2[0] and elem1[1][-1][0] <= elem2[1][-1][0] and elem1[1][-1][1] >= elem2[1][-1][1]:
                    to_add = in_scope_dict[elem1[0]]
                    should_add = True
                    for k in range(len(to_add) - 1, -1, -1):
                        if to_add[k][1] == elem2[1]:
                            if to_add[k][0][0] <= elem1[1][0] and to_add[k][0][1] >= elem1[1][1]:
                                del to_add[k]
                            elif elem1[1][0] <= to_add[k][0][0] and elem1[1][1] >= to_add[k][0][1]:
                                should_add = False
                    if should_add:
                        path = [e for e in elem1[1] if e[1] >= elem2[1][-1][1]] + [e for e in elem2[1] if e[1] >= elem2[1][-1][1]]
                        to_add.append(sorted(set(path)))
        scope_list = []
        for tid in in_scope_dict:
            for scope_vector in in_scope_dict[tid]:
                if ((tid, scope_vector)) not in scope_list:
                    scope_list.append((tid, scope_vector))
        return scope_list

    def out_scope_test(self, x, i, scope_list_x, y, j, scope_list_y):
        scope_list = []
        for elem1 in scope_list_x:
            for elem2 in scope_list_y:
                if elem1 != elem2 and elem1[0] == elem2[0]:
                    to_be_added = None
                    if j + 1 < len(elem1[1]) and elem1[1][j][0] <= elem2[1][-1][0] and elem1[1][j][1] >= elem2[1][-1][1] and elem1[1][j + 1][1] < elem2[1][-1][0]:
                        to_be_added = (elem1[0], elem1[1][: j + 1] + [elem2[1][-1]])
                    elif j < len(elem1[1]) and elem1[1][j][1] < elem2[1][-1][0] and elem1[1][j][0] <= elem2[1][j][0] and elem1[1][j][1] >= elem2[1][j][1]:
                        to_be_added = (elem1[0], elem2[1][: j + 1] + [elem2[1][-1]])
                    if to_be_added is not None:
                        should_add = True
                        for k in range(len(scope_list) - 1, -1, -1):
                            if scope_list[k][0] == to_be_added[0] and scope_list[k][1][-1] == to_be_added[1][-1] and scope_list[k][1][-2][0] > to_be_added[1][-2][0]:
                                should_add = False
                            if scope_list[k][0] == to_be_added[0] and scope_list[k][1][-1] == to_be_added[1][-1] and scope_list[k][1][-2][0] < to_be_added[1][-2][0]:
                                del scope_list[k]
                        if should_add and to_be_added not in scope_list:
                            scope_list.append(to_be_added)
        return scope_list


    def get_patterns(self):
        """
        Calculate f_2 and start recursive search for f_n
        """
        f_2_support = [[set() for _ in range(len(self.f_1))] for _ in range(len(self.f_1))]
        scope_list_f_1 = {}
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
                        if tree[j] != -1:
                            node_count += 1
                            depth_stack += 1
                            if tree[j] in self.f_1:
                                f_2_support[self.label2id[elem1]][self.label2id[tree[j]]].add(tid)
                        else:
                            depth_stack -= 1
                        j += 1
                    if self.f_1.index(elem1) not in scope_list_f_1:
                        scope_list_f_1[self.label2id[elem1]] = []
                    scope_list_f_1[self.label2id[elem1]].append((tid, (orig_node_count, node_count)))
                    node_count = orig_node_count
        for x in self.f_1:
            subclass = []
            for y in self.f_1:
                if len(f_2_support[self.label2id[x]][self.label2id[y]]) / self.tree_count >= self.support:
                    scope_dict = defaultdict(list)
                    for elem1 in scope_list_f_1[self.label2id[x]]:
                        for elem2 in scope_list_f_1[self.label2id[y]]:
                            if elem1 != elem2 and elem1[0] == elem2[0] and elem1[1][0] <= elem2[1][0] and elem1[1][1] >= elem2[1][1]:
                                to_add = scope_dict[elem1[0]]
                                should_add = True
                                for i in range(len(to_add) - 1, -1, -1):
                                    if to_add[i][1] == elem2[1]:
                                        if to_add[i][0][0] <= elem1[1][0] and to_add[i][0][1] >= elem1[1][1]:
                                            del to_add[i]
                                        elif elem1[1][0] <= to_add[i][0][0] and elem1[1][1] >= to_add[i][0][1]:
                                            should_add = False
                                if should_add:
                                    to_add.append([elem1[1], elem2[1]])
                    scope_list = []
                    for tid in scope_dict:
                        for scope_vector in scope_dict[tid]:
                            scope_list.append((tid, scope_vector))
                    subclass.append((self.label2id[y], 0, scope_list))
            self.enumerate_frequent_subtrees((x,), subclass)
        return self.frequent_patterns


class Treeminer(MinerAlgorithm):
    """
    Class representing a treeminer analysis
    """

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
        f_2_support = [
            [set() for _ in range(len(self.f_1))] for _ in range(len(self.f_1))
        ]
        scope_list_f_1 = {}
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
                        if tree[j] != -1:
                            node_count += 1
                            depth_stack += 1
                            if tree[j] in self.f_1:
                                f_2_support[self.label2id[elem1]][
                                    self.label2id[tree[j]]
                                ].add(tid)
                        else:
                            depth_stack -= 1
                        j += 1
                    if self.f_1.index(elem1) not in scope_list_f_1:
                        scope_list_f_1[self.label2id[elem1]] = []
                    scope_list_f_1[self.label2id[elem1]].append(
                        (tid, (orig_node_count, node_count))
                    )
                    node_count = orig_node_count
        f_2 = {}
        for x in self.f_1:
            f_2[(x,)] = []
            for y in self.f_1:
                if (
                    len(f_2_support[self.label2id[x]][self.label2id[y]])
                    / self.tree_count
                    >= self.support
                ):
                    scope_list = []
                    for elem1 in scope_list_f_1[self.label2id[x]]:
                        for elem2 in scope_list_f_1[self.label2id[y]]:
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

    def analyze_trees(trees):
        td_patterns = Treeminerd(trees).get_patterns()
        print(td_patterns)
        t_patterns = Treeminer(trees).get_patterns()
        print(t_patterns)
        print()
        assert (
            t_patterns == td_patterns
        ), f"difference in patterns for {trees}: {td_patterns - t_patterns} {t_patterns - td_patterns} {len(t_patterns)} {len(td_patterns)}"

    analyze_trees(
        [
            {
                "name": "1",
                "children": [
                    {"name": "3", "children": [{"name": "2", "children": []}]},
                    {"name": "3", "children": [{"name": "1", "children": []}]},
                ],
            },
            {
                "name": "1",
                "children": [
                    {"name": "3", "children": [{"name": "2", "children": []}]},
                    {"name": "3", "children": [{"name": "1", "children": []}]},
                ],
            },
        ]
    )
    analyze_trees(
        [
            {
                "name": "1",
                "children": [
                    {"name": "3", "children": []},
                    {
                        "name": "3",
                        "children": [
                            {"name": "1", "children": []},
                            {"name": "2", "children": []},
                        ],
                    },
                ],
            },
            {
                "name": "1",
                "children": [
                    {"name": "3", "children": [{"name": "2", "children": []}]},
                    {"name": "3", "children": [{"name": "1", "children": []}]},
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
                        "children": [{"name": "3", "children": []}],
                    },
                    {
                        "name": "2",
                        "children": [
                            {
                                "name": "2",
                                "children": [{"name": "3", "children": []}],
                            }
                        ],
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
                    {
                        "name": "3",
                        "children": [
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                        ],
                    },
                    {
                        "name": "3",
                        "children": [
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                        ],
                    },
                ],
            },
            {
                "name": "2",
                "children": [
                    {
                        "name": "1",
                        "children": [
                            {"name": "5", "children": []},
                            {"name": "2", "children": []},
                            {"name": "4", "children": [{"name": "4", "children": []}]},
                        ],
                    },
                    {"name": "2", "children": []},
                    {"name": "3", "children": []},
                ],
            },
            {
                "name": "1",
                "children": [
                    {"name": "3", "children": [{"name": "2", "children": []}]},
                    {
                        "name": "5",
                        "children": [
                            {
                                "name": "1",
                                "children": [
                                    {"name": "2", "children": []},
                                    {
                                        "name": "4",
                                        "children": [{"name": "4", "children": []}],
                                    },
                                ],
                            }
                        ],
                    },
                ],
            },
        ]
    )
