def to_string_encoding(tree):
    yield tree['name']
    for child in tree['children']:
        for elem in to_string_encoding(child):
            yield elem
        yield -1

def treeminer(database, support=0.5):
    """
    database: list of trees
    support: minimum frequency of patterns in final solution
    """
    database = list(map(lambda t: list(to_string_encoding(t)), database))
    frequent_patterns = set()
    tree_count = len(database)
    f_1 = {}
    for tree in database:
        for elem in set(tree):
            if elem != -1:
                if elem not in f_1:
                    f_1[elem] = 0
                f_1[elem] += 1
    f_1 = [ k for k, v in f_1.items() if v / tree_count >= support ]
    f_1.sort()
    for p in f_1:
        frequent_patterns.add((p,))
    label2id = {el: i for i, el in enumerate(f_1)}

    # Helper method using some "globals" defined above
    def enumerate_frequent_subtrees(prefix, clazz):
        frequent_patterns.add(prefix)
        for x, i, scope_list_x in clazz:
            subclass = []
            for y, j, scope_list_y in clazz:
                if i == j:
                    scope_list = []
                    for elem1 in scope_list_x:
                        for elem2 in scope_list_y:
                            if elem1 != elem2 and elem1[0] == elem2[0] and elem1[1] == elem2[1] and elem1[2][0] <= elem2[2][0] and elem1[2][1] >= elem2[2][1]:
                                scope_list.append((elem1[0], elem1[1] + [elem1[2][1]], elem2[2]))
                    if len(set(elem[0] for elem in scope_list)) / tree_count >= support:
                        subclass.append((y, j + 1, scope_list))
                scope_list = []
                for elem1 in scope_list_x:
                    for elem2 in scope_list_y:
                        if elem1 != elem2 and elem1[0] == elem2[0] and elem1[1] == elem2[1] and elem1[2][1] < elem2[2][0]:
                            scope_list.append((elem1[0], elem1[1] + [elem1[2][1]], elem2[2]))
                if len(set(elem[0] for elem in scope_list)) / tree_count >= support:
                    subclass.append((y, j, scope_list))
            enumerate_frequent_subtrees(prefix + (-1,) * (len(prefix) - prefix.count(-1) - i - 1) + (f_1[x],), subclass)


    f_2_support = [[set() for _ in range(len(f_1))] for _ in range(len(f_1))]
    scope_list_f_1 = {}
    for tid, tree in enumerate(database):
        node_count = -1
        for i, elem1 in enumerate(tree[:-1]):
            if elem1 != -1:
                node_count += 1
            if elem1 in f_1:
                orig_node_count = node_count
                depth_stack = 1
                j = i + 1
                while depth_stack > 0 and j < len(tree):
                    if tree[j] != -1:
                        node_count += 1
                        depth_stack += 1
                        if tree[j] in f_1:
                            f_2_support[label2id[elem1]][label2id[tree[j]]].add(tid)
                    else:
                        depth_stack -= 1
                    j += 1
                if f_1.index(elem1) not in scope_list_f_1:
                    scope_list_f_1[label2id[elem1]] = []
                scope_list_f_1[label2id[elem1]].append((tid, (orig_node_count, node_count)))
                node_count = orig_node_count
    f_2 = {}
    for x in f_1:
        f_2[(x,)] = []
        for y in f_1:
            if len(f_2_support[label2id[x]][label2id[y]]) / tree_count >= support:
                scope_list = []
                for elem1 in scope_list_f_1[label2id[x]]:
                    for elem2 in scope_list_f_1[label2id[y]]:
                        if elem1 != elem2 and elem1[0] == elem2[0] and elem1[1][0] <= elem2[1][0] and elem1[1][1] >= elem2[1][1]:
                            scope_list.append((elem1[0], [elem1[1][0]], elem2[1]))
                f_2[(x,)].append((label2id[y], 0, scope_list))
    for prefix, clazz in f_2.items():
        enumerate_frequent_subtrees(prefix, clazz)

    return frequent_patterns


if __name__ == '__main__':
    print(treeminer([
        {'name': '1', 'children': [
            {'name': '2', 'children': []},
            {'name': '3', 'children': [
                {'name': '4', 'children': [
                    {'name': '4', 'children': []}
                ]}
            ]}
        ]},
        {'name': '2', 'children': [
            {'name': '1', 'children': [
                {'name': '5', 'children': []},
                {'name': '2', 'children': []},
                {'name': '4', 'children': [
                    {'name': '4', 'children': []}
                ]}
            ]},
            {'name': '2', 'children': []},
            {'name': '3', 'children': []}
        ]},
        {'name': '1', 'children': [
            {'name': '3', 'children': [
                {'name': '2', 'children': []}
            ]},
            {'name': '5', 'children': [
                {'name': '1', 'children': [
                    {'name': '2', 'children': []},
                    {'name': '4', 'children': [
                        {'name': '4', 'children': []}
                    ]}
                ]}
            ]}
        ]}
    ]))
