import datetime
from collections import Counter

from pqdm.processes import pqdm

from analyze import subtree_on_line, analyze
from treeminer import numbers_to_string, Treeminerd, to_string_encoding


def subtree_matches_3(subtree, pattern):
    """
    >>> subtree_matches_3((0, 1, 3, 7, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_3((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 2, -1, 1, -1)) # Fig 2
    False
    >>> subtree_matches_3((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 1, -1), (1, 2, -1, 1, -1)) # Fig 2 modified
    False
    >>> subtree_matches_3((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_3((0, 1, 3, 1, -1, 4, -1, -1, 4, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_3((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_3((0, 1, 3, 1, 2, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1), (1, 1, -1, 2, -1))
    False
    >>> subtree_matches_3((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1))
    True
    >>> subtree_matches_3((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2, -1, -1))
    True
    >>> subtree_matches_3((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2, -1))
    False
    >>> subtree_matches_3((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2, -1, -1))
    False
    >>> subtree_matches_3((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches_3((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T0
    True
    >>> subtree_matches_3((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4, -1)) # Fig 5 T1
    True
    >>> subtree_matches_3((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T2
    True
    >>> subtree_matches_3((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches_3((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    """
    if not set(pattern).issubset(subtree):
        return False

    subtree = numbers_to_string(subtree)
    pattern = numbers_to_string(pattern)
    results = Treeminerd([pattern, subtree], 1, False).get_patterns()

    stripped_pattern = pattern[:]
    while stripped_pattern[-1] == -1:
        stripped_pattern = stripped_pattern[:-1]

    return tuple(stripped_pattern) in results


def subtree_matches_2(subtree, pattern):
    """
    >>> subtree_matches_2((0, 1, 3, 7, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 2, -1, 1, -1)) # Fig 2
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 1, -1), (1, 2, -1, 1, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_2((0, 1, 3, 1, -1, 4, -1, -1, 4, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_2((0, 1, 3, 1, 2, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1), (1, 1, -1, 2, -1))
    False
    >>> subtree_matches_2((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1))
    True
    >>> subtree_matches_2((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2, -1, -1))
    True
    >>> subtree_matches_2((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2, -1))
    False
    >>> subtree_matches_2((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2, -1, -1))
    False
    >>> subtree_matches_2((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches_2((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T0
    True
    >>> subtree_matches_2((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4, -1)) # Fig 5 T1
    True
    >>> subtree_matches_2((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T2
    True
    >>> subtree_matches_2((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches_2((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    """
    if not set(pattern).issubset(subtree):
        return False

    pattern_length = len(pattern)
    p_i = 0
    pattern_depth = 0
    depth = 0
    depth_stack = []
    for i, item in enumerate(subtree):
        if item == -1:
            if depth_stack and depth - 1 == depth_stack[-1]:
                last_depth = depth_stack.pop()
                if pattern[p_i] != -1 and (last_depth < pattern_depth or not depth_stack):
                    p_i = 0
            depth -= 1
        else:
            if pattern[p_i] == item and item != -1:
                depth_stack.append(depth)
            depth += 1

        if pattern[p_i] == item:
            if item == -1:
                pattern_depth -= 1
            else:
                pattern_depth += 1
            p_i += 1

        if p_i == pattern_length:
            return True

    return False


def subtree_matches(subtree, pattern):
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
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1))
    True
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2))
    True
    >>> subtree_matches((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2))
    False
    >>> subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2))
    False
    >>> subtree_matches((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4)) # Fig 5 T0
    True
    >>> subtree_matches((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4)) # Fig 5 T1
    True
    >>> subtree_matches((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4)) # Fig 5 T2
    True
    >>> subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    """
    if not set(pattern).issubset(subtree):
        return False

    index = 0
    depth = 0
    depth_stack = [0]
    for atom in pattern:
        if atom == -1:
            while index < len(subtree) and depth > depth_stack[-1]:
                if subtree[index] == -1:
                    depth -= 1
                else:
                    depth += 1
                index += 1
            if index == len(subtree):
                return False
            del depth_stack[-1]
        else:
            while index < len(subtree) and depth > depth_stack[-1] and subtree[index] != atom:
                if subtree[index] == -1:
                    depth -= 1
                else:
                    depth += 1
                index += 1
            if index == len(subtree) or depth < depth_stack[-1]:
                return False
            depth_stack.append(depth)
            index += 1
            depth += 1
    return True


def most_likely_messages(pattern_collection, pattern_scores, subtree):

    def match_percentage(m):
        matches = list(filter(lambda pattern: subtree_matches_2(subtree, pattern), pattern_collection[m]))
        matches_score = sum(pattern_scores[match] for match in matches)
        total = len(pattern_collection[m])
        return matches_score / total, f"{matches_score}/{total}"

    message_matches = [(message, match_percentage(message)) for message in pattern_collection.keys()]
    message_matches.sort(key=lambda x: x[1][0], reverse=True)
    return message_matches


def result_for_file(patterns, messages, pattern_scores, n=5):
    total = Counter()
    first_n = [Counter() for _ in range(n)]
    tree = messages[0]

    for m, line in messages[1]:
        subtree = subtree_on_line(tree, line)
        if subtree is not None:
            subtree_string = list(to_string_encoding(subtree))
            total[m] += 1
            matched = most_likely_messages(patterns, pattern_scores, subtree_string)

            messages_sorted = list(map(lambda x: x[0], matched))
            if m in messages_sorted:
                i = messages_sorted.index(m)
                if i < n:
                    first_n[i][m] += 1

    return total, first_n


def test_all_files(test, patterns, pattern_scores, n=5):
    results = pqdm(map(lambda i: (patterns, i, pattern_scores, n), test.values()), result_for_file, n_jobs=8, argument_type='args')
    if not isinstance(results[0], tuple):
        print(results)

    total = Counter()
    first_n = [Counter() for _ in range(n)]
    for file_total, counters in results:
        total += file_total
        for i, counter in enumerate(counters):
            first_n[i] += counter

    total_first_n = sum(first_n, Counter())
    for m in total:
        result = (m, total[m], first_n[0][m] / total[m], total_first_n[m] / total[m])
        print(result)

    sum_total = sum(total.values())
    sum_first = sum(first_n[0].values())
    sum_first_n = sum([sum(f.values()) for f in first_n])
    print(f"\nTotal: {sum_first / sum_total}, {sum_first_n / sum_total}")

    return total, first_n, total_first_n


def test_one_file(test, patterns, pattern_scores):
    start = datetime.datetime.now()
    result = result_for_file(patterns, test, pattern_scores)
    print(f"Elapsed time: {datetime.datetime.now() - start}")
    print(result)


def main(test_file=None):
    training, test, patterns, pattern_scores = analyze(load_analysis=True, load_patterns=True)

    print("Testing...")
    if test_file is None:
        test_all_files(test, patterns, pattern_scores, n=5)
    else:
        test = test[test_file]
        test_one_file(test, patterns, pattern_scores)


if __name__ == '__main__':
    main()
    # main(test_file='submissions/exam-group-1-examen-groep-1/Lena Lardinois/centrifuge.py')