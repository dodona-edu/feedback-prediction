import datetime
from collections import Counter

from pqdm.processes import pqdm

from analyze import subtree_on_line, analyze
from treeminer import numbers_to_string, Treeminerd, to_string_encoding
from util import list_fully_contains_other_list


def subtree_matches_2_fix(subtree, pattern):
    """
    >>> subtree_matches_2_fix((0, 1, 3, 7, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2_fix((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 2, -1, 1, -1)) # Fig 2
    False
    >>> subtree_matches_2_fix((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 1, -1), (1, 2, -1, 1, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2_fix((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_2_fix((0, 1, 3, 1, -1, 4, -1, -1, 4, -1, -1, 2, -1), (1, 1, -1, 2, -1)) # Fig 2 modified
    False
    >>> subtree_matches_2_fix((0, 1, 3, 1, -1, 2, -1, -1, 2, -1, -1, 2, -1), (0, 1, -1, 2, -1, 2, -1, 2, -1)) # Fig 2
    True
    >>> subtree_matches_2_fix((0, 1, 3, 1, 2, -1, -1, 4, -1, -1, 4, -1, -1, 4, -1), (1, 1, -1, 2, -1))
    False
    >>> subtree_matches_2_fix((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1))
    True
    >>> subtree_matches_2_fix((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, 2, -1, -1))
    True
    >>> subtree_matches_2_fix((1, 1, 2, -1, -1, 1, 2, -1, -1), (1, 1, -1, 1, -1, 2, -1))
    False
    >>> subtree_matches_2_fix((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (0, 1, 2, -1, -1))
    False
    >>> subtree_matches_2_fix((0, 3, 4, -1, 5, -1, 6, 7, -1, 8, -1, 2, -1, -1, 9, -1, 1, 10, 11, 12, -1, -1, -1, -1, -1), (13,))
    False
    >>> subtree_matches_2_fix((1, 2, -1, 3, 4, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T0
    True
    >>> subtree_matches_2_fix((2, 1, 2, -1, 4, -1, -1, 2, -1, 3, -1), (1, 2, -1, 4, -1)) # Fig 5 T1
    True
    >>> subtree_matches_2_fix((1, 3, 2, -1, -1, 5, 1, 2, -1, 3, 4, -1, -1, -1, -1), (1, 2, -1, 4, -1)) # Fig 5 T2
    True
    >>> subtree_matches_2_fix((0, 1, 2, 3, -1, -1, 2, -1, 4, -1, -1, 1, 2, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    True
    >>> subtree_matches_2_fix((0, 1, 2, 3, -1, -1, 2, -1, 5, -1, -1, 1, 4, -1, -1), (0, 1, 3, -1, 4, -1, -1))
    False
    """
    if not set(pattern).issubset(subtree):
        return False

    pattern_length = len(pattern)

    start = 0
    p_i = 0
    pattern_depth = 0
    depth = 0
    depth_stack = []
    history = []

    def find_in_subtree():
        nonlocal start, p_i, pattern_depth, depth, depth_stack

        for i, item in enumerate(subtree[start:]):
            if item == -1:
                if depth_stack and depth - 1 == depth_stack[-1]:
                    last_depth = depth_stack.pop()
                    if pattern[p_i] != -1 and (last_depth < pattern_depth or not depth_stack):
                        p_i = 0
                    if pattern[p_i] == -1:
                        pattern_depth -= 1
                        p_i += 1
                depth -= 1
            else:
                if pattern[p_i] == item:
                    history.append((start + i + 1, depth + 1, depth_stack[:], pattern_depth, p_i))
                    depth_stack.append(depth)
                    pattern_depth += 1
                    p_i += 1

                depth += 1

            if p_i == pattern_length:
                return True

        return False

    result = find_in_subtree()
    while not result and history:
        start, depth, depth_stack, pattern_depth, p_i = history.pop()
        # Subtree needs to contain all items from pattern
        if len(pattern) - p_i <= len(subtree) - start and list_fully_contains_other_list(subtree[start:], pattern[p_i:]):
            result = find_in_subtree()

    return result


def most_likely_messages(pattern_collection, pattern_scores, subtree):
    def match_percentage(m):
        matches = list(filter(lambda pattern: subtree_matches_2_fix(subtree, pattern), pattern_collection[m]))
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