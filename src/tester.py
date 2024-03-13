import collections
import time
from typing import List, Dict, Counter, Tuple

from pqdm.processes import pqdm

from custom_types import AnnotatedTree
from feedback_model import FeedbackModel
from tree_algorithms.subtree_on_line import find_subtree_on_line


def test_one_file(annotated_tree: AnnotatedTree, model: FeedbackModel, n=5) -> Tuple[Counter, List[Counter], List[float]]:
    tree, annotation_instances = annotated_tree

    total = collections.Counter()
    first_n = [collections.Counter() for _ in range(n)]
    times = []
    for a_id, line in annotation_instances:
        subtree = find_subtree_on_line(tree, line)
        if subtree is not None:
            start = time.perf_counter()

            matching_scores = model.calculate_matching_scores(subtree)
            annotation_ids_sorted = sorted(matching_scores.keys(), key=lambda x: matching_scores[x], reverse=True)

            end = time.perf_counter()
            times.append(end - start)

            if a_id in annotation_ids_sorted:
                i = annotation_ids_sorted.index(a_id)
                if i < n:
                    first_n[i][a_id] += 1

        total[a_id] += 1

    return total, first_n, times


def test_all_files(test: Dict[str, AnnotatedTree], model: FeedbackModel, n=5, n_procs=8) -> Tuple[Counter, List[Counter], Counter, List[float]]:
    results = pqdm(map(lambda ms: (ms, model, n), test.values()), test_one_file, n_jobs=n_procs, argument_type='args', exception_behaviour='immediate')

    if not isinstance(results[0], tuple):
        print(results)

    total = collections.Counter()
    first_n = [collections.Counter() for _ in range(n)]
    all_times = []
    for file_total, counters, times in results:
        all_times.extend(times)
        total += file_total
        for i, counter in enumerate(counters):
            first_n[i] += counter

    total_first_n = sum(first_n, collections.Counter())
    for a_id in total:
        result = (a_id, total[a_id], first_n[0][a_id] / total[a_id], total_first_n[a_id] / total[a_id])
        print(result)

    sum_total = sum(total.values())
    sum_first = sum(first_n[0].values())
    sum_first_n = sum([sum(f.values()) for f in first_n])
    print(f"\nTotal: {sum_first / sum_total}, {sum_first_n / sum_total}")

    return total, first_n, total_first_n, all_times
