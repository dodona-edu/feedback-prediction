import collections
import time
from glob import glob
from typing import List, Dict, Counter, Tuple

from pqdm.processes import pqdm
from tqdm import tqdm

from analyze import Analyzer, FeedbackAnalyzer
from custom_types import AnnotatedTree
from feedback_model import FeedbackModel
from constants import ROOT_DIR
from tree_algorithms.subtree_on_line import find_subtree_on_line


def test_one_file(annotated_tree: AnnotatedTree, model: FeedbackModel, n=5) -> Tuple[Counter, List[Counter], List[float]]:
    tree, annotations = annotated_tree

    total = collections.Counter()
    first_n = [collections.Counter() for _ in range(n)]
    times = []
    for m, line in annotations:
        subtree = find_subtree_on_line(tree, line)
        if subtree is not None:
            start = time.perf_counter()

            matching_scores = model.calculate_matching_scores(subtree)
            messages_sorted = sorted(matching_scores.keys(), key=lambda ms: matching_scores[ms], reverse=True)

            end = time.perf_counter()
            times.append(end - start)

            if m in messages_sorted:
                i = messages_sorted.index(m)
                if i < n:
                    first_n[i][m] += 1

        total[m] += 1

    return total, first_n, times


def test_all_files(test: Dict[str, AnnotatedTree], model: FeedbackModel, n=5, n_procs=8) -> Tuple[Counter, List[Counter], Counter, List[float]]:
    if n_procs > 1:
        results = pqdm(map(lambda ms: (ms, model, n), test.values()), test_one_file, n_jobs=8, argument_type='args')
    else:
        results = []
        for value in tqdm(test.values()):
            result = test_one_file(value, model, n)
            results.append(result)

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
    for m in total:
        result = (m, total[m], first_n[0][m] / total[m], total_first_n[m] / total[m])
        print(result)

    sum_total = sum(total.values())
    sum_first = sum(first_n[0].values())
    sum_first_n = sum([sum(f.values()) for f in first_n])
    print(f"\nTotal: {sum_first / sum_total}, {sum_first_n / sum_total}")

    return total, first_n, total_first_n, all_times


def main(analyzer: Analyzer, test_file=None):
    training, test = analyzer.create_train_test_set()

    model = FeedbackModel()
    model.train(training)

    start = time.time()

    print("Testing...")
    if test_file is None:
        test_all_files(test, model, n=5)
    else:
        test = test[test_file]
        test_one_file(test, model)

    print(f"Total testing time: {time.time()- start}")


if __name__ == '__main__':
    # message_analyzer = PylintAnalyzer()
    # main(message_analyzer, test_file='pylint/submissions/exam-group-1-examen-groep-1/Lena Lardinois/centrifuge.py')

    # message_analyzer = FeedbackAnalyzer()
    # exercise_ids = ['505886137', '933265977', '1730686412', '1875043169', '2046492002', '2146239081']
    # for e_id in exercise_ids:
    #     print(f"Results for excercise with ID {e_id}")
    #     message_analyzer.set_files(glob(f'data/excercises/{e_id}/*.py'))
    #     main(message_analyzer)

    message_analyzer = FeedbackAnalyzer()
    message_analyzer.set_files(glob(f'{ROOT_DIR}/data/excercises/505886137/*.py'))
    message_analyzer.not_in_train_test_filter = False
    main(message_analyzer)
