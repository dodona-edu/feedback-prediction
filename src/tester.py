import collections
import datetime
from glob import glob
from typing import List, Dict, Counter, Tuple

from pqdm.processes import pqdm
from tqdm import tqdm

from src.analyze import Analyzer, FeedbackAnalyzer
from src.custom_types import FeedbackTree
from src.feedback_model import FeedbackModel
from src.constants import ROOT_DIR


def test_one_file(messages: FeedbackTree, model: FeedbackModel, n=5) -> Tuple[Counter, List[Counter]]:
    total = collections.Counter()
    first_n = [collections.Counter() for _ in range(n)]
    tree = messages[0]

    for m, line in messages[1]:
        messages_sorted = model.predict_most_likely_messages(tree, line)

        if messages_sorted is not None:
            total[m] += 1
            if m in messages_sorted:
                i = messages_sorted.index(m)
                if i < n:
                    first_n[i][m] += 1

    return total, first_n


def test_all_files(test: Dict[str, FeedbackTree], model: FeedbackModel, n=5, n_procs=8) -> Tuple[Counter, List[Counter], Counter]:
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
    for file_total, counters in results:
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

    return total, first_n, total_first_n


def main(analyzer: Analyzer, test_file=None):
    training, test = analyzer.create_train_test_set()

    model = FeedbackModel()
    model.train(training)

    start = datetime.datetime.now()

    print("Testing...")
    if test_file is None:
        test_all_files(test, model, n=5)
    else:
        test = test[test_file]
        test_one_file(test, model)

    print(f"Total testing time: {datetime.datetime.now() - start}")


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
