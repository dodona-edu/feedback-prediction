import csv
import datetime
import json
from abc import abstractmethod, ABC
import random
from collections import defaultdict
from glob import glob
from io import StringIO
from typing import List, Dict, Tuple

from pylint import lint
from pylint.reporters import JSONReporter
from tqdm import tqdm
from tree_sitter import Language, Parser, Node

from src.custom_types import AnnotatedCode, Annotation, Tree
from src.constants import ROOT_DIR


class Analyzer(ABC):

    def __init__(self):
        PYTHON = Language(f"{ROOT_DIR}/build/languages.so", "python")
        parser = Parser()
        parser.set_language(PYTHON)
        self.parser = parser

        self.not_in_train_test_filter = False  # Whether to filter messages from test files that are not present in any training files
        self.files: List[str] = []

        self.train: Dict[str, AnnotatedCode] = {}
        self.test: Dict[str, AnnotatedCode] = {}

    def map_tree(self, node: Node) -> Tree:
        children = [self.map_tree(child) for child in node.children if child.type != "comment"]
        name = node.text.decode("utf-8") if node.type == "identifier" else node.type
        return {"name": name, "children": children}

    @abstractmethod
    def messages_for_file(self, file: str) -> List[Annotation]:
        pass

    def analyze_file(self, file: str) -> AnnotatedCode:
        with open(file, "rb") as f:
            code = f.readlines()

        return code, self.messages_for_file(file)

    def analyze_files(self) -> Dict[str, AnnotatedCode]:
        """
        Analyze the files by parsing them and adding messages.
        """
        print("Analyzing files")

        result = {}
        for filename in tqdm(self.files):
            result[filename] = self.analyze_file(filename)

        return result

    def _filter_test(self, train: Dict[str, AnnotatedCode], test: Dict[str, AnnotatedCode]) -> Dict[str, AnnotatedCode]:
        train_messages = set()
        for (_, items) in train.values():
            train_messages.update(item[0] for item in items)

        for filename, (tree, messages_lines) in test.items():
            messages_lines = list(filter(lambda x: x[0] in train_messages, messages_lines))
            test[filename] = (tree, messages_lines)

        return test

    def create_train_test_set(self) -> Tuple[Dict[str, AnnotatedCode], Dict[str, AnnotatedCode]]:
        random.seed(314159)
        files = self.files[:]

        random.shuffle(files)

        train = {}
        test = {}

        for filename in files[:len(files) // 2]:
            train[filename] = self.analyze_file(filename)

        for filename in files[len(files) // 2:]:
            test[filename] = self.analyze_file(filename)

        if self.not_in_train_test_filter:
            test = self._filter_test(train, test)

        self.train = train
        self.test = test

        return train, test


class PylintAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.files = glob(f'{ROOT_DIR}/pylint/submissions/*/*/*.py')

    def messages_for_file(self, file: str) -> List[Annotation]:
        pylint_output = StringIO()
        reporter = JSONReporter(pylint_output)
        lint.Run(["--module-naming-style=any", "--disable=C0304", file], reporter=reporter, exit=False)
        lint_result = list(map(lambda m: (m["line"] - 1, f"{m['message-id']}-{m['symbol']}"), json.loads(pylint_output.getvalue())))
        return [(message, line) for (line, message) in lint_result]


class FeedbackAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

        self.annotations_file = f"{ROOT_DIR}/data/annotations.tsv"
        self.date_time_format = "%Y-%m-%d %H:%M:%S.%f"

        self.submission_annotations_map: Dict[str, List[Tuple[str, int, datetime.datetime]]] = defaultdict(list)
        self.load_submission_annotations_map()
        self.set_files(glob(f'{ROOT_DIR}/data/excercises/*/*.py'))

    def load_submission_annotations_map(self) -> None:
        with open(self.annotations_file) as annotations_file:
            rows = csv.reader(annotations_file, delimiter='\t')
            next(rows)
            for row in rows:
                line = row[1]
                submission_id = row[2]
                annotation = row[4]
                creation_time = datetime.datetime.strptime(row[5], self.date_time_format)
                if line != 'NULL':
                    self.submission_annotations_map[submission_id].append((annotation, int(line), creation_time))

    def messages_for_file(self, file: str) -> List[Annotation]:
        submission_id = file.split('/')[-1].split('.')[0]
        return list(map(lambda x: x[0:2], self.submission_annotations_map[submission_id]))

    def set_files(self, files) -> None:
        self.files = [file for file in files if file.split('/')[-1].split('.')[0] in self.submission_annotations_map.keys()]

    def get_sorted_files(self) -> List[str]:
        return sorted(self.files, key=lambda i: self.submission_annotations_map[i.split('/')[-1].split('.')[0]][0][2])
