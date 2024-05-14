import csv
import datetime
import json
import pickle
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

from config import USE_AST, RANDOM_SEED
from custom_types import AnnotatedTree, AnnotationInstance, LineTree
from constants import ROOT_DIR


class Analyzer(ABC):

    def __init__(self):
        PYTHON = Language(f"{ROOT_DIR}/build/languages.so", "python")
        parser = Parser()
        parser.set_language(PYTHON)
        self.parser = parser

        self.files: List[str] = []

        self.train: Dict[str, AnnotatedTree] = {}
        self.test: Dict[str, AnnotatedTree] = {}

        self.id_annotation_map: Dict[int, str] = {}
        self.annotations_per_submission_per_line: Dict[str, Dict[int, Dict[int, str]]] = {}

    def map_tree(self, node: Node) -> LineTree:
        child_nodes = node.named_children if USE_AST else node.children
        children = [self.map_tree(child) for child in child_nodes]
        lines = sorted(set(range(node.start_point[0], node.end_point[0] + 1)))
        if node.type in ["identifier", "string", "integer"]:
            name = node.text.decode("utf-8")
            if node.type == "string":
                name = f"'{name[1:-1]}'"
        elif USE_AST and node.type in ["binary_operator", "boolean_operator", "comparison_operator", "augmented_assignment"]:
            name = node.children[1].type
        elif USE_AST and node.type == "unary_operator":
            name = node.children[0].type
        else:
            name = node.type
        return {"name": name, "lines": lines, "children": children}

    @abstractmethod
    def get_annotation_instances(self, file: str) -> List[AnnotationInstance]:
        pass

    def set_files(self, files: List[str]) -> None:
        self.files = files

    def analyze_file(self, file: str) -> AnnotatedTree:
        with open(file, "rb") as f:
            tree = self.map_tree(self.parser.parse(f.read()).root_node)

        return tree, self.get_annotation_instances(file)

    def analyze_files(self) -> Dict[str, AnnotatedTree]:
        """
        Analyze the files by parsing them and adding messages.
        """
        print("Analyzing files")

        result = {}
        for filename in tqdm(self.files):
            result[filename] = self.analyze_file(filename)

        return result

    def create_train_test_set(self) -> Tuple[Dict[str, AnnotatedTree], Dict[str, AnnotatedTree]]:
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
        files = self.files[:]

        random.shuffle(files)

        train = {}
        test = {}

        for filename in tqdm(files[:len(files) // 2]):
            train[filename] = self.analyze_file(filename)

        for filename in tqdm(files[len(files) // 2:]):
            test[filename] = self.analyze_file(filename)

        self.train = train
        self.test = test

        return train, test


class PylintAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.files = glob(f'{ROOT_DIR}/pylint/submissions/*/*/*.py')

        self.annotation_id_map = {}

    def get_annotation_instances(self, file: str) -> List[AnnotationInstance]:
        pylint_output = StringIO()
        reporter = JSONReporter(pylint_output)
        lint.Run(["--module-naming-style=any", "--disable=C0304,C0301,C0303,C0305,C0114,C0115,C0116", file], reporter=reporter, exit=False)
        lint_result = list(map(lambda m: (m["line"] - 1, f"{m['message-id']}-{m['symbol']}"), json.loads(pylint_output.getvalue())))

        result = []
        for line, annotation in lint_result:
            if annotation not in self.annotation_id_map:
                next_id = len(self.id_annotation_map) + 1
                self.id_annotation_map[next_id] = annotation
                self.annotation_id_map[annotation] = next_id

            result.append((self.annotation_id_map[annotation], line))
        return result

    def load_train_test(self, file: str) -> None:
        print("Loading train and test data")
        with open(f'{ROOT_DIR}/pylint/output/analysis/{file}', 'rb') as data_file:
            self.train, self.test = pickle.load(data_file)

    def save_train_test(self, file: str) -> None:
        print("Saving train and test data")
        with open(f'{ROOT_DIR}/pylint/output/analysis/{file}', 'wb') as data_file:
            pickle.dump((self.train, self.test), data_file, pickle.HIGHEST_PROTOCOL)


class FeedbackAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()

        self.annotations_file = f"{ROOT_DIR}/data/annotations.tsv"
        self.date_time_format = "%Y-%m-%d %H:%M:%S.%f"

        self.submission_annotations_map: Dict[str, List[Tuple[int, int, datetime.datetime]]] = defaultdict(list)
        self.load_submission_annotations_map()
        self.set_files(glob(f'{ROOT_DIR}/data/exercises/*/*.py'))

    def load_submission_annotations_map(self) -> None:
        with open(self.annotations_file) as annotations_file:
            rows = csv.DictReader(annotations_file, delimiter='\t')
            for row in rows:
                line = row["line_nr"]
                if line != 'NULL':
                    line = int(line)
                    submission_id = row["submission_id"]
                    annotation = row["annotation_text"]
                    saved_id = int(row["saved_annotation_id"])
                    creation_time = datetime.datetime.strptime(row["created_at"], self.date_time_format)

                    if saved_id not in self.id_annotation_map:
                        self.id_annotation_map[saved_id] = annotation
                    self.submission_annotations_map[submission_id].append((saved_id, line, creation_time))

                    if submission_id not in self.annotations_per_submission_per_line:
                        self.annotations_per_submission_per_line[submission_id] = {}
                    if line not in self.annotations_per_submission_per_line[submission_id]:
                        self.annotations_per_submission_per_line[submission_id][line] = {}
                    self.annotations_per_submission_per_line[submission_id][line][saved_id] = annotation

    def get_annotation_instances(self, file: str) -> List[AnnotationInstance]:
        submission_id = file.split('/')[-1].split('.')[0]
        return list(map(lambda x: x[0:2], self.submission_annotations_map[submission_id]))

    def set_files(self, files, annotations_only=True) -> None:
        self.files = [file for file in files if not annotations_only or file.split('/')[-1].split('.')[0] in self.submission_annotations_map.keys()]

    def get_sorted_files(self) -> List[str]:
        return sorted(self.files, key=lambda i: self.submission_annotations_map[i.split('/')[-1].split('.')[0]][0][2])
