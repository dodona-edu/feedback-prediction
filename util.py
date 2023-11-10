import datetime
from collections import Counter


def timed(func):
    start = datetime.datetime.now()
    res = func()
    print(datetime.datetime.now() - start)
    return res


def list_fully_contains_other_list(list1, list2):
    """
    Return True if list1 fully contains all items (with duplicates) form list2
    Source: https://stackoverflow.com/questions/68390939/how-to-check-if-a-list-contains-all-the-elements-of-another-list-including-dupli
    """
    return not (Counter(list2) - Counter(list1))
