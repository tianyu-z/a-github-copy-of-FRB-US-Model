import re
import pandas
import itertools

# For mypy typing
from typing import List, Tuple, Iterable, Dict, TypeVar, Iterator
from typing_extensions import Protocol
from pandas import DataFrame, PeriodIndex
from numpy import ndarray


def list_abs(y: List[float]) -> List[float]:
    return [abs(x) for x in y]


def np2df(vals: ndarray, index=None, columns=None) -> DataFrame:
    data_frame = pandas.DataFrame(vals)
    data_frame.index = index
    data_frame.columns = columns
    return data_frame


# Retrieves indices in data frame index of CONTIGUOUS period span
def get_periods_idxs(periods: PeriodIndex, data: DataFrame) -> List[int]:
    start = list(data.index).index(periods[0])
    end = list(data.index).index(periods[-1])
    return list(range(start, end + 1))


# Looping over l, we replace every regex match with its corresponding element of repls
# This is generally the fastest way to do a mass replace
def sub_dict_or_raise(l: Iterable[str], regex: str, repls: Dict[str, str]) -> List[str]:
    return [re.sub(regex, lambda mobj: repls[mobj.group(0)], s) for s in l]


# Helper used to form a regex to match dict keys
# Note: currently only escapes square brackets
def join_escape_regex(repls: Dict[str, str]) -> str:
    return (
        "("
        + "|".join(
            [key.replace("[", "\\[").replace("]", "\\]") for key in repls.keys()]
        )
        + ")"
    )


# Type variables used in remove, flatten, unzip
# Prefixed with _ so they are not accidentally referenced
_T = TypeVar("_T")
_S = TypeVar("_S")
_C = TypeVar("_C", covariant=True)


# Type of something that is both Sized and Iterable
# Normally, you'd call that Collection, but pandas.Index is not a Collection
# as it doesn't implement __contains__
class SizedIterable(Protocol[_C]):
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator[_C]:
        pass


# Removes x from list l and returns the resulting list
def remove(l: List[_T], x: _T) -> List[_T]:
    l.remove(x)
    return l


# Takes a list-of-lists ands flattens it one level
def flatten(l: List[List[_T]]) -> List[_T]:
    return list(itertools.chain(*l))


# Takes a list of pairs and returns a pair of lists
def unzip(l: Iterable[Tuple[_T, _S]]) -> Tuple[List[_T], List[_S]]:
    l1 = []
    l2 = []
    for (x, y) in l:
        l1.append(x)
        l2.append(y)
    return (l1, l2)


# Takes a list and returns a dictionary from list elements to their indices
def idx_dict(l: SizedIterable[_T]) -> Dict[_T, int]:
    return dict(zip(l, range(0, len(l))))


# Takes a hash of unique key-value pairs and returns the inverted value-key map
def invert_dict(d: Dict[_T, _S]) -> Dict[_S, _T]:
    return {val: key for key, val in d.items()}


# Takes a function f of n-arity
# and applies it over a list l of n-arity items
def splatmap(f, l):
    return map(lambda x: f(*x), l)


# Returns the (first) indices of all elements of l2 in l1
def indices(l1: List[_T], l2: List[_T]) -> List[int]:
    return [l1.index(x) for x in l2 if x in l1]


# Return T/F if s can be coerced to a number
def is_float(s) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False
