from functools import partial
import numpy
import pandas

# For mypy typing
from typing import List, Tuple, Optional, Union
from pandas.core.frame import DataFrame
from pandas import Period, PeriodIndex

# Imports from this package
from pyfrbus.lib import is_float


# This is the end-user class for manipulating time series data with the usual DSL
class TimeSeriesData:
    def __init__(self, data: DataFrame, range: Optional[PeriodIndex] = None):
        # If you've already constructed a TimeSeriesData before,
        # there will be attributes e.g. TimeSeriesData.xgdp with
        # the SeriesDSL Descriptor. Need to delete them to un-override the setter
        for name in list(TimeSeriesData.__dict__.keys()):
            if not name[0] == "_":
                delattr(TimeSeriesData, name)

        # Keep as a reference, not a copy, so original will be modified
        self._data = data
        self.range = range

        for series_name in self._data.columns:
            # Need to do in this order, as the second line overrides the setter
            # Each e.g. d.xgdp is a function for an unevaluated time series operation
            setattr(self, series_name, make_attr(series_name, self._data))
            # The setter evaluates the intermediate expression as recursive time series
            # See SeriesDSL.__set__ below
            setattr(TimeSeriesData, series_name, SeriesDSL(series_name))

    # Define getattr to implement "method missing"-like functionality
    # At first, only series present in the dataframe at construction have functions
    # If the name of a series is requested, we check the data object to see if it was
    # added to the dataset after construction
    def __getattr__(self, name):
        # Already defined - just return
        if name in self.__dict__:
            return self.__dict__[name]
        # In dataframe but not defined
        elif name in self._data.columns:
            setattr(self, name, make_attr(name, self._data))
            setattr(TimeSeriesData, name, SeriesDSL(name))
            return self.__dict__[name]
        # Not present in either, i.e. an incorrect variable name
        else:
            raise AttributeError(f"Series `{name}` not present in dataset")

    # Similar to the above, we define setattr so that assignments to new series work
    def __setattr__(self, name, value):
        # Already defined - just set
        if name in self.__dict__:
            object.__setattr__(self, name, value)
        # Skip special attributes
        elif name == "range" or name[0] == "_":
            object.__setattr__(self, name, value)
        # In dataframe but not defined, i.e. on init
        # Just pass through to set
        elif name in self._data.columns:
            object.__setattr__(self, name, value)
        # Not present in either, i.e. a new variable
        # We need to set up the descriptor
        else:
            # Create attribute as method corresponding to series name
            object.__setattr__(self, name, make_attr(name, self._data))
            # Set up descriptor
            setattr(TimeSeriesData, name, SeriesDSL(name))
            # Set value, i.e. evaluate in descriptor __set__
            setattr(self, name, value)


# This is the Descriptor that allows us to do the time series DSL
class SeriesDSL:
    # Each attribute maintains a reference to its series name
    def __init__(self, series_name: str):
        self.series_name = series_name

    # For attributes declared as SeriesDSL, this overrides the setter, e.g. d.xgdp = ...
    def __set__(self, instance: TimeSeriesData, value: Union[float, "TimeSeriesOp"]):
        p_range: PeriodIndex = instance._data.index if instance.range is None else instance.range  # noqa: E501

        if not isinstance(value, (int, float, TimeSeriesOp)):
            raise TypeError(
                f"unsupported type set to series in TimeSeriesData: '{type(value).__name__}'"  # noqa: E501
            )
        elif isinstance(value, (int, float)):
            # If input is a number, we just set it
            instance._data.loc[p_range, self.series_name] = value
        else:
            # Instead of e.g. overwriting d.xgdp, we modify the DataFrame recursively
            # by evaluating the time series expression at each period
            for period in p_range:
                instance._data.loc[period, self.series_name] = value.nonrecur_eval(
                    period
                )[0]


# Defines the intermediate state of a recursive time series operation,
# allowing for declarative-style statements which are finally eval'd
# when the variable is to be set
class TimeSeriesOp:
    # Initialize an atomic time series expression, i.e. one variable with offset
    def __init__(self, data: DataFrame, series_name: str, offset: int):
        # Acceptable types for other in operands
        self.CONST_ACCEPTABLE_TYPES = (int, float, TimeSeriesOp)
        # Reference to the input data, used at eval time
        self._data = data

        # Check that offset is present or past (fwd-looking not supported)
        if offset > 0:
            raise ValueError(
                "Indices in time-series operations must be less than or equal to 0"
            )
        # Token list for single atomic expression
        self.tokens: List[Tuple[str, Union[str, float], int, str]] = [
            ("", series_name, offset, "")
        ]

    # Non-recursively evaluate the time series expression to get an actual series
    def nonrecur_eval(self, periods: Optional[Union[Period, PeriodIndex]] = None):
        if periods is None:
            periods = self._data.index
        elif type(periods) == Period:
            periods = pandas.period_range(periods, periods, freq="Q")

        return periods.map(
            lambda period: eval(
                "".join(map(partial(token_tuple_to_str, period=period), self.tokens))
            )
        )

    # Create a compound time series expression by adding two series,
    # or a series and a scalar
    def __add__(self, other: Union[float, "TimeSeriesOp"]):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for +: 'TimeSeriesOp' and '{type(other).__name__}'"  # noqa: E501
            )
        self.tokens = concat_tokens(self.tokens, other, "+")
        return self

    # Define the commuted "reverse" version, to handle e.g. 1.0 + d.rff(-1)
    # The 'r' methods will technically only be called if other is not a TimeSeriesOp
    def __radd__(self, other: float):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and 'TimeSeriesOp'"  # noqa: E501
            )
        return self + other

    # Create a compound time series expression by multiplying two series,
    # or a series and a scalar
    def __mul__(self, other: Union[float, "TimeSeriesOp"]):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for *: 'TimeSeriesOp' and '{type(other).__name__}'"  # noqa: E501
            )
        self.tokens = concat_tokens(self.tokens, other, "*")
        return self

    # Define the commuted "reverse" version, to handle e.g. 0.9 * d.rff(-1)
    def __rmul__(self, other: float):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for *: '{type(other).__name__}' and 'TimeSeriesOp'"  # noqa: E501
            )
        return self * other

    # Create a compound time series expression by dividing two series,
    # or a series and a scalar
    def __truediv__(self, other: Union[float, "TimeSeriesOp"]):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for /: 'TimeSeriesOp' and '{type(other).__name__}'"  # noqa: E501
            )
        self.tokens = concat_tokens(self.tokens, other, "/")
        return self

    # Define the commuted "reverse" version, to handle e.g. 1.0 / d.rff(-1)
    def __rtruediv__(self, other: float):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for /: '{type(other).__name__}' and 'TimeSeriesOp'"  # noqa: E501
            )
        self.tokens = concat_tokens([("", other, 0, "")], self, "/")
        return self

    # Create a compound time series expression by exponentiating two series,
    # or a series and a scalar
    def __pow__(self, other: Union[float, "TimeSeriesOp"]):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for **: 'TimeSeriesOp' and '{type(other).__name__}'"  # noqa: E501
            )

        self.tokens = concat_tokens(self.tokens, other, "**")
        return self

    # Define the commuted "reverse" version, to handle e.g. 0.9 ** d.rff(-1)
    def __rpow__(self, other: float):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for **: '{type(other).__name__}' and 'TimeSeriesOp'"  # noqa: E501
            )
        self.tokens = concat_tokens([("", other, 0, "")], self, "**")
        return self

    # Create a compound time series expression by subtracting two series,
    # or a series and a scalar
    def __sub__(self, other: Union[float, "TimeSeriesOp"]):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for -: 'TimeSeriesOp' and '{type(other).__name__}'"  # noqa: E501
            )

        self.tokens = concat_tokens(self.tokens, other, "-")
        return self

    # Define the commuted "reverse" version, to handle e.g. 1.0 - d.rff(-1)
    def __rsub__(self, other: float):
        if not isinstance(other, self.CONST_ACCEPTABLE_TYPES):
            raise TypeError(
                f"unsupported operand type(s) for -: '{type(other).__name__}' and 'TimeSeriesOp'"  # noqa: E501
            )
        return -self + other

    # Unary minus operator
    def __neg__(self):
        return -1 * self


# We have to define the lambda in a separate method for arcane scoping reasons
def make_attr(series_name: str, data: DataFrame):
    return lambda i: TimeSeriesOp(data, series_name, i)


# Assemble two TimeSeriesOp objects into a single one with an infix operator
# Note: the reason why this parenthesizing strategy works is that we are effectively
# offloading the logic of parentheses and precedence to the Python interpreter itself
def concat_tokens(
    tokens: List[Tuple[str, Union[str, float], int, str]],
    other: Union[float, "TimeSeriesOp"],
    op: str,
    skip_parens: bool = False,
):

    open_paren = "" if skip_parens else "("
    close_paren = "" if skip_parens else ")"
    # Parenthesize, starting with first token
    first_token = tokens[0]
    tokens[0] = (
        open_paren + first_token[0],
        first_token[1],
        first_token[2],
        first_token[3],
    )

    # Put operator between last token and the next set of tokens
    last_token = tokens[len(tokens) - 1]
    tokens[len(tokens) - 1] = (
        last_token[0],
        last_token[1],
        last_token[2],
        last_token[3] + op,
    )

    # Check for scalars
    if not isinstance(other, TimeSeriesOp):
        tokens += [("", other, 0, close_paren)]
    else:
        # End parenthetical
        other_last_token = other.tokens[len(other.tokens) - 1]
        other.tokens[len(other.tokens) - 1] = (
            other_last_token[0],
            other_last_token[1],
            other_last_token[2],
            other_last_token[3] + close_paren,
        )
        tokens += other.tokens

    return tokens


# Turn each token into a self._data -referencing string that can be eval'd
def token_tuple_to_str(
    tup: Tuple[str, Union[str, float], int, str], period: Period
) -> str:
    leading_paren = tup[0]
    name = tup[1]
    offset = tup[2]
    op = tup[3]

    # Check if the token is a scalar
    if is_float(name):
        return f"{leading_paren} {name} {op}"
    # Otherwise, it is an atomic series
    else:
        return f'{leading_paren} self._data["{name}"].shift(-{offset})["{period}"] {op}'


# Some importable overrides to enable use of functions with TimeSeriesOp objects
def log(x: Union[float, "TimeSeriesOp"]):
    if isinstance(x, TimeSeriesOp):
        x.tokens = wrap_function_call(x.tokens, "log")
        return x
    else:
        return numpy.log(x)


def exp(x: Union[float, "TimeSeriesOp"]):
    if isinstance(x, TimeSeriesOp):
        x.tokens = wrap_function_call(x.tokens, "exp")
        return x
    else:
        return numpy.exp(x)


def abs(x: Union[float, "TimeSeriesOp"]):
    if isinstance(x, TimeSeriesOp):
        x.tokens = wrap_function_call(x.tokens, "abs")
        return x
    else:
        return numpy.abs(x)


def max(*args):
    is_tsop = [isinstance(t, TimeSeriesOp) for t in args]
    if any(is_tsop):
        # Get first object, where we will store the tokens
        x = next(t for t in args if isinstance(t, TimeSeriesOp))

        # Concat max arguments together with ,
        if isinstance(args[0], TimeSeriesOp):
            tokens = args[0].tokens
        else:
            tokens = [("", args[0], 0, "")]
        for t in args[1 : len(args)]:
            tokens = concat_tokens(tokens, t, ",", skip_parens=True)

        x.tokens = wrap_function_call(tokens, "max")
        return x
    else:
        return numpy.max(list(args))


def min(*args):
    is_tsop = [isinstance(t, TimeSeriesOp) for t in args]
    if any(is_tsop):
        # Get first object, where we will store the tokens
        x = next(t for t in args if isinstance(t, TimeSeriesOp))

        # Concat min arguments together with ,
        if isinstance(args[0], TimeSeriesOp):
            tokens = args[0].tokens
        else:
            tokens = [("", args[0], 0, "")]
        for t in args[1 : len(args)]:
            tokens = concat_tokens(tokens, t, ",", skip_parens=True)

        x.tokens = wrap_function_call(tokens, "min")
        return x
    else:
        return numpy.min(list(args))


# Helper that wraps the full expression in the function
def wrap_function_call(tokens: List[Tuple[str, Union[str, float], int, str]], fun: str):
    first_token = tokens[0]
    tokens[0] = (
        f"{fun}(" + first_token[0],
        first_token[1],
        first_token[2],
        first_token[3],
    )
    last_token = tokens[len(tokens) - 1]
    tokens[len(tokens) - 1] = (
        last_token[0],
        last_token[1],
        last_token[2],
        last_token[3] + ")",
    )
    return tokens
