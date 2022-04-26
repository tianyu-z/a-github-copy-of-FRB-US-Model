import re
from functools import reduce

# For mypy typing
from typing import List, Tuple, Optional, Dict
from pandas.core.frame import DataFrame
from pandas import Period

# Imports from this package
import pyfrbus.constants as constants


# "lex" equations - separate into identifier tokens (aka variables) and everything else
def lex_eqs(eqs: List[str]) -> List[List[Tuple[str, Optional[Tuple[str, int]]]]]:
    return [lex_eq(eq) for eq in eqs]


# Lex individual equation; output is a list of pairs (eq_text, token)
# Last pair will be (eq_text, None)
# 'eq_text' is a string of non-identifier equation text
# Token is represented as a pair (varname, period)
def lex_eq(eq: str) -> List[Tuple[str, Optional[Tuple[str, int]]]]:
    # Regex matches variables like rff, rff(-1), rff(1)
    # Gives pre-token and post-token text as other capture groups
    regex = r"(.*?)\b([a-z]\w+)(?:\((-?\d+)\))?(?!\w)(.*)"

    output: List[Tuple[str, Optional[Tuple[str, int]]]] = []

    # Search for first match
    m = re.match(regex, eq)
    # Placeholder to store a match from CONST_SUPPORTED_FUNCTIONS
    keyword_text = ""
    while m:
        # Break into groups
        prefix = m.group(1)
        identifier = m.group(2)
        # Optional third capture group, e.g. the -1 in rff(-1)
        period = m.group(3)
        rest = m.group(4)

        # Check if detected identifier is actually a supported function: log, exp, etc.
        if identifier in constants.CONST_SUPPORTED_FUNCTIONS_EX:
            # If so, save text and add to the next group
            keyword_text += prefix + (
                f"{identifier}({period})" if period else identifier
            )
        else:
            output.append(
                (keyword_text + prefix, (identifier, 0 if not period else int(period)))
            )
            # Reset CONST_SUPPORTED_FUNCTIONS text holder
            keyword_text = ""

        # Move on to rest of equation
        m = re.match(regex, rest)

    # No more tokens; final pair has None for token
    output.append((keyword_text + rest, None))
    return output


# Turn lexed equations back into equation strings
def to_eqs(lexed_eqs: List[List[Tuple[str, Optional[Tuple[str, int]]]]]) -> List[str]:
    return [to_eq(lexed_eq) for lexed_eq in lexed_eqs]


# Transform individual lexed equation back into an equation string
# Identifier tokens are turned into variable names and inserted inside equation text
def to_eq(lexed_eq: List[Tuple[str, Optional[Tuple[str, int]]]]) -> str:
    return reduce(
        lambda accum, pair: "".join([accum, pair[0], to_varname(pair[1])]), lexed_eq, ""
    )


# Turn identifier token into string variable name
def to_varname(identifier: Optional[Tuple[str, int]]) -> str:
    # Return empty if None
    if not identifier:
        return ""
    elif identifier[1] == 0:
        return identifier[0]
    # Handle lags/leads
    else:
        return f"{identifier[0]}({identifier[1]})"


# Shifts variables in lexed equation forward by n periods
def shift_eq(
    lexed_eq: List[Tuple[str, Optional[Tuple[str, int]]]], n: int
) -> List[Tuple[str, Optional[Tuple[str, int]]]]:

    return [(prefix, shift_var(identifier, n)) for (prefix, identifier) in lexed_eq]
    return reduce(
        lambda accum, pair: "".join([accum, pair[0], shift_var(pair[1], n)]),
        lexed_eq,
        "",
    )


# Shift individual variable forward by n periods
def shift_var(
    identifier: Optional[Tuple[str, int]], n: int
) -> Optional[Tuple[str, int]]:
    if not identifier:
        return identifier
    else:
        return (identifier[0], identifier[1] + n)


# Turn leads in lexed equations into contemporaneous new variables
def remove_leads(
    lexed_eqs: List[List[Tuple[str, Optional[Tuple[str, int]]]]],
    data: DataFrame,
    start_date: Period,
    n_periods: int,
) -> List[List[Tuple[str, Optional[Tuple[str, int]]]]]:

    return [
        remove_leads_from_eq(lexed_eq, data, start_date, n_periods)
        for lexed_eq in lexed_eqs
    ]


# Turn leads in lexed equation into contemporaneous new variables
# e.g. (rff, 1) -> (rff_1, 0)
def remove_leads_from_eq(
    lexed_eq: List[Tuple[str, Optional[Tuple[str, int]]]],
    data: DataFrame,
    start_date: Period,
    n_periods: int,
) -> List[Tuple[str, Optional[Tuple[str, int]]]]:

    tmp = []

    for i in range(len(lexed_eq)):
        (prefix, identifier) = lexed_eq[i]

        # Last entry is (eq_text, None) - skip it
        if not identifier:
            tmp.append(lexed_eq[i])
        # Replace with data if you are past the terminal period
        elif identifier[1] >= n_periods:
            # Take current equation text, append terminal value
            prefix += str(data.loc[start_date + identifier[1], identifier[0]])
            # Then concatenate that to the next piece of text
            lexed_eq[i + 1] = (prefix + lexed_eq[i + 1][0], lexed_eq[i + 1][1])
        # Turn lead variables into contemporaneous variables with suffix, e.g. rff_5
        elif identifier[1] > 0:
            tmp.append((prefix, (f"{identifier[0]}_{identifier[1]}", 0)))
        # Otherwise just continue
        else:
            tmp.append(lexed_eq[i])

    return tmp


# Substitutes variable identifiers for solution vector (x) and exo/lag data (data)
def xsub(
    lexed_eq: List[Tuple[str, Optional[Tuple[str, int]]]],
    is_exo: Dict[str, bool],
    lag_exo_idx_dict: Dict[str, int],
    endo_idx_dict: Dict[str, int],
) -> str:
    output = ""
    for eq_text, identifier in lexed_eq:
        output += eq_text
        if not identifier:
            continue
        elif identifier[1] < 0:
            lag_var_idx = lag_exo_idx_dict[identifier[0]]
            # Lag period; -1 because the last row of data is the current period
            period = identifier[1] - 1
            output += f"data[{period},{lag_var_idx}]"
        elif is_exo[identifier[0]]:
            exo_idx = lag_exo_idx_dict[identifier[0]]
            output += f"data[-1,{exo_idx}]"
        else:
            endo_idx = endo_idx_dict[identifier[0]]
            output += f"x[{endo_idx}]"
    return output
