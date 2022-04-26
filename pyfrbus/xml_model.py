import re

# Imports from this package
from pyfrbus.equations import clean_eq
from pyfrbus.exceptions import InvalidArgumentError

# For mypy typing
from typing import List, Dict, cast, Tuple
from lxml.etree import Element


# Retrieve names of endogenous variables from XML
# i.e. variables with equations
def endo_names_from_xml(root: Element) -> List[str]:
    # "if"s are only there so t types correctly
    return [
        name.text
        for name in root.xpath("./variable[standard_equation]/name")
        if name.text is not None
    ]


# Retrieve equations for endogenous variables
def equations_from_xml(root: Element) -> List[str]:
    return [
        # Strip trailing whitespace, and reformat internal whitespace to be " "
        clean_eq(eq.text)
        for eq in root.xpath("./variable/standard_equation/python_equation")
        if eq.text is not None
    ]


# Load MCE equations from xml, with variable names so var equations can be replaced
def mce_from_xml(root: Element, mce_type: str) -> Tuple[List[str], List[str]]:
    mce_xpath = get_mce_xpath(mce_type)
    mce_eqs = [clean_eq(eq.text) for eq in root.xpath(mce_xpath) if eq.text is not None]
    mce_vars = [
        name.text
        for name in root.xpath(f"{mce_xpath}/../../name")
        if name.text is not None
    ]
    return (mce_eqs, mce_vars)


# Retrieve names of exogenous variables from XML
# i.e. variables with equation_type "Exogenous"
def exo_names_from_xml(root: Element) -> List[str]:
    return [
        name.text
        for name in root.xpath("./variable[equation_type='Exogenous']/name")
        if name.text is not None
    ]


# Retrieve constants that appear in equations
def constants_from_xml(root: Element) -> Dict[str, float]:
    return get_constants(root, "./variable/standard_equation/coeff")


# Retrieve constants that appear in equations
def mce_constants_from_xml(root: Element, mce_type: str) -> Dict[str, float]:
    mce_xpath = get_mce_xpath(mce_type)
    return get_constants(root, f"{mce_xpath}/../coeff")


# Refactored to simplify standard/mce constant retrieval
def get_constants(root: Element, coeff_xpath: str) -> Dict[str, float]:
    coeff_names = [
        # Re-format constant names to the format that shows up in python equations
        re.sub(r"y_(.*?)\((\d+)\)", r"y_\1_\2", cf_name.text)
        for cf_name in root.xpath(f"{coeff_xpath}/cf_name")
        if cf_name.text is not None
    ]
    coeff_vals = [
        # Cast to satisfy type checker
        # Obviously this will crash if the XML is malformed
        float(cast(float, cf_val.text))
        for cf_val in root.xpath(f"{coeff_xpath}/cf_value")
        if cf_val is not None
    ]
    return dict(zip(coeff_names, coeff_vals))


# Retrieve list of variables with a stochastic_type tag, if it's not NO
# Variables to be shocked in stochastic sims
def stoch_shocks(root: Element) -> List[str]:
    return [
        name.text
        for name in root.xpath("./variable[stochastic_type!='NO']/name")
        if name.text is not None
    ]


# Generate xpath based on mce specification
def get_mce_xpath(mce_type: str) -> str:
    if mce_type == "all":
        return "./variable/mce_equation/python_equation"
    elif mce_type == "mcap":
        return "./variable/mce_equation[mce_group='mcap']/python_equation"
    elif mce_type == "wp":
        return "./variable/mce_equation[mce_group='mcwp']/python_equation"
    elif mce_type == "mcap+wp":
        return "./variable/mce_equation[mce_group='mcap' or mce_group='mcwp']/python_equation"  # noqa: E501
    else:  # Should not occur - error raised in Frbus.__init__
        raise InvalidArgumentError("get_mce_xpath", "mce_type", mce_type)
