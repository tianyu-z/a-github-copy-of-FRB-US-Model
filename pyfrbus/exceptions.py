# For mypy typing
from typing import Optional


class MissingDataError(Exception):
    """Exception raised when missing data causes Frbus object initialization to fail."""

    def __init__(self, var: str):
        message = f"The variable `{var}` appears in the model but has no corresponding series in the input data. Failed to initialize model."  # noqa: E501
        super().__init__(message)


class ConvergenceError(Exception):
    """Exception raised when the solver fails to converge."""


class ComputationError(Exception):
    """Exception raised when evaluating some expression leads to overflow or etc."""

    def __init__(self, error_txt: str, caller: str):
        message = f"Computation error in {caller}: {error_txt}"
        super().__init__(message)


class InvalidModelError(Exception):
    """Exception raised when model is invalid, e.g. has an undeclared constant"""

    def __init__(self, model_issue: str):
        message = f"Model loaded is invalid: {model_issue}"
        super().__init__(message)


class InvalidArgumentError(Exception):
    """Exception raised when an invalid argument is passed as configuration"""

    def __init__(self, caller: str, arg_name: str, value: Optional[str] = None):
        # If two arguments passed
        if not value:
            message = f"Invalid argument passed to {caller}(): {arg_name}"
        # If three arguments passed
        else:
            message = f"Invalid argument passed to {caller}(): `{value}` is not a valid value for argument `{arg_name}`."  # noqa: E501
        super().__init__(message)
