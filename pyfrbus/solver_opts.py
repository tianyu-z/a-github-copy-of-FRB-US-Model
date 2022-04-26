# For mypy typing
from typing import Dict, Optional


# Fill in default solver options
def solver_defaults(options: Optional[Dict]) -> Dict:
    defaults = {
        "newton": None,
        "single_block": False,
        "debug": False,
        "xtol": 1e-4,
        "rtol": 5e-4,
        "maxiter": 50,
        "trust_radius": 1000000,
        "precond": True,
    }

    # Merge options passed by user with other defaults
    if options:
        defaults.update(options)

    return defaults
