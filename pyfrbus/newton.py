from numpy.linalg import norm
from numpy import array, isnan, concatenate, identity
from scikits import umfpack
from scipy.optimize import minimize
import warnings

# For mypy typing
from typing import Callable, Dict
from numpy import ndarray
from scipy.sparse import csr_matrix

# Imports from this package
from pyfrbus.exceptions import ConvergenceError


# Newton's method root finder
def newton(
    call_fun: Callable[[ndarray, ndarray, ndarray], ndarray],
    call_jac: Callable[[ndarray, ndarray, ndarray], csr_matrix],
    guess: ndarray,
    vals: ndarray,
    solution: ndarray,
    options: Dict,
) -> ndarray:

    # Retrieve solver options
    debug: bool = options["debug"]
    xtol: float = options["xtol"]
    rtol: float = options["rtol"]
    maxiter: int = options["maxiter"]
    precond: bool = options["precond"]

    # Initial iteration
    # Evaluate model at guess
    fun_val = array(call_fun(guess, vals, solution))
    # Evaluate Jacobian at guess
    jac = call_jac(guess, vals, solution)

    # Compute step up to maxiter times
    for iter in range(maxiter):
        print(f"resid={norm(fun_val)}") if debug else None  # type: ignore

        # Compute scaling preconditioner to improve condition of matrix
        scale = (
            get_preconditioner(jac) if precond else csr_matrix(identity(jac.shape[0]))
        )
        # Compute solution sparsely
        with warnings.catch_warnings():
            if not debug:
                warnings.simplefilter("ignore")
            delta = umfpack.spsolve(scale @ jac, scale @ -fun_val)

        # Choose a step length
        # Starting with the full Newton step
        alpha = 1.0
        while True:
            # Scale step
            delta_tmp = alpha * delta
            # Update guess
            guess_tmp = guess + delta_tmp

            # Check if the step produces no NaNs and no warnings
            with warnings.catch_warnings():
                # So that we can check the step length and damp if it goes out of bounds
                warnings.filterwarnings("error")
                try:
                    # Call the function and jacobian to check for warnings or NaNs
                    # Save output for next iteration
                    # Evaluate model at guess
                    fun_val = array(call_fun(guess_tmp, vals, solution))
                    # Evaluate Jacobian at guess
                    jac = call_jac(guess_tmp, vals, solution)

                    if not any(isnan(fun_val)) and not any(isnan(jac.data)):
                        # No issues, save the step and continue
                        delta = delta_tmp
                        guess = guess_tmp
                        break
                except RuntimeWarning:
                    pass

            # Otherwise, scale step down by half and try again
            alpha = alpha / 2
            # Throw an error if we get a bad step
            if alpha < 1e-5:
                raise ConvergenceError("Newton solver has diverged, no solution found.")
        print(f"alpha:{alpha}") if debug else None  # type: ignore

        # Throw an error if we get a bad step
        if isnan(norm(delta)):
            raise ConvergenceError("Newton solver has diverged, no solution found.")

        # Return if next step is within specified tolerances
        print(f"delta={norm(delta)}") if debug else None  # type: ignore
        print("") if debug else None  # type: ignore
        if norm(delta) < xtol:
            # Throw error if step tolerance is reached, but residual is still large
            if norm(fun_val) < rtol:
                return guess
            else:
                raise ConvergenceError(
                    f"Newton solver has reached xtol, but with large residual; resid = {norm(fun_val)}"  # noqa: E501
                )

    # Throw an error if solver has iterated for too long
    raise ConvergenceError(
        f"Exceeded maxiter = {maxiter} in Newton solver, solution has not converged; last stepsize: {norm(delta)}"  # noqa: E501
    )


# Dogleg trust-region method root finder
def trust(call_fun, call_jac, guess, vals, solution, options: Dict):

    # Retrieve solver options
    debug: bool = options["debug"]
    xtol: float = options["xtol"]
    rtol: float = options["rtol"]
    maxiter: int = options["maxiter"]
    trust_radius: float = options["trust_radius"]
    precond: bool = options["precond"]

    eta = 0.1
    radius = trust_radius / 2

    for iter in range(maxiter):

        print(f"iteration: {iter}") if debug else None  # type: ignore
        print(f"radius={radius}") if debug else None  # type: ignore

        fun_val = array(call_fun(guess, vals, solution))

        print(f"resid={norm(fun_val)}") if debug else None  # type: ignore

        jac = call_jac(guess, vals, solution)
        p = dogleg(fun_val, jac, radius, precond, debug)
        ratio = reduction_ratio_refactored(
            call_fun, fun_val, guess, vals, solution, jac, p
        )

        print(f"ratio={ratio}") if debug else None  # type: ignore

        if ratio < 0.25:
            radius = 0.25 * norm(p)
        # Condition on norm(p) for radius expansion is given some wiggle room
        # as long as we get within 5% of radius, I think it's good enough to expand
        elif ratio > 0.75 and norm(p) > radius * 0.95:
            radius = min(2 * radius, trust_radius)
        else:
            radius = radius

        og = guess
        if ratio > eta:
            guess = guess + p
        else:
            guess = guess

        delta = og - guess

        print(f"delta={norm(delta)}") if debug else None  # type: ignore
        print("") if debug else None  # type: ignore
        # Check solution if step is small or radius has contracted
        if (norm(delta) > 0 and norm(delta) < xtol) or radius < 1e-8:
            # Throw error if step tolerance is reached, but residual is still large
            if norm(fun_val) < rtol:
                return guess
            else:
                raise ConvergenceError(
                    f"Trust-region solver has reached xtol, but with large residual; resid = {norm(fun_val)}"  # noqa: E501
                )

    # Throw an error if solver has iterated for too long
    raise ConvergenceError(
        f"Exceeded maxiter = {maxiter} in trust-region solver, solution has not converged; last stepsize: {norm(delta)}"  # noqa: E501
    )


def cauchy_point(fun_val, jac, radius) -> ndarray:
    tk: float = min(
        1,
        (norm(jac.transpose() @ fun_val) ** 3)
        / (
            radius
            * ((fun_val @ jac) @ (jac.transpose() @ jac) @ (jac.transpose() @ fun_val))
        ),
    )
    return -tk * (radius / norm(jac.transpose() @ fun_val)) * jac.transpose() @ fun_val


def reduction_ratio(call_fun, fun_val, guess, vals, solution, jac, p) -> float:
    return (norm(fun_val) ** 2 - norm(call_fun(guess + p, vals, solution)) ** 2) / (
        norm(fun_val) ** 2 - norm(fun_val + jac @ p) ** 2
    )


def reduction_ratio_refactored(
    call_fun, fun_val, guess, vals, solution, jac, p
) -> float:
    return (
        merit(call_fun, guess, vals, solution)
        - merit(call_fun, guess + p, vals, solution)
    ) / (
        merit(call_fun, guess, vals, solution)
        - model(call_fun, guess, p, jac, vals, solution)
    )


def merit(call_fun, point, vals, solution):
    return (norm(call_fun(point, vals, solution)) ** 2) / 2


def model(call_fun, guess, p, jac, vals, solution):
    return (
        merit(call_fun, guess, vals, solution)
        + (p @ jac.transpose() @ call_fun(guess, vals, solution))
        + (p @ jac.transpose() @ jac @ p) / 2
    )


def dogleg(fun_val, jac, radius, precond, debug) -> ndarray:
    p: ndarray = cauchy_point(fun_val, jac, radius)
    if norm(p) == radius:
        return p
    else:
        scale = (
            get_preconditioner(jac) if precond else csr_matrix(identity(jac.shape[0]))
        )
        with warnings.catch_warnings():
            if not debug:
                warnings.simplefilter("ignore")
            z: ndarray = umfpack.spsolve(scale @ jac, scale @ -fun_val)

        # Using a minimizer to find largest tau in [0,1]
        def max_tau(tau):
            return 1 - tau

        # Define constraint on norm of output vector
        def constraint(tau):
            return radius - norm(p + tau * (z - p))

        # Choose largest tau in [0,1] such that ||p+tau(z-p)|| < radius
        minim = minimize(
            max_tau, 1, bounds=[(0, 1)], constraints={"type": "ineq", "fun": constraint}
        )

        tmp = p + minim.x[0] * (z - p)
        print(f"{minim.x[0]}: norm(p)={norm(tmp)}") if debug else None  # type: ignore
        return tmp


# Compute preconditioner to improve condition number of Jacobian
# which may improve solution quality
def get_preconditioner(jac) -> csr_matrix:
    return csr_matrix(
        (
            1 / concatenate(abs(jac).max(1).toarray()),
            (range(jac.shape[0]), range(jac.shape[1])),
        )
    )
