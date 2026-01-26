from __future__ import annotations

from dataclasses import dataclass

import casadi
import numpy as np

import pybamm


class CasadiAlgebraicSolver(pybamm.BaseSolver):
    """Solve a discretised model which contains only (time independent) algebraic
    equations using CasADi's root finding algorithm.

    Note: this solver could be extended for quasi-static models, or models in
    which the time derivative is manually discretised and results in a (possibly
    nonlinear) algebraic system at each time level.

    Parameters
    ----------
    tol : float, optional
        The tolerance for the maximum residual error (default is 1e-6).
    step_tol : float, optional
        The tolerance for the maximum step size (default is 1e-4).
    extra_options : dict, optional
        Any options to pass to the CasADi roots.
        Please consult `CasADi documentation <https://web.casadi.org/python-api/#rootfinding>`_ for
        details. By default:

        .. code-block:: python

            extra_options = {
                # Whether to throw an error if the solver fails to converge.
                "error_on_fail": False,
                # Verbosity level
                "verbose": False,
                # Whether to show warnings when evaluating the model
                "show_eval_warnings": False,
            }
    """

    def __init__(self, tol=1e-6, step_tol=1e-4, extra_options=None):
        super().__init__()
        default_extra_options = {
            "error_on_fail": False,
            "verbose": False,
            "show_eval_warnings": False,
        }
        extra_options = extra_options or {}
        self.extra_options = extra_options | default_extra_options
        self.step_tol = step_tol
        self.tol = tol
        self.name = "CasADi algebraic solver"
        self._algebraic_solver = True
        pybamm.citations.register("Andersson2019")

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    @property
    def step_tol(self):
        return self._step_tol

    @step_tol.setter
    def step_tol(self, value):
        self._step_tol = value

    def set_up_root_solver(
        self, model: pybamm.BaseModel, inputs_dict: dict, t_eval: np.ndarray
    ):
        """Create and return a _CasadiRootfinder object. The rootfinder is
        contains two internal rootfinders, one with a step tolerance that limits
        the step size, and one without. The second rootfinder is only used if the
        first rootfinder terminates due to a Newton step size which satisfies the
        step size tolerance, but does not satisfy the absolute tolerance.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        inputs_dict : dict
            Dictionary of inputs.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution (not used).
        """
        # Initialization is handled on the first call to solve
        model.algebraic_root_solver = _CasadiRootfinder()

    def _integrate_single(self, model, t_eval, inputs_dict, y0):
        """
        Calculate the solution of the algebraic equations through root-finding

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution
        inputs_dict : dict, optional
            Any input parameters to pass to the model when solving
        y0 : array-like
            The initial conditions for the model

        Returns
        -------
        :class:`pybamm.Solution`
            A Solution object containing the times and values of the solution,
            as well as various diagnostic messages.
        """
        # Record whether there are any symbolic inputs
        inputs_dict = inputs_dict or {}

        # Create casadi objects for the root-finder
        inputs = casadi.vertcat(*[v for v in inputs_dict.values()])

        # The casadi algebraic solver can read rhs equations, but leaves them unchanged
        # i.e. the part of the solution vector that corresponds to the differential
        # equations will be equal to the initial condition provided. This allows this
        # solver to be used for initialising the DAE solvers
        if model.rhs == {}:
            len_rhs = 0
            y0_diff = casadi.DM()
            y0_alg = y0
        else:
            len_rhs = model.len_rhs
            y0_diff = y0[:len_rhs]
            y0_alg = y0[len_rhs:]

        y_alg = None

        # Set the parameter vector
        p = np.empty(1 + len_rhs + inputs.shape[0])
        # Set the time
        p[0] = 0
        # Set the differential states portion of the parameter vector
        p[1 : 1 + len_rhs] = np.asarray(y0_diff).reshape(-1)
        # Set the inputs portion of the parameter vector
        p[1 + len_rhs :] = np.asarray(inputs).reshape(-1)

        if getattr(model, "algebraic_root_solver", None) is None:
            self.set_up_root_solver(model, inputs_dict, t_eval)
        roots = model.algebraic_root_solver

        timer = pybamm.Timer()
        integration_time = 0
        for t in t_eval:
            # Set the time for the parameter vector
            p[0] = t

            try:
                timer.reset()
                result = roots.solve(
                    model=model,
                    inputs=inputs,
                    t=t,
                    y0_alg=y0_alg,
                    y0_diff=y0_diff,
                    p=p,
                    tol=self.tol,
                    step_tol=self.step_tol,
                    options=self.extra_options,
                )
                integration_time += timer.time()
            except RuntimeError as err:
                message = err.args[0]
                raise pybamm.SolverError(
                    f"Could not find acceptable solution: {message}"
                ) from err

            # update initial guess for the next iteration
            y_alg_sol = result.y_alg_sol
            y0_alg = y_alg_sol
            y0 = casadi.vertcat(y0_diff, y0_alg)
            # update solution array
            if y_alg is None:
                y_alg = y_alg_sol
            else:
                y_alg = casadi.horzcat(y_alg, y_alg_sol)

        # Concatenate differential part
        y_diff = casadi.horzcat(*[y0_diff] * len(t_eval))
        y_sol = casadi.vertcat(y_diff, y_alg)

        # Return solution object (no events, so pass None to t_event, y_event)
        sol = pybamm.Solution(
            [t_eval],
            y_sol,
            model,
            inputs_dict,
            termination="final time",
        )
        sol.integration_time = integration_time
        return sol


def max_abs(x) -> float:
    return np.linalg.norm(x, ord=np.inf).astype(float)


@dataclass
class _CasadiRootfinderResult:
    y_alg_sol: casadi.DM
    success: bool
    fval_max: float


class _CasadiRootfinder:
    roots: casadi.Function | None = None
    roots_refine: casadi.Function | None = None

    def get_roots(self, model, inputs, tol, step_tol, options):
        roots = self.roots
        if roots is not None:
            return roots

        roots = self._set_up_root_solver(
            model=model, inputs=inputs, tol=tol, step_tol=step_tol, options=options
        )
        self.roots = roots
        return roots

    def get_roots_refine(self, model, inputs, tol, options):
        roots_refine = self.roots_refine
        if roots_refine is not None:
            return roots_refine

        roots_refine = self._set_up_root_solver(
            model=model, inputs=inputs, tol=tol, step_tol=0, options=options
        )
        self.roots_refine = roots_refine
        return roots_refine

    def solve(
        self, model, inputs, t, y0_alg, y0_diff, p, tol, step_tol, options
    ) -> _CasadiRootfinderResult:
        # Get the roots
        roots = self.get_roots(
            model=model, inputs=inputs, tol=tol, step_tol=step_tol, options=options
        )
        result = self._solve(
            roots,
            model=model,
            y0_alg=y0_alg,
            y0_diff=y0_diff,
            p=p,
            t=t,
            inputs=inputs,
            tol=tol,
        )
        if result.success:
            return result

        cannot_refine = step_tol == 0
        self._check_solution(result, tol, check_abstol=cannot_refine)

        roots_refine = self.get_roots_refine(
            model=model, inputs=inputs, tol=tol, options=options
        )
        results_refine = self._solve(
            roots_refine,
            model=model,
            y0_alg=result.y_alg_sol,
            y0_diff=y0_diff,
            p=p,
            t=t,
            inputs=inputs,
            tol=tol,
        )
        self._check_solution(results_refine, tol, check_abstol=True)
        return results_refine

    @staticmethod
    def _solve(
        roots: casadi.Function,
        model: pybamm.BaseModel,
        y0_alg: casadi.DM,
        y0_diff: casadi.DM,
        p: casadi.DM,
        t: float,
        inputs: casadi.DM,
        tol: float,
    ) -> _CasadiRootfinderResult:
        y_alg_sol = roots(y0_alg, p)

        # Check final output
        y_sol = casadi.vertcat(y0_diff, y_alg_sol)
        residuals = model.casadi_algebraic(t, y_sol, inputs)
        # Casadi does not give us the appropriate info to check if the solution
        # succeeds, so we check that the solution is finite and the maximum residual
        # is within the tolerance
        y_is_finite = np.isfinite(max_abs(y_alg_sol))
        fval_max = max_abs(residuals)
        success = y_is_finite and fval_max <= tol
        return _CasadiRootfinderResult(
            y_alg_sol=y_alg_sol, success=success, fval_max=fval_max
        )

    @staticmethod
    def _set_up_root_solver(model, inputs, tol, step_tol, options):
        """Create and return a CasADi roots object. The parameter argument to the
        roots is the concatenated time, differential states, and flattened inputs.

        Parameters
        ----------
        model : :class:`pybamm.BaseModel`
            The model whose solution to calculate.
        inputs_dict : dict
            Dictionary of inputs.
        t_eval : :class:`numpy.array`, size (k,)
            The times at which to compute the solution (not used).
        step_tol : float, optional
            The tolerance for the maximum step size.
            If None, the step tolerance is set to the provided step_tol.
        Returns
        -------
        None
            The roots function is stored in the model as `algebraic_root_solver`.
        """
        y0 = model.y0_list[0]

        # The casadi algebraic solver can read rhs equations, but leaves them unchanged
        # i.e. the part of the solution vector that corresponds to the differential
        # equations will be equal to the initial condition provided. This allows this
        # solver to be used for initialising the DAE solvers
        if model.rhs == {}:
            len_rhs = 0
        elif model.len_rhs_and_alg == y0.shape[0]:
            len_rhs = model.len_rhs
        else:
            len_rhs = model.len_rhs + model.len_rhs_sens

        len_alg = y0.shape[0] - len_rhs

        # Set up symbolic variables
        t_sym = casadi.MX.sym("t")
        y_diff_sym = casadi.MX.sym("y_diff", len_rhs)
        y_alg_sym = casadi.MX.sym("y_alg", len_alg)
        y_sym = casadi.vertcat(y_diff_sym, y_alg_sym)

        # Create parameter dictionary
        inputs_sym = casadi.MX.sym("inputs", inputs.shape[0])

        p_sym = casadi.vertcat(
            t_sym,
            y_diff_sym,
            inputs_sym,
        )

        alg = model.casadi_algebraic(t_sym, y_sym, inputs_sym)

        # Set constraints vector in the casadi format
        # Constrain the unknowns. 0 (default): no constraint on ui, 1: ui >= 0.0,
        # -1: ui <= 0.0, 2: ui > 0.0, -2: ui < 0.0.
        model_alg_lb = model.bounds[0][len_rhs:]
        model_alg_ub = model.bounds[1][len_rhs:]
        constraints = np.zeros_like(model_alg_lb, dtype=int)
        # If the lower bound is positive then the variable must always be positive
        constraints[model_alg_lb >= 0] = 1
        # If the upper bound is negative then the variable must always be negative
        constraints[model_alg_ub <= 0] = -1

        # Set up roots
        return casadi.rootfinder(
            "roots",
            "newton",
            dict(x=y_alg_sym, p=p_sym, g=alg),
            {
                **options,
                "abstol": tol,
                "abstolStep": step_tol,
                "constraints": list(constraints),
            },
        )

    @staticmethod
    def _check_solution(
        result: _CasadiRootfinderResult, tol: float, check_abstol: bool = True
    ):
        fval_max = result.fval_max
        if not np.isfinite(fval_max):
            raise pybamm.SolverError(
                "Could not find acceptable solution: solver returned NaNs or Infs"
            )

        if check_abstol and fval_max > tol:
            raise pybamm.SolverError(
                f"Could not find acceptable solution: "
                "solver terminated unsuccessfully and maximum solution error "
                f"({fval_max}) above tolerance ({tol})"
            )
