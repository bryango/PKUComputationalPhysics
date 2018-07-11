#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
import numpy as np
from toolkit.generic import abs_file_path, mrange
from toolkit.linearsol import gem
from toolkit.rootfinder import find_root_secant


# <codecell>
class InitialValueProblem(object):
    """ Initial Value Problem D[y] = f(t, y), initiated with:

    Args:
        func_with_params: f, wrapped in a parent function,
            with possible parameters; for example:

                def func_with_params(params=1.):
                    return (
                        lambda y-tuple: f_i(y-tuple), ...
                        # iterate i for every y-component
                    )   # expr rely on params=1.

            DEFAULT PARAMETER SHOULD BE SPECIFIED!
                Namely, always use KEYWORD ARGUMENTS for `func_with_params()`.

            MORE ON f(y-tuple): y-tuple = (t, y1, y2, ...),
                does NOT require unpacking;

        initial_values=None: (t, y1, y2, ...)_at_initial as numpy array;
            set initial_values now, or later, whenever u like!
    """
    def __init__(self, func_with_params, initial_values=None):
        self.rhs = func_with_params
        self.dim = len(self.rhs())
        self.initial = None if initial_values is None else \
            self.__check_initial_values(initial_values)

    def __check_initial_values(self, initial_values):
        value_error_msg = "illegal initial values for InitialValueProblem!"
        try:
            initial_array = np.array(initial_values)
        except ValueError:
            raise ValueError(value_error_msg)
        if initial_array.shape != (self.dim + 1, ):
            raise ValueError(value_error_msg)
        return initial_array

    def set_initial_values(self, initial_values):
        self.initial = self.__check_initial_values(initial_values)

    def dsolve_rk4(self, h: float, endpoint: float,
                   output=None, shooting_mode=False,
                   **kwargs) -> list:
        """ Differential solution given INITIAL values, using RK4;
            Equation(s): D[y] = f(t, y), y = (y1, y2, ...);

        Args:
            h: FIXED stepsize;
            output: write to specified file path in real time,
                SLOW but memory-saving;
            shooting_mode: get endpoint values ONLY,
                NO OUTPUT, fast & memory-saving;
                NOTE: `shooting_mode = True` overwrites `output` option
            **kwargs: extra parameters for f(t, y),
                namely `func_with_params(**kwargs)`;

        Returns:
            list of points, namely (t, y1, y2, ...).
        """

        # Some shorthands:
        output_enabled = (output is not None) and (not shooting_mode)
        derivatives = self.rhs(**kwargs)
        initial_values = self.initial
        n = self.dim

        if (endpoint > initial_values[0]) and (h > 0):
            pass
        elif (endpoint < initial_values[0]) and (h < 0):
            pass  # Time reversal (h < 0) is allowed!
        else:
            raise ValueError('inconsistent stepsize & endpoint'
                             'for `dsolve_rk4`!')

        def k_array_constructor(list_of_parameters):
            """ Constructs Δy (k_array) for every component. """
            return np.fromiter(
                (h * f(list_of_parameters) for f in derivatives),
                np.float, n
            )   # numpy.array from iterator, dimension: n

        pointset = [initial_values]
        if output_enabled:
            output_file = open(output, 'w')
            csv.writer(output_file).writerow(initial_values)
        try:
            while abs(endpoint - pointset[-1][0]) > 10**-16:
                # Always cover endpoint
                if h > 0:
                    h = min(h, endpoint - pointset[-1][0])
                else:
                    h = max(h, endpoint - pointset[-1][0])

                # array for all y-components
                k1_array = k_array_constructor(pointset[-1])

                # Elements of result are np.array; easy syntax:
                k2_parameters = pointset[-1] + np.insert(
                    k1_array / 2, 0, h / 2
                )  # incremment: (h/2, k/2), k-list for every y-component
                k2_array = k_array_constructor(k2_parameters)

                k3_parameters = pointset[-1] + np.insert(
                    k2_array / 2, 0, h / 2
                )
                k3_array = k_array_constructor(k3_parameters)

                k4_parameters = pointset[-1] + np.insert(
                    k3_array, 0, h
                )
                k4_array = k_array_constructor(k4_parameters)

                new_pt = pointset[-1] + np.insert(
                    (k1_array + 2 * k2_array + 2 * k3_array + k4_array) / 6,
                    0, h
                )  # incremment: (h, k_corrected)

                if output_enabled:
                    csv.writer(output_file).writerow(new_pt)
                if output_enabled or shooting_mode:
                    pointset[-1] = new_pt
                else:
                    pointset.append(new_pt)

        except IndexError:
            raise ValueError("illegal derivatives! `dsolve_rk4` failed.")

        if output_enabled:
            output_file.close()
        return pointset


# <codecell>
class SimpleBVPbyShooting(object):
    """ Simplified Boundary Value Problem, solved by shooting method,
        a.k.a. by converting it to InitialValueProblem,
        along with a lot of trial and error.
        SEE InitialValueProblem for more info.

    Args:
        domain: simple interval (left, right);
        boundary_constraints: 2 sides,
            @left: (n - 1) constraints,
                a11 * y1a + a12 * y2a + ... = a10,
                a21 * y1a + a22 * y2a + ... = a20,
                    ...
            @right: ONLY 1 constraint,
                b1 * y1b + b2 * y2b + ... = b0,
            ... wrapped in 2-level tuple:
            (
                ( (a11, a12, ... , a10),
                  (a21, a22, ... , a20),
                  ...
                  (a[n-1]1, a[n-11]2, ... , a[n-1]0)
                ),                  # (n-1) * (n+1) MATRIX
                (b1, b2, ... , b0)  # (n+1) tuple
            )

    NOTE: Set domain & boundaries now, or later, whenever u like!
    """
    def __init__(self, func_with_params,
                 domain=None, boundary_constraints=None):
        self.rhs = func_with_params
        self.dim = len(func_with_params())
        self.domain = None if domain is None else \
            self.__check_domain(domain)
        self.boundary = None if boundary_constraints is None else \
            self.__check_boundary(boundary_constraints)

    def __check_boundary(self, boundary_constraints):
        value_error_msg = "illegal boundary values " \
                          "for SimpleBVPbyShooting!"
        if len(boundary_constraints) != 2:
            raise ValueError(value_error_msg)
        try:
            boundary_left = np.array(boundary_constraints[0])
            boundary_right = np.array(boundary_constraints[1])
        except ValueError:
            raise ValueError(value_error_msg)
        if ( boundary_left.shape != (self.dim - 1, self.dim + 1) ) \
           or ( boundary_right.shape != (self.dim + 1, ) ):
            raise ValueError(value_error_msg)
        return boundary_left, boundary_right

    def __check_domain(self, domain):
        value_error_msg = "illegal domain for SimpleBVPbyShooting!"
        try:
            domain_np = np.array(domain)
        except ValueError:
            raise ValueError(value_error_msg)
        if domain_np.shape != (2,):
            raise ValueError(value_error_msg)
        return domain

    def set_boundary(self, boundary_constraints):
        self.boundary = self.__check_boundary(boundary_constraints)

    def set_domain(self, domain):
        self.domain = self.__check_domain(domain)

    def dsolve_shooting(self, h: float,
                        first_guess: (int, float), accuracy_goal=1e-16,
                        output=None, **kwargs) -> list:
        """ Differential solution given certain boundary values, via shooting;
            See docstring for `InitialValueProblem.dsolve_rk4()`.

        Additional Args:
            first_guess: (k: int, y[k]: float),
                guess the initial value of k-th component y[k];
            accuracy_goal: passed to rootfinder `find_root_secant()`,
                as a "stop-iteration" criterion;
        """

        n = self.dim
        boundary_left, boundary_right = self.boundary

        k, yk_guess = first_guess
        if (k not in mrange(1, n)) or (type(yk_guess) is not float):
            raise ValueError('illegal first guess for `dsolve_shooting()`')

        # GUESS an initial value for the k-th component;
        # ... and FIND initial values for other components:
        sq_matrix = np.delete(boundary_left[:, :-1], k - 1, axis=1)

        def initial_values(initial_yk):
            # Solve a (n-1) dimensional matrix!
            rhs_vec = boundary_left[:, -1] \
                - initial_yk * boundary_left[:, (k - 1)]
            return (
                self.domain[0],
                *np.insert(gem(sq_matrix, rhs_vec), k - 1, initial_yk)
            )

        ivp = InitialValueProblem(self.rhs)

        def shooting_result(initial_yk):
            ivp.set_initial_values(initial_values(initial_yk))
            endpoint_values = ivp.dsolve_rk4(
                h, self.domain[1], shooting_mode=True,
                **kwargs
            )[-1]
            return np.dot(
                boundary_right[:-1], endpoint_values[1:]
            )

        def shooting_error(initial_yk):
            return shooting_result(initial_yk) - boundary_right[-1]

        try1 = yk_guess
        try2 = yk_guess * (boundary_right[-1] / shooting_result(try1))
        best_yk = find_root_secant(
            shooting_error, try1, try2, accuracy=accuracy_goal
        )
        best_init = initial_values(best_yk)

        ivp.set_initial_values(best_init)
        return ivp.dsolve_rk4(
            h, self.domain[1], output=output,
            **kwargs
        )


if __name__ == "__main__":
    # <codecell>
    complaint = 'Number of steps should be positive int! instead we get '
    try:
        nsteps = int(sys.argv[1])
    except IndexError:
        nsteps = 10000
    except (NameError, TypeError, ValueError):
        print(complaint + sys.argv[1])
        nsteps = 10000
    if nsteps <= 0:
        print(complaint + sys.argv[1])
        nsteps = 10000

    def derivatives(omega=1.):
        return (
            lambda y: y[4],
            lambda y: y[5],
            lambda y: y[6],
            lambda y: omega * y[5],
            lambda y: - omega * y[4],
            lambda y: 0
        )  # y[0] for t, how convenient!
    init = (0., 0., 0., 0., 0., 2., 0.1)

    magnetic_cyclic = InitialValueProblem(derivatives, init)

    file_path = '/../../csv/magnetic_cyclic.csv'
    abs_path = abs_file_path(file_path)

    # <codecell>
    # %%timeit
    magnetic_cyclic.dsolve_rk4(
        40 / nsteps, 40, output=abs_path)
    print(f'{nsteps} steps, solved!')
