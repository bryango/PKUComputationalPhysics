#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from toolkit.generic import grid_generator


# <codecell>
class PDEBase(object):
    """ PDE base class, designed to be generic! """

    def __init__(self):
        pass

    def _check_domain(self, domain):
        value_error_msg = "illegal domain for PDE!"
        try:
            domain_np = np.array(domain)
        except ValueError:
            raise ValueError(value_error_msg)
        if domain_np.shape != (2,):
            raise ValueError(value_error_msg)
        if not (domain_np[0] < domain_np[1]):
            raise ValueError(value_error_msg)
        return domain_np

    def set_initial(self, initial_distributions):
        self.initial_func = initial_distributions

    def _output_init(self, output_file, x_grid, initial_pt):
        csv.writer(output_file).writerows([
            ['x_grid', *x_grid],
            ['t', 'u'],
            initial_pt
        ])

    def _plot_init(self, plot_path, fig, ax, x0, x_max, discrete_initial):
        os.makedirs(plot_path, exist_ok=True)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u$')
        ax.set_xlim([x0, x_max])
        ax.set_ylim(list(map(
            lambda func: func(discrete_initial) * 1.05,
            [min, max]
        )))


# <codecell>
class Advection1D(PDEBase):
    """ PDE: 1D advecture, equation:
            (D_t + a(x, t) * D_x) [u(x, t)] = 0;
        the equation is rather trivial, while the initial input is not.

    Args:

        velocity_function: function of t, x,
            the velocity of propagation, namely a(x, t)

            DEFAULT PARAMETER(S) EXPECTED!
                Namely, always use KEYWORD ARGUMENTS
                ... for `velocity_function()`.
                NOTE: Upwind direction is fixed by a(x0, t0).

        initial_distributions: wrapped in a parent function,
            with possible parameters; for example:

                def initial_distributions(params=1.):
                    return lambda x: some_expression_of(x)
                    # ... rely on params=1.

            DEFAULT PARAMETER(S) EXPECTED!
                Namely, always use KEYWORD ARGUMENTS
                ... for `initial_distributions()`.

        supp_domain: interval (start, end),
            NON-ZERO x-region of `initial_distributions`,
            data beyond this region is assumed to be ZERO.
            NOTE: 'supp' is short for 'support'.

        boundary: (function_of_t, function_of_t),
            ONLY accepts FIXED boundary (Dirichlet, or 1st type);
            if None then ASSUME ZERO for _upstream_ boundary.

    NOTE: `initial_distributions`, `supp_domain`, `boundary`
        can be set when initialized,
        OR set later by `set_whatever()` methods.
    """

    def __init__(self, velocity_function,
                 initial_distributions=None,
                 supp_domain=None, boundary=None):
        self.a_func = velocity_function
        self.domain = None if supp_domain is None else \
            self._check_domain(supp_domain)
        self.initial_func = None if initial_distributions is None else \
            initial_distributions
        self.boundary = self._check_boundary(boundary)

    def _check_boundary(self, boundary):
        if boundary is None:
            return None

        value_error_msg = "illegal boundary values " \
                          "for SimpleBVPbyShooting!"
        if len(boundary) != 2:
            raise ValueError(value_error_msg)
        if not all(map(
            callable, boundary
        )):
            raise ValueError(value_error_msg)
        return boundary

    def set_domain(self, supp_domain):
        self.domain = self._check_domain(supp_domain)

    def set_boundary(self, boundary_constraints):
        self.boundary = self._check_boundary(boundary_constraints)

    def _u_generator_upwind(self, old_u, i, a_value, coeff):
        """ Returns upwind result """
        u = old_u
        if a_value > 0:
            if i == 0:
                raise IndexError('left boundary encountered!')
            return u[i] - coeff * (u[i] - u[i - 1])
        else:
            return u[i] - coeff * (u[i + 1] - u[i])

    def pde_solve_upwind(self,
                         dt: float, dx: float,
                         t_end: float, t0=0.,
                         output=None, plot=None,
                         plot_t_samplesize=50,
                         **kwargs) -> list:
        """ Solve using upwind scheme.

        Args:

            output: write to specified csv file path in real time,
                SLOW but memory-saving;

            plot: generate plot & save to designated path (directory);

            **kwargs: extra parameters for initial distributions,
                namely `initial_distributions(**kwargs)`;

        Returns:
            ( x_grid,
              ( (t, u_values) for t in t_grid )
            ), if output is set then returns None.

            NOTE:
                1st component: shape ( len(x_grid), );
                2nd component: shape ( len(t_grid), len(x_grid) + 1 );
        """

        if (t_end < t0) or (dt < 0):
            # Time reversal (dt < 0) not supported yet!
            raise ValueError('inconsistent stepsize & endpoint'
                             'for `pde_solve_upwind`!')

        # Some shorthands:
        output_enabled = (output is not None)
        x0, x_max = self.domain
        func = self.initial_func(**kwargs)
        x_grid = grid_generator(x0, x_max, dx)
        x_len = x_grid.shape[0]

        # Initial input
        discrete_initial = list(map(func, x_grid))

        # For output (memory-saving) & plotting (prototyping)
        if output_enabled:
            # One-step memory
            t_sections = np.empty([1, x_len + 1])
            output_file = open(output, 'w')
            self._output_init(output_file, x_grid, t_sections[0])
        else:
            t_grid = grid_generator(t0, t_end, dt)
            t_len = t_grid.shape[0]
            t_sections = np.empty([t_len + 1, x_len + 1])

        t_sections[0] = np.array([t0, *discrete_initial])

        if plot is not None:
            fig, ax = plt.subplots()
            self._plot_init(plot, fig, ax, x0, x_max, discrete_initial)
            nframe = 0
            plot_t_sample_interval = round((
                (t_end - t0) / dt + 1
            ) / plot_t_samplesize)
            line, = ax.plot(x_grid, [0.] * x_len, linewidth=2)

        try:
            # <core>
            j = 0
            t = t0
            while t_end - t > 1e-16:
                # Always reach endpoint
                dt = min(dt, t_end - t)
                t += dt
                j += 1
                if output_enabled:
                    old_u = t_sections[-1][1:]
                    new_sect = np.empty(x_len + 1)
                else:
                    old_u = t_sections[j - 1][1:]
                    new_sect = t_sections[j]

                new_sect[0] = t
                new_u = new_sect[1:]

                for i, x in enumerate(x_grid):
                    a_value = self.a_func(x, t)
                    coeff = a_value * dt / dx

                    # Take care of boundaries first
                    if i in {0, x_len - 1}:
                        if self.boundary is not None:
                            boundary_index = 0 if i == 0 else 1
                            new_u[i] = self.boundary[boundary_index](t)
                        else:
                            try:
                                new_u[i] = self._u_generator_upwind(
                                    old_u, i, a_value, coeff
                                )
                            # Fixed boundary at upwind side
                            except IndexError:
                                new_u[i] = 0
                    else:
                        new_u[i] = self._u_generator_upwind(
                            old_u, i, a_value, coeff
                        )

                if output_enabled:
                    csv.writer(output_file).writerow(new_sect)
                    t_sections[-1] = new_sect
                # </core>

                if plot is not None:
                    if nframe % plot_t_sample_interval == 0:
                        line.set_ydata(new_u)
                        fig.savefig(plot + f'frame_{nframe:03d}.png')
                    nframe += 1

        except IndexError:
            raise ValueError("illegal velocity! `pde_solve_upwind` failed.")

        if output_enabled:
            output_file.close()
            return None
        else:
            # Remove (possible) trailing zeros
            t_sections = t_sections[:j + 1]

        return x_grid, t_sections


# <codecell>
if __name__ == '__main__':

    def square_input(center=-.15, half_width=.15):
        def distribution(x):
            return 1. if abs(x - center) <= half_width else 0.
        return distribution

    advecture = Advection1D(
        lambda x=0., t=0.: -1.,
        square_input,
        (-15., 0.)
    )
    advecture_solve_params = {
        'dt': .02,
        'dx': .05,
        't_end': 15.
    }
    filename_prefix = 'advecture' \
        + f'_dt{int(advecture_solve_params["dt"] * 100)}' \
        + f'_dx{int(advecture_solve_params["dx"] * 100)}'
    advecture.pde_solve_upwind(
        **advecture_solve_params,
        output=('../../csv/' + filename_prefix + '.csv'),
        plot=('../../csv/figs/' + filename_prefix + '/')
    )
