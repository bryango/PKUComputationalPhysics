#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
import csv
import numpy as np
from toolkit.generic import grid_generator
from toolkit.pde import PDEBase
from backstage import PDESolve
from IPython.display import display, Markdown
from numba import jit


# <codecell>
class KruskalZabusky(PDEBase):
    """ PDE: Kruskal, Zabusky soliton, (non-linear) equation:
            ( D_t + [u(x, t)] * D_x + (delta)**2 * (D_x)**3 ) [u(x, t)] = 0,
        defined in a PERIODIC x-domain, with real parameter `delta`.

    Args:

        delta=.022:
            single real parameter for the equation;

        domain=[0., 2.]:
            interval (start, end);

        initial_distributions:
            PERIODIC function on specified domain.

    NOTE: `initial_distributions` can be set when initialized,
        OR set later by `set_initial()` methods.
    """

    def __init__(self, delta=.022,
                 domain=[0., 2.],
                 initial_distributions=lambda x: np.cos(np.pi * x)):
        self.delta = delta
        self.domain = self._check_domain(domain)
        self.initial_func = None if initial_distributions is None else \
            initial_distributions

    def _u_generator_default(self, old_u, i, dt, coef1, coef2,
                             plus1=lambda i: i + 1,
                             plus2=lambda i: i + 2):
        """ Returns value at the next time node,
            using the default scheme.
            NOTE: `plus1` & `plus2` can be hacked for boundary situation,
        """
        u = old_u
        return u[i] - dt * (
            coef1 * (
                (u[plus1(i)] + u[i] + u[i - 1]) * (u[plus1(i)] - u[i - 1])
            )
            + coef2 * (
                u[plus2(i)] - 2 * u[plus1(i)] + 2 * u[i - 1] - u[i - 2]
            )
        )

    def pde_solve(self,
                  dt: float, dx: float,
                  t_end: float, t0=0.,
                  output=None) -> list:
        """ Solve using specified scheme.

        Args:

            output: write to specified csv file path in real time,
                SLOW but memory-saving;

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
                             'for `pde_solve`!')

        # Some shorthands:
        output_enabled = (output is not None)
        x0, x_max = self.domain
        func = self.initial_func
        x_grid = grid_generator(x0, x_max, dx)
        x_len = x_grid.shape[0]

        discrete_initial = list(map(func, x_grid))

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

        try:
            # <core>
            j = 0
            t = t0
            coef1 = (1 / 6) / dx
            coef2 = (self.delta**2 / 2) / dx**3
            np.seterr(all='raise')
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

                # <catch_overflow>
                try:
                    for i, x in enumerate(x_grid):
                        new_u[i] = self._u_generator_default(
                            old_u, i, dt, coef1, coef2,
                            plus1=lambda i: (i + 1) % x_len,
                            plus2=lambda i: (i + 2) % x_len,
                        )  # Automatically ensures periodic boundary.
                except FloatingPointError:
                    print('PDE blew up! Time evolution terminated.')
                    break
                # </catch_overflow>

                if output_enabled:
                    csv.writer(output_file).writerow(new_sect)
                    t_sections[-1] = new_sect
                # </core>

        except IndexError:
            raise ValueError("illegal velocity! `pde_solve` failed.")

        if output_enabled:
            output_file.close()
            return None
        else:
            # Remove (possible) trailing zeros
            t_sections = t_sections[:j + 1]

        return x_grid, t_sections

    def pde_solve_fast(self,
                       dt: float, dx: float,
                       t_end: float, t0=0.) -> list:
        """ Basic `pde_solve` with numba jit acceleration """

        if (t_end < t0) or (dt < 0):
            # Time reversal (dt < 0) not supported yet!
            raise ValueError('inconsistent stepsize & endpoint'
                             'for `pde_solve`!')

        # Some shorthands:
        x0, x_max = self.domain
        func = self.initial_func
        x_grid = grid_generator(x0, x_max, dx)
        x_len = x_grid.shape[0]
        t_grid = grid_generator(t0, t_end, dt)
        t_len = t_grid.shape[0]

        discrete_initial = list(map(func, x_grid))
        t_sections = np.empty([t_len, x_len + 1])
        t_sections[0] = np.array([t0, *discrete_initial])

        try:
            @jit(nopython=True, nogil=True, cache=True)
            def time_evolution(delta,
                               dt, t_grid,
                               dx, x_grid,
                               x_len, t_sections):
                t = t_grid[0]
                coef1 = (1 / 6) / dx
                coef2 = (delta**2 / 2) / dx**3
                for j in range(1, t_len):
                    t = t_grid[j]
                    dt = t - t_grid[j - 1]

                    u = t_sections[j - 1][1:]
                    new_u = t_sections[j][1:]

                    t_sections[j][0] = t

                    for i in range(x_len):
                        new_u[i] = u[i] - dt * (
                            coef1 * (
                                (u[(i + 1) % x_len] + u[i] + u[i - 1])
                                * (u[(i + 1) % x_len] - u[i - 1])
                            )
                            + coef2 * (
                                u[(i + 2) % x_len] - 2 * u[(i + 1) % x_len]
                                + 2 * u[i - 1] - u[i - 2]
                            )
                        )
                        # <catch_overflow>
                        if abs(new_u[i]) > 1e300:  # if overflows:
                            j -= 1  # Remove last t_section
                            print('PDE blew up! Time evolution terminated.')
                            break
                    else:  # only executed if the inner loop did NOT break
                        continue
                    break  # only executed if the inner loop DID break
                    # </catch_overflow>

                return t_sections[:j + 1]

            t_sections = time_evolution(
                self.delta,
                dt, t_grid, dx, x_grid, x_len, t_sections
            )

        except IndexError:
            raise ValueError("illegal velocity! `pde_solve` failed.")

        return x_grid, t_sections


# <codecell>
class SolitonVisualize(PDESolve):
    def __init__(self, working_dir='csv/', memory_saving=False):
        super().__init__(
            working_dir=working_dir,
            memory_saving=memory_saving
        )

        self.problem = KruskalZabusky(
            delta=.022,
            domain=[0., 2.],
            initial_distributions=lambda x: np.cos(np.pi * x)
        )
        self.params = {
            'tsteps': 5e4,
            'xsteps': 128,
            't_end': 2.1
        }  # Default params
        self._pathname_builder()

    def _pathname_builder(self):
        self.name = (lambda args: (
            'soliton'
            + f"_tsteps{int(args['tsteps'] / 1000)}k"
            + f"_xsteps{int(args['xsteps'])}"
        ))(self.params)
        self.plot_dir = self.working_dir + 'figs/' + self.name + '/'
        self.path = self.working_dir + self.name + '.csv'

    def solve_with_settings(self, output_enabled=False,
                            fast=True, extend=False,
                            **solve_params):
        super().solve_with_settings(output_enabled, **solve_params)

        solve_method = self.problem.pde_solve_fast if fast else \
            self.problem.pde_solve
        self.x_grid, self.u_values_list \
            = (lambda args, domain: solve_method(
                t_end=args['t_end'],
                dt=args['t_end'] / args['tsteps'],
                dx=(domain[1] - domain[0]) / args['xsteps']
            ))(self.params, self.problem.domain)

        if output_enabled:
            self.csv_output()
        super().post_solving_cleanup()

    def visualize(self,
                  t_sample_size=100, including_end=True,
                  pointsize=None,
                  linewidth=2, frame_counter_base=0,
                  gif_gen=False, show=True):
        try:
            super().visualize(
                x_range=[0., 2.], y_range=[-2., 3.6],
                t_sample_size=t_sample_size,
                including_end=including_end,
                pointsize=pointsize, linewidth=linewidth,
                frame_counter_base=frame_counter_base,
                gif_gen=gif_gen, show=show
            )
            if show:
                display(Markdown(
                    '$t$ `steps: '
                    f'{self.params["tsteps"]}`, '
                    '$x$ `steps: '
                    f'{self.params["xsteps"]}`, '
                ))
                display(Markdown(
                    '![](' + self.plot_dir.strip("/") + '.gif' + ')'
                ))
        except AttributeError:
            raise AttributeError('PDE not solved yet, '
                                 'please invoke `solve_with_settings` '
                                 'before visualization. ')


if __name__ == '__main__':
    pass
