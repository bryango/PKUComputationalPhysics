#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from kruskal_zabusky import KruskalZabusky, SolitonVisualize


class KruskalZabuskyNew(KruskalZabusky):
    def pde_solve(self,
                  dt: float, dx: float,
                  t_end: float, t0=0.,
                  output=None) -> list:

        raise AttributeError('deprecated method; '
                             'use `pde_solve_fast` instead.')

    def pde_solve_fast(self, t_grid: list, xsteps: float) -> list:
        """ See parent docstring for more info. """

        # Some shorthands:
        x0, x_max = self.domain
        x_grid = np.linspace(x0, x_max, xsteps + 1)
        x_len = x_grid.shape[0]
        dx = (x_max - x0) / (x_len - 1)

        discrete_initial = list(map(self.initial_func, x_grid))

        # t_grid = np.linspace(t0, t_end, tsteps + 1)
        t_len = t_grid.shape[0]
        dt = (t_grid[-1] - t_grid[0]) / (t_len - 1)

        t_sections = np.empty([t_len, x_len + 1])
        t_sections[0] = np.array([t_grid[0], *discrete_initial])

        try:
            @jit(nopython=True, nogil=True, cache=True)
            def time_evolution(delta,
                               dt, t_grid,
                               dx, x_grid,
                               x_len, t_sections):

                deltasq = delta**2
                denominator1 = 3 * dx
                denominator2 = dx**3

                for j in range(1, t_len):
                    t = t_grid[j]
                    dt = t - t_grid[j - 1]

                    u = t_sections[j - 1][1:]
                    new_u = t_sections[j][1:]

                    t_sections[j][0] = t

                    # <scheme>
                    if j == 1:
                        for i in range(x_len):
                            new_u[i] = u[i] - dt * (
                                (
                                    (u[(i + 1) % x_len] + u[i] + u[i - 1])
                                    * (u[(i + 1) % x_len] - u[i - 1])
                                    / (2 * denominator1)
                                )
                                + deltasq * (
                                    u[(i + 2) % x_len] - 2 * u[(i + 1) % x_len]
                                    + 2 * u[i - 1] - u[i - 2]
                                ) / (2 * denominator2)
                            )
                    else:
                        old_old_u = t_sections[j - 2][1:]
                        for i in range(x_len):
                            new_u[i] = old_old_u[i] - dt * (
                                (
                                    (u[(i + 1) % x_len] + u[i] + u[i - 1])
                                    * (u[(i + 1) % x_len] - u[i - 1])
                                    / denominator1
                                )
                                + deltasq * (
                                    u[(i + 2) % x_len] - 2 * u[(i + 1) % x_len]
                                    + 2 * u[i - 1] - u[i - 2]
                                ) / denominator2
                            )
                    # </scheme>

                return t_sections

            t_sections = time_evolution(
                self.delta,
                dt, t_grid,
                dx, x_grid,
                x_len, t_sections
            )

        except IndexError:
            raise ValueError("illegal velocity! `pde_solve_fast` failed.")

        return x_grid, t_sections

    def pde_solve_ext(self, t_grid, x_grid, extend_from=None) -> list:

        t_sections = np.array(extend_from)
        x_len = x_grid.shape[0]
        t_len = t_sections.shape[0]

        if t_sections.shape != (t_len, x_len + 1):
            raise ValueError('illegal dataframe '
                             'for `pde_solve_ext`!')

        t_span = t_sections[-1][0] - t_sections[0][0]
        dt = t_span / t_len
        dx = x_grid[1] - x_grid[0]

        try:
            @jit(nopython=True, nogil=True, cache=True)
            def time_evolution_ext(delta,
                                   dt, t_grid,
                                   dx, x_grid,
                                   x_len, t_sections):

                deltasq = delta**2
                denominator1 = 3 * dx
                denominator2 = dx**3

                for j in range(t_len):
                    t = t_grid[j]
                    dt = t - t_sections[j - 1][0]

                    u = t_sections[j - 1][1:]
                    new_u = t_sections[j][1:]
                    old_old_u = t_sections[j - 2][1:]  # noqa: F841

                    t_sections[j][0] = t

                    # <scheme>
                    for i in range(x_len):

                        # new_u[i] = \
                        one_step = u[i] - dt * (  # noqa: F841
                            (
                                (u[(i + 1) % x_len] + u[i] + u[i - 1])
                                * (u[(i + 1) % x_len] - u[i - 1])
                                / (2 * denominator1)
                            )
                            + deltasq * (
                                u[(i + 2) % x_len] - 2 * u[(i + 1) % x_len]
                                + 2 * u[i - 1] - u[i - 2]
                            ) / (2 * denominator2)
                        )

                        # new_u[i] = \
                        two_step = old_old_u[i] - dt * (  # noqa: F841
                            (
                                (u[(i + 1) % x_len] + u[i] + u[i - 1])
                                * (u[(i + 1) % x_len] - u[i - 1])
                                / denominator1
                            )
                            + deltasq * (
                                u[(i + 2) % x_len] - 2 * u[(i + 1) % x_len]
                                + 2 * u[i - 1] - u[i - 2]
                            ) / denominator2
                        )

                        if abs(one_step) < abs(two_step):
                            new_u[i] = one_step
                        else:
                            new_u[i] = two_step
                        # </scheme>

                        # <catch_overflow>
                        if abs(new_u[i]) > 1e300:  # if overflows:
                            j -= 1  # Remove last t_section
                            print('PDE blew up! Time evolution terminated.')
                            converged = False
                            break
                    else:  # only executed if the inner loop did NOT break
                        converged = True
                        continue
                    break  # only executed if the inner loop DID break
                    # </catch_overflow>

                return t_sections, converged

            t_sections, converged = time_evolution_ext(
                self.delta,
                dt, t_grid,
                dx, x_grid,
                x_len, t_sections
            )

            if not converged:
                raise FloatingPointError('PDE blew up! '
                                         'Time evolution terminated.')

        except IndexError:
            raise ValueError("illegal velocity! `pde_solve_fast` failed.")

        return x_grid, t_sections


class SolitonVisualizeNew(SolitonVisualize):
    def __init__(self, working_dir='csv/', t_end=2.1,
                 large_t=False, memory_saving=False):

        self.large_t = large_t
        super().__init__(
            working_dir=working_dir,
            memory_saving=memory_saving
        )

        self.problem = KruskalZabuskyNew(
            delta=.022,
            domain=[0., 2.],
            initial_distributions=lambda x: np.cos(np.pi * x)
        )
        self.params = {
            'tsteps': 5e4,
            'xsteps': 128,
            't_end': t_end
        }  # Default params
        self._pathname_builder()

    def _pathname_builder(self):
        if not self.large_t:
            self.name = (lambda args: (
                'soliton_new'
                + f"_tsteps{int(args['tsteps'] / 1000)}k"
                + f"_xsteps{int(args['xsteps'])}"
            ))(self.params)
        else:
            self.name = 'soliton_large_t'
        self.plot_dir = self.working_dir + 'figs/' + self.name + '/'
        self.path = self.working_dir + self.name + '.csv'

    def solve_with_settings(self, output_enabled=False, t_max=3.,
                            extend=False, **solve_params):
        super(SolitonVisualize, self).solve_with_settings(
            output_enabled, **solve_params
        )
        self.params['tsteps'] = int(self.params['tsteps'])

        if not extend:
            # Generate uniform SUPER t_grid:
            try:
                tsteps = self.params['tsteps']
                t_end = self.params['t_end']
                t0 = self.params['t0']
            except KeyError:
                t0 = 0.

            t_span = t_end - t0
            dt = t_span / tsteps
            chunks = self.chunks = np.ceil(
                (t_max - t0 + dt) / (t_span + dt)
            ).astype('int')

            total_pts = (tsteps + 1) * chunks
            t_max = t0 + (t_span + dt) * chunks - dt
            self.super_t_grid = np.linspace(
                t0, t_max, total_pts
            ).reshape((chunks, tsteps + 1))

            t_grid = self.super_t_grid[0]
            self.x_grid, self.u_values_list \
                = (lambda args, domain: self.problem.pde_solve_fast(
                    t_grid=t_grid,
                    xsteps=args['xsteps']
                ))(self.params, self.problem.domain)
        else:
            try:
                self.ext_counter += 1
            except AttributeError:
                self.ext_counter = 1
            try:
                if self.ext_counter > self.chunks:
                    raise StopIteration('t_max reached! '
                                        'PDE extension stopped.')
                t_grid = self.super_t_grid[self.ext_counter]
                self.x_grid, self.u_values_list \
                    = self.problem.pde_solve_ext(
                        t_grid=t_grid,
                        x_grid=self.x_grid,
                        extend_from=self.u_values_list
                    )
            except AttributeError:
                raise AttributeError('PDE not solved yet; '
                                     'solve at least once '
                                     'before extending it. ')

        if output_enabled:
            self.csv_output()
        super().post_solving_cleanup()

    def visualize(self, t_sample_size=100,
                  pointsize=None, linewidth=2,
                  frame_counter_base=0,
                  gif_gen=False, show=False):
        """ Returns visualization;
            fix `including_end` = False, and default `show` = False.
        """

        super().visualize(
            t_sample_size=t_sample_size, including_end=False,
            pointsize=pointsize, linewidth=linewidth,
            frame_counter_base=frame_counter_base,
            gif_gen=gif_gen, show=show
        )
