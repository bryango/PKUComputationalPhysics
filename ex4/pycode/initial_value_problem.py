#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
from backstage import csvPlot
from toolkit.dsolve import InitialValueProblem
import matplotlib.pyplot as plt


# <codecell>
class MagneticCylic(csvPlot):
    def __init__(self, file_path):
        self.problem = InitialValueProblem(
            lambda omega=1.: (
                lambda y: y[4],
                lambda y: y[5],
                lambda y: y[6],
                lambda y: omega * y[5],
                lambda y: - omega * y[4],
                lambda y: 0
            ),
            initial_values=(0., 0., 0., 0., 0., 2., 0.1)
        )
        self.columns = ['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z']
        self.file_path = file_path
        super().__init__(
            columns=self.columns, file_path=file_path, header=None
        )

    def solve_with_settings(self, nsteps, **kwargs):
        self.nsteps = nsteps
        pointset = self.problem.dsolve_rk4(
            h=(40. / nsteps),
            endpoint=40.,
            **kwargs
        )
        with open(self.file_path, 'w') as output_file:
            for pt in pointset:
                csv.writer(output_file).writerow(pt)

    def visualize(self, angle=(15, -48)):
        fig, ax = super().parametric_plot3d(['x', 'y', 'z'], join=True)
        ax.view_init(*angle)
        plt.title(f'Magnetic cyclic with $N = {self.nsteps}$ steps',
                  fontsize=16)
        # plt.gcf().subplots_adjust(left=None)


# <codecell>
class EqWithExternalForce(csvPlot):
    def __init__(self, file_path):
        self.problem = InitialValueProblem(
            lambda: (
                lambda y: y[2],
                lambda y: np.exp(2 * y[0]) * np.sin(y[0]) + 2 * (y[2] - y[1])
            ),
            initial_values=(0., -.4, -.6)
        )
        self.columns = ['t', 'y', 'v_y']
        self.file_path = file_path
        super().__init__(
            columns=self.columns, file_path=file_path, header=None
        )

    def solve_with_settings(self, nsteps, endpoint=1., **kwargs):
        self.nsteps = nsteps
        self.endpoint = endpoint
        pointset = self.problem.dsolve_rk4(
            h=(1. / nsteps),
            endpoint=endpoint,
            **kwargs
        )
        with open(self.file_path, 'w') as output_file:
            for pt in pointset:
                csv.writer(output_file).writerow(pt)

    def visualize(self, logy=False):
        fig, ax = super().plot2d(['y'])

        # Exact solution
        t_data = self.dataframe.index
        t = np.linspace(t_data.min(), t_data.max(), 300)
        y = (1 / 5) * np.exp(2 * t) * (-2 * np.cos(t) + np.sin(t))
        ax.plot(t, y, linestyle='--', zorder=0, label='Exact')

        plt.title(f'Motion with driving force, $N = {self.nsteps}$ steps '
                  f'in domain $[{self.problem.initial[0]}, {self.endpoint}]$',
                  fontsize=16, y=1.035)
        plt.xlabel('$t$', fontsize=18, labelpad=4)
        plt.ylabel('$y$', fontsize=18, labelpad=10)
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(.5, .95),
            fontsize=12
        ).get_texts()[0].set_text('Numerical')

        if logy:
            plt.yscale('symlog')
            plt.legend(
                loc='upper left',
                bbox_to_anchor=(.05, .95),
                fontsize=12
            ).get_texts()[0].set_text('Numerical')

            ticks_log = np.linspace(8, 32, 32 / 8)
            ax.set_yticks(np.concatenate((
                - np.power(10, np.flipud(ticks_log)),
                (0,),
                np.power(10, ticks_log)
            )))


# <codecell>
class LorenzAttractor(csvPlot):
    def __init__(self, file_path):
        self.params = {
            'sigma': 10,
            'rho': 28,
            'beta': 5 / 3
        }
        self.problem = InitialValueProblem(
            lambda
            sigma=self.params['sigma'],
            rho=self.params['rho'],
            beta=self.params['beta']: (
                lambda y: - beta * y[1] + y[2] * y[3],
                lambda y: - sigma * y[2] + sigma * y[3],
                lambda y: - y[2] * y[1] + rho * y[2] - y[3]
            ),
            initial_values=(0., 12., 4., 0.)
        )
        self.columns = ['t', 'x', 'y', 'z']
        self.file_path = file_path
        super().__init__(
            columns=self.columns, file_path=file_path, header=None
        )

    def solve_with_settings(self, nsteps=1000, **kwargs):
        self.nsteps = nsteps
        if kwargs != {}:
            self.params = kwargs
        pointset = self.problem.dsolve_rk4(
            h=(10. / nsteps),
            endpoint=10.,
            **kwargs
        )
        with open(self.file_path, 'w') as output_file:
            for pt in pointset:
                csv.writer(output_file).writerow(pt)

    def visualize(self, angle=(36, -45)):
        fig, ax = super().parametric_plot3d(['x', 'y', 'z'])
        ax.view_init(*angle)
        plt.title(f'Lorenz attractor with $N = {self.nsteps}$ steps\n'
                  r'\large'
                  fr'$\sigma = {self.params["sigma"]},\,'
                  fr'\rho = {self.params["rho"]},\,'
                  fr'\beta \simeq {self.params["beta"]:.3f}$',
                  fontsize=16)

if __name__ == '__main__':
    pass
