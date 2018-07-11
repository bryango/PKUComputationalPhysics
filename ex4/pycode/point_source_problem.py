#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import matplotlib.pyplot as plt
from backstage import csvPlot
from toolkit.dsolve import SimpleBVPbyShooting


# <codecell>
def point_source(center, half_width, charge):

    def distribution(x):
        return charge / (2 * half_width) \
            if abs(x - center) < half_width else 0.

    return distribution


class PointChargePotential(csvPlot):
    def __init__(self, file_path):
        self.params = {
            'half_width': .05
        }
        self.problem = SimpleBVPbyShooting(
            lambda half_width=self.params['half_width']: (
                lambda y: - y[2],
                lambda y: - point_source(
                    center=.4, half_width=half_width, charge=1.
                )(y[0])
            ),
            domain=(0., 1.),
            boundary_constraints=(
                ((1., 0., 0.),),
                (1., 0., 0.)
            )
        )
        self.columns = ['x', 'y', 'Dy']
        self.file_path = file_path
        super().__init__(
            columns=self.columns, file_path=file_path, header=None
        )

    def solve_with_settings(self, h=.02, first_guess=(2, 1.),
                            **kwargs):
        self.stepsize = h
        if kwargs != {}:
            self.params = kwargs
        pointset = self.problem.dsolve_shooting(
            h=h,
            first_guess=first_guess,
            **kwargs
        )
        with open(self.file_path, 'w') as output_file:
            for pt in pointset:
                csv.writer(output_file).writerow(pt)

    def visualize(self, pointsize=None, plot_Dy=False):
        if not plot_Dy:
            fig, ax = super().plot2d(['y'], pointsize=pointsize)
            plt.title('Point charge potential, '
                      f'stepsize $h = {self.stepsize}$\n'
                      fr'\large half width: ${self.params["half_width"]}$',
                      fontsize=16, y=1.035)
            plt.ylabel('$y$', fontsize=18, labelpad=10)
        else:
            fig, ax = super().plot2d(['Dy'], pointsize=pointsize)
            plt.title('Point charge field strength, '
                      f'stepsize $h = {self.stepsize}$\n'
                      fr'\large half width: ${self.params["half_width"]}$',
                      fontsize=16, y=1.035)
            plt.ylabel('$y$', fontsize=18, labelpad=10)

        plt.xlabel('$x$', fontsize=18, labelpad=4)
        ax.legend().set_visible(False)


# <codecell>
if __name__ == '__main__':
    pass
