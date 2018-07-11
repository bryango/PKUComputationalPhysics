#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
import csv
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from IPython.display import display, Markdown
from math import cos, pi


# <codecell>
class csvPlot(object):
    def __init__(self, columns,
                 file_path=None, header=None):
        self.path = file_path
        self.variable = columns[0]
        self.columns = columns[1:]
        self.update(file_path, header)

    def update(self, file_path=None, header=None):
        if file_path is None:
            file_path = self.path
        if file_path is not None:
            self.path = file_path
        self.dataframe = pd.read_csv(
            self.path, header=header, index_col=0
        )
        self.dataframe.columns = self.columns
        self.dataframe.index.name = self.variable

    def plot2d(self, columns, pointsize=None):
        self.update()
        fig, ax = plt.subplots(dpi=80)
        self.dataframe[columns].plot(
            style='.', markersize=pointsize, ax=ax)

        plt.tight_layout()
        return fig, ax

    def parametric_plot3d(self, three_columns, join=False):
        self.update()
        fig = plt.figure(dpi=100,
                         num='3D Plot')
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        t_data = self.dataframe.index
        plot_pts = list(map(
            self.dataframe.__getitem__, three_columns
        ))
        ax.scatter(*plot_pts, s=16, c=t_data, cmap='coolwarm')

        # Smooth line:
        if join:
            t_resampled = np.linspace(t_data.min(), t_data.max(), 300)
            ax.plot(*list(map(
                lambda xi: interp1d(t_data, xi, kind='cubic')(t_resampled),
                plot_pts
            )), linewidth=.5)
            plt.tight_layout()

        ax.set_xlabel('$x$', fontsize=16, labelpad=8)
        ax.set_ylabel('$y$', fontsize=16, labelpad=8)
        ax.set_zlabel('$z$', fontsize=16, labelpad=2)
        plt.tight_layout()

        return fig, ax


# <codecell>
if __name__ == '__main__':
    plot = csvPlot(
        columns=['t', 'x', 'y', 'z', 'v_x', 'v_y', 'v_z'],
        file_path='../csv/magnetic_cyclic.csv'
    )
    plot.parametric_plot3d(['x', 'y', 'z'])

# <codecell>
if __name__ == '__main__':
    plot = csvPlot(
        columns=['t', 'x', 'y', 'z'],
        file_path='../csv/lorenz_attractor.csv'
    )
    plot.parametric_plot3d(['x', 'y', 'z'])

# <codecell>
if __name__ == '__main__':
    plot = csvPlot(
        columns=['t', 'x', 'v_x'],
        file_path='../csv/eq_with_external_force.csv'
    )
    plot.plot2d(['x'])
