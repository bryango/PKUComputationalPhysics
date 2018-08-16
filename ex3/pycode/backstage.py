#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import cos, pi
from IPython.display import display, Markdown
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# <codecell>
from qr_benchmark import qr_compare
from lattice_vibrations import lattice_max_mode


# <codecell>
class QRBench(object):
    def __init__(self):
        self.dataframe = pd.DataFrame(
            columns=['Householder', 'Givens', 'ratio']
        )
        self.dataframe.columns.name = '$n$ as dim'

    def add(self, dim, sample_size=5, runs=3, re_runs=2):
        time_householder, time_givens \
            = qr_compare(dim, sample_size, runs, re_runs)
        self.dataframe.loc[dim] \
            = {'Householder': time_householder,
               'Givens': time_givens,
               'ratio': time_givens / time_householder}
        return self.dataframe

    def plot(self):
        fig, ax = plt.subplots(dpi=80)
        self.dataframe[['Householder', 'Givens']].plot(
            loglog=True, marker='.', markersize=10, ax=ax)
        ax.legend(['Householder', 'Givens'])

        plt.xlabel(self.dataframe.columns.name, fontsize=18, labelpad=-5)
        plt.ylabel('Time in $\mathrm{sec}$', fontsize=18, labelpad=10)

        plt.tick_params(axis='both', which='major', labelsize=13.5, pad=5)
        plt.legend(loc='upper left', bbox_to_anchor=(.05, .935), fontsize=13.5)
        plt.title('QR decomposition: Average time',
                  fontsize=16, y=1.035)

    def fit(self, start_index=1):
        order_householder, order_givens = map(
            lambda qr_method: np.polyfit(*map(
                np.log10,
                [ self.dataframe.index[start_index:],
                  self.dataframe[qr_method][start_index:] ]
            ), deg=1),
            ['Householder', 'Givens'])
        print(
            'Estimated order: n = ',
            self.dataframe.index[start_index:].tolist(),
            ', ',
            sep=''
        )
        word_length = 18
        print('  -> Householder:'.ljust(word_length), order_householder[0])
        print('  -> Givens:'.ljust(word_length), order_givens[0])


# <codecell>
class LatticeMaxModes(object):
    def __init__(self):
        self.dataframe = pd.DataFrame(columns=['Eigenvalue', 'Eigenvector'])
        self.dataframe.columns.name = 'Length $N$'

    def show(self):
        return self.dataframe.style.format(
            {'Eigenvalue': '{:6f}'}
        )

    def add(self, length, show=True):
        eigenvalue, eigenvector = lattice_max_mode(length)
        self.dataframe.loc[length] \
            = {'Eigenvalue': eigenvalue,
               'Eigenvector': np.array2string(
                   np.array(eigenvector),
                   separator=', ')}
        return self.show() if show else self.dataframe

    def eigenvalue_calc(self, length):
        print(
            f'N = {length}, Eigenvalue (numerical result):',
            self.dataframe.loc[length]['Eigenvalue']
        )

    def eigenvalue_ref(self, length):
        display(Markdown(
            r"$\mrm{Ref}\colon\ "
            r"4 \cos^2 \pqty{\frac{\pi}{2N}} \big|_{\,N = 9} = %.16f $"
            % (4 * cos(pi / 2 / length)**2)
        ))

    def add_multiple(self, length_list):
        for n in length_list:
            self.add(int(n), show=False)
        self.dataframe.sort_index(inplace=True)
        return self.show()

    def eigenvalues_plot(self):
        fig, ax = plt.subplots(dpi=80)

        x = np.arange(1., 11., .1)
        y = (4 * np.cos(pi / 2 / x)**2)
        plt.plot(x, y, linestyle='--', label='Prediction for odd $N$')

        self.dataframe['Eigenvalue'].plot(
            style='o',
            marker='.', markersize=12, ax=ax, label='Numerical result')

        plt.xlabel(self.dataframe.columns.name, fontsize=16, labelpad=4.5)
        plt.ylabel(r'$\lambda_{\max}$', fontsize=22.5, labelpad=8)

        plt.tick_params(axis='both', which='major', labelsize=13.5, pad=5)
        plt.legend(loc='lower right', bbox_to_anchor=(.95, .08), fontsize=13.5)
        plt.title('Lattice vibrations: max eigenvalues',
                  fontsize=18, y=1.075)

    def eigenvector_list(self, length, max_size=float('inf'),
                         output=None):
        count = 0
        eigenvector_set = set()
        try:
            while count < max_size:
                count += 1
                eigenvalue, eigenvector \
                    = lattice_max_mode(length, notify=False)
                if not any(
                    x in eigenvector_set
                    for x in [eigenvector, tuple(np.negative(eigenvector))]
                ):
                    eigenvector_set.add(eigenvector)
                    if max_size == float('inf'):
                        print(eigenvector)
        except KeyboardInterrupt:
            pass

        if output is not None:
            with open(output, 'w') as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerow(['x', 'y', 'z'])
                for pt in eigenvector_set:
                    csv_writer.writerow(pt)
            return pd.read_csv(output)
        else:
            return eigenvector_set


def eigenvectors_plot_3d(file_path):
    fig = plt.figure(dpi=100,
                     num='Eigenvectors for N = 3')
    ax = fig.add_subplot(111, projection='3d')

    df = pd.read_csv(file_path)
    ax.scatter(df['x'], df['y'], df['z'])

    def construct_surface():
        random_indices = list(map(
            random.choice, [range(df.shape[0])] * 3
        ))
        anchor_pts = np.array(df.iloc[random_indices])
        a, b, c = normal = np.cross(
            anchor_pts[2] - anchor_pts[0],
            anchor_pts[1] - anchor_pts[0]
        )
        d = np.dot(normal, anchor_pts[0])
        x = y = np.arange(-1.2, 1.2, .05)
        x, y = np.meshgrid(x, y)
        z = (d - a * x - b * y) / c
        return x, y, z

    np.seterr(all='raise')
    while True:
        try:
            x, y, z = construct_surface()
            break
        except FloatingPointError:
            pass

    ax.plot_surface(x, y, z, alpha=0.1)

    ax.set_xlabel('$x$', fontsize=16, labelpad=8)
    ax.set_ylabel('$y$', fontsize=16, labelpad=8)
    ax.set_zlabel('$z$', fontsize=16, labelpad=-2)

    ax.grid(False)
    ax.view_init(45, -30)
    ax.tick_params(which='major', labelsize=10.5, pad=1)
    ticks_range = np.arange(-1., 1.1, .5)
    ax.set_xticks(ticks_range)
    ax.set_yticks(ticks_range)
    plt.tight_layout()
    plt.title('Lattice vibrations: '
              '$\lambda_{\max}$ eigenvectors for $N = 3$',
              fontsize=14.5, y=.925)
    plt.gcf().subplots_adjust(top=.95, bottom=.1)
