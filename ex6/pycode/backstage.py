#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
import gc
import csv
import os
import platform
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# <codecell>
class csvPlotBase(object):
    def __init__(self, columns, file_path=None, header=None):
        self.path = file_path
        self.variable = columns[0]
        self.columns = columns[1:]
        self.read_header = header

    def _renew_path(self, file_path=None, header=None):
        if file_path is None:
            file_path = self.path
        else:
            self.path = file_path
        self.read_header = header


class csvPlot(csvPlotBase):
    def __init__(self, columns, file_path=None, header=None):
        super().__init__(columns, file_path, header)
        self.update(file_path, header)

    def update(self, file_path=None, header=None):
        super()._renew_path(file_path, header)
        if self.path is not None:
            self.dataframe = pd.read_csv(
                self.path, header=self.read_header, index_col=0
            )
            self.dataframe.columns = self.columns
            self.dataframe.index.name = self.variable

    def plot2d(self, column, pointsize=None, update=True):
        if update:
            self.update()
        fig, ax = plt.subplots(dpi=80)
        self.dataframe[column].plot(
            style='.', markersize=pointsize, ax=ax)

        plt.tight_layout()
        return fig, ax

    def parametric_plot3d(self, three_columns, join=False, update=True):
        if update:
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


class csvPlotPDE1d(csvPlotBase):
    # NOTE: csv formatted according to `toolkit.pde`
    def __init__(self, columns=['x', 'u'],
                 file_path=None):

        # Check if column names are valid:
        if np.array(columns).shape != (2,):
            raise ValueError('only support 1D (spatial) PDE, '
                             "namely columns ~ ['x', 'u(x, t)']")
        else:
            self.dim = len(columns)

        # Basic attributes for a t-section
        super().__init__(columns, file_path, header=None)
        # Only one column for a t-section
        self.column = self.columns[0]
        self.update(file_path)

    def _trim_master_df(self):
        try:
            # NOTE: `set_index` triggers a COPY! [memory-leaking]
            # self.master_df.set_index(
            #     self.master_df.columns[0],
            #     inplace=True
            # )
            # self.master_df.index.name = 't'
            self.master_df.columns.name = self.variable
            self.t_grid = np.array(self.master_df['t'])
        except NameError:
            pass

    def update(self, file_path=None):
        # `header` option is invalid for `csvPlotPDE1d.update`
        super()._renew_path(file_path, header=None)
        if self.path is not None:
            # Works only if csv is formatted according to `toolkit.pde`
            self.master_df = pd.read_csv(
                self.path,
                header=None, dtype=float, skiprows=[0, 1]
            )
            self.master_df.columns = pd.read_csv(
                self.path,
                header=None, nrows=1
            ).iloc[0]
            self.master_df.columns = ['t', *self.master_df.columns[1:]]
            self._trim_master_df()

    def t_section_by_index(self, t_index):
        """ Create a t-section by index. """
        self.dataframe = pd.DataFrame(self.master_df.iloc[t_index])[1:]
        self.dataframe.columns = self.columns
        self.dataframe.index.name = self.variable

    def t_section(self, t, show=False):
        """Create a t-section by t value. """
        self.t_index = np.searchsorted(self.t_grid, t)
        self.t_section_by_index(self.t_index)
        if show:
            return self.t_index, self.dataframe

    def _t_samples(self, sample_size, including_end=True):
        t0, t_end = np.take(self.t_grid, [0, -1])

        if including_end:
            perfect_samples = np.linspace(t0, t_end, sample_size)
        else:
            perfect_samples = np.linspace(t0, t_end, sample_size + 1)[:-1]

        approx_indices = list(map(
            lambda t: np.searchsorted(self.t_grid, t),
            perfect_samples
        ))
        return np.take(self.t_grid, approx_indices)

    def _plot_init(self, ax, x_range, y_range, update,
                   export_dir=None):

        if update:
            self.update()
        if export_dir is not None:
            try:
                os.makedirs(export_dir, exist_ok=False)
                print('makedir:', export_dir)
            except OSError:
                pass

        ax.set_xlabel('$' + self.variable + '$')
        ax.set_ylabel('$' + self.column + '$')
        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_ylim(y_range)

    def _section_plot(self, line, t, title):
        self.t_section(t)
        t_taken = self.t_grid[self.t_index]
        line.set_ydata(np.array(self.dataframe))
        plt.setp(title, text=f'$t = {t_taken:.2f}$')
        plt.tight_layout()

    def export_plots(self, export_dir, t_sample_size=100,
                     including_end=True,
                     x_range=None, y_range=None,
                     linewidth=2, frame_counter_base=0,
                     update=False):

        fig, ax = plt.subplots()
        self._plot_init(ax, x_range, y_range,
                        update=update, export_dir=export_dir)

        line, = (lambda x_values: ax.plot(
            x_values, [0.] * len(x_values),
            linewidth=linewidth
        ))(self.master_df.columns[1:])
        title = plt.title('init')

        for index, t in enumerate(
            self._t_samples(t_sample_size, including_end=including_end)
        ):
            frame_counter = frame_counter_base + index
            self._section_plot(line, t, title)
            fig.savefig(export_dir + f'frame_{frame_counter:03d}.png')

        plt.close(fig)


class PDESolve(csvPlotPDE1d):
    def __init__(self, working_dir='csv/', memory_saving=False):
        self.working_dir = working_dir

        super().__init__(
            columns=['x', 'u'],
            file_path=None,
        )  # initialize `self.path` with None, change later
        self.memory_saving = memory_saving

    def solve_with_settings(self, output_enabled=False, **solve_params):
        if solve_params != {}:
            self.params.update(solve_params)
        self._pathname_builder()
        # Define `self.u_values_list` in child class

    def post_solving_cleanup(self):
        if self.memory_saving:
            del self.u_values_list
            self.update()
        else:
            self.master_df = pd.DataFrame(self.u_values_list, copy=False)
            self.master_df.columns = ['t', *self.x_grid]
            self._trim_master_df()
        gc.collect()

    def csv_output(self):
        with open(self.path, 'w') as output_file:
            csv.writer(output_file).writerows([
                ['x_grid', *self.x_grid],
                ['t', 'u']
            ])
            csv.writer(output_file).writerows(self.u_values_list)

    def visualize(self, x_range, y_range,
                  t_sample_size=100, including_end=True,
                  pointsize=None, linewidth=2,
                  frame_counter_base=0,
                  gif_gen=False, show=True):
        if self.memory_saving:
            super().update(self.path)
            update = True
        else:
            update = False

        super().export_plots(self.plot_dir, t_sample_size,
                             including_end=including_end,
                             x_range=x_range, y_range=y_range,
                             linewidth=2,
                             frame_counter_base=frame_counter_base,
                             update=update)
        if gif_gen:
            if platform.system() != 'Windows':
                try:
                    print('Generating GIF by `convert` command...')
                    get_ipython().run_cell_magic(  # noqa: F821
                        'bash', '',
                        'cd ' + self.plot_dir + '\n'
                        'convert frame_* ../"${PWD##*/}".gif'
                    )
                except:  # noqa: E722
                    print('Unsuccessful GIF generation, '
                          'showing archived data instead.')
            else:
                print('Non-UNIX environment detected, '
                      'showing archived GIF instead of re-generating.')
        else:
            print('`gif_gen` is off; using archived GIF. ')


# <codecell>
if __name__ == '__main__':
    test = csvPlotPDE1d(
        file_path='../csv/advecture_dt2_dx5.csv'
    )
    test.plot(
        '../csv/figs/advecture_dt2_dx5/',
        y_range=[0, 1.05]
    )
