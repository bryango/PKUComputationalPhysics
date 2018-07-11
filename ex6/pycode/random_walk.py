#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


# <codecell>
# Native impletementation
@jit(nopython=True, nogil=True, cache=True)
def random_walk(walkers, summary, steps, prob_l, prob_r, means, varis,
                base_steps=0):
    walkers_count = walkers.size
    stays_count = 0
    # `summary` center index
    center_index = int(summary.size / 2)

    # <walk>
    for i in range(steps):
        # [0, 1) random reals for each walker
        rand_array = np.random.random_sample(walkers_count)
        for j in range(walkers_count):
            randnum = rand_array[j]
            if randnum < prob_l:  # Walk left
                walkers[j] -= 1
            elif randnum >= 1 - prob_r:  # Walk right
                walkers[j] += 1
            else:  # Stay
                stays_count += 1
        # </walk>

        # <integrated>
        if summary.size != 0:
            for j in range(walkers_count):
                summary[center_index + walkers[j]] += 1
            # </integrated>

            # Statistics
            for k in range(1, base_steps + i + 1):  # At most i steps
                means[base_steps + i] += k * (
                    summary[center_index + k] - summary[center_index - k]
                )  # small to large, numerical stable
            means[base_steps + i] /= (
                walkers_count * (base_steps + i + 1)
            )
            for k in range(1, base_steps + i + 1):
                varis[base_steps + i] += (
                    (k - means[i])**2 * summary[center_index + k]
                    + (-k - means[i])**2 * summary[center_index - k]
                )
            varis[base_steps + i] /= walkers_count * (base_steps + i + 1) - 1
        else:
            means[base_steps + i] = np.sum(walkers) / walkers_count
            # <variance>
            varis[base_steps + i] = np.sum(
                (walkers - means[i])**2
            ) / (walkers_count - 1)
            # </variance>

        # np.random.seed(randnum * 1e16)  # Re-seed (not necessary)

    if stays_count != 0:
        print("# of stays:", stays_count)


# Built-in implementations
# DEPRECATED!
@jit(nopython=True, nogil=True, cache=True)
def random_walk_binomial(walkers, steps, prob_l, prob_r):
    walkers_count = walkers.size

    if prob_l + prob_r != 1.:
        print('binomial impletementation: '
              'illegal left / right probability! '
              'left value used.')

    for i in range(steps):
        walkers += np.random.random_integers(1, .5, walkers_count) * 2 - 1


@jit(nopython=True, nogil=True, cache=True)
def random_walk_symmetric(walkers, steps, prob_l, prob_r):
    walkers_count = walkers.size

    if prob_l != .5 or prob_r != .5:
        print('symmetric impletementation: '
              'ignore left / right probability! ')

    for i in range(steps):
        walkers += np.random.random_integers(1, 2, size=walkers_count) * 2 - 1


# <codecell>
class RandomWalkers(object):

    def __init__(self, walkers_count, **kwargs):

        self.walkers_count = int(walkers_count)
        self.walkers = np.zeros(self.walkers_count, dtype=int)
        self.step_counter = 0
        # Statistics
        self.means, self.varis = np.empty([2, 0])

        if kwargs != {}:
            self.set_prob(**kwargs)

    def set_prob(self, left, right):

        self.p_left = left
        self.p_right = right
        p_stay = 1 - (left + right)

        if p_stay < 0:
            raise ValueError('illegal left / right probability!')

    def walk(self, steps, method=random_walk,
             from_zero=False, integrated=False):

        steps = int(steps)

        if self.step_counter != 0:  # Walk history found
            if not from_zero:  # Wish to resume
                if hasattr(self, 'summary'):  # integration history found
                    if not integrated:
                        print('Integration history found; '
                              'integrated mode enforced.')
                        integrated = True
                else:  # No integration history
                    if integrated:
                        print('Initiate integrated mode; '
                              'start from zero.')
                        integrated = True
                        from_zero = True

        if from_zero:
            self.__init__(self.walkers_count)
            for attr in ('summary'):
                if hasattr(self, attr):
                    delattr(self, attr)

        if integrated:
            new_summary = np.zeros(
                2 * (self.step_counter + steps) + 1, dtype=int
            )
            if hasattr(self, 'summary'):  # Import walk history
                new_summary[steps:-steps] = self.summary
            self.summary = new_summary
            summary_input = self.summary
        else:
            summary_input = np.empty(0, dtype=int)

        # Statistics
        self._means, self._varis = np.zeros([2, self.step_counter + steps])
        for attr in ('means', 'varis'):  # Import walk history
            new = getattr(self, '_' + attr)
            new[:self.step_counter] = getattr(self, attr)
            setattr(self, attr, new)
            delattr(self, '_' + attr)

        method(
            self.walkers, summary_input,
            steps, self.p_left, self.p_right, self.means, self.varis,
            base_steps=self.step_counter
        )

        self.step_counter += steps

        print('Steps:', self.step_counter)
        return self.walkers

    def count(self, integrated=False):
        if integrated:
            if hasattr(self, 'summary'):
                return (
                    np.arange(
                        -self.step_counter, self.step_counter + 1, 1,
                        dtype=int
                    ),
                    self.summary
                )
            else:
                print('WARNING: `integrated` = False')
                integrated = False
        if not integrated:
            return np.unique(self.walkers, return_counts=True)

    def covered_range(self):
        return np.take(
            self.count()[0], [0, -1]
        )

    def _pre_plot(self):
        fig, ax = plt.subplots(dpi=100, figsize=(5, 3.2))
        return fig, ax

    def _post_plot(self, integrated):
        title_style = {
            'fontsize': 13.5,
            'y': 1.024
        }
        if integrated:
            plt.title(
                f'Integrated distribution, {self.walkers_count} walkers',
                **title_style
            )
        else:
            plt.title(
                f'{self.step_counter}-step-only distribution, '
                f'{self.walkers_count} walkers',
                **title_style
            )

        farthest = max(self.covered_range())
        plt.xlim(-farthest, farthest)
        plt.tight_layout()

    def plot(self, normalize=False, integrated=False, **kwargs):
        plt_options = {
            's': 2
        }
        plt_options.update(kwargs)

        fig, ax = self._pre_plot()
        data = list(self.count(integrated=integrated))
        if normalize:
            data[1] = data[1] / data[1].sum()
        plt.scatter(
            *data,
            **plt_options
        )
        self._post_plot(integrated)

    def hist(self, integrated=False, bin_width=2, **kwargs):
        plt_options = {
            # 's': 2
        }
        plt_options.update(kwargs)

        fig, ax = self._pre_plot()
        if not integrated:
            plt.hist(
                self.walkers,
                np.arange(
                    *np.add(self.covered_range(), [- 1., 1.1]),
                    bin_width
                ),
                histtype='step'
            )
        else:
            plt.step(
                *self.count(integrated=True),
                **plt_options
            )

        self._post_plot(integrated)

    def stats_plot(self, **kwargs):
        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=100, figsize=(8, 3))
        subtitle_style = {
            'fontsize': 13.5,
            'y': 1.024
        }
        refline_style = {
            'linewidth': .6,
            'color': 'grey'
        }

        ax1.plot(self.varis)
        ax1.axvline(100, **refline_style)
        ax1.axhline(100, **refline_style)
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, max(ymax, 108))
        ax1.set_xlabel('steps')
        ax1.set_title(r'Variance $\sigma^2$', **subtitle_style)

        ax2.plot(self.means)
        ax2.axhline(0, **refline_style)
        ax2.set_xlabel('steps')
        ax2.set_title(r'Mean value', **subtitle_style)

        plt.suptitle(
            f'{self.walkers_count} walkers statistics',
            fontsize=16, y=1.015
        )
        plt.tight_layout()
