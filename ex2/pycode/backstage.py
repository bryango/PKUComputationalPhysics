#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
from toolkit.generic import relative_error, transpose, mrange
from toolkit.rootfinder import find_root_secant
from IPython.display import display, Markdown
from zeta_study import zeta, eps, phase_shift


# <codecell>
def runge(x: float):
    return 1 / (1 + 25 * x**2)


x_list = num.arange(-1., 1.1, .1).tolist()
y_list = [runge(x) for x in x_list]

x_plot = num.arange(-1., 1., 0.01)
y_ref = [runge(x) for x in x_plot]


def interp_test(interpol_function, title: str, y_range, error_range):
    """ Interpolation test of `interpol_function()`,
        :param title: displayed title of graph;
        :param y_range: plot range of y;
        :param error_range: plot range of dy / y;
        :return: error list for future use.
    """

    y_interp = [interpol_function(x) for x in x_plot]
    y_error = list(map(lambda x: relative_error(*x),
        [ [y_interp[i], y_ref[i]] for i in range(len(y_ref)) ]    # noqa: E128
    ))

    plt.subplots(nrows=1, ncols=2, figsize=(8, 3), dpi=100)

    plt.subplot(121)
    plt.plot(x_plot, y_ref)
    plt.plot(x_plot, y_interp)
    plt.scatter(x_list, y_list, s=12, zorder=10)
    plt.grid(linestyle='dotted')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.title(title, fontsize=13.5, y=1.035)
    plt.ylim(y_range)

    plt.subplot(122)
    plt.plot(x_plot, y_error)
    plt.grid(linestyle='dotted')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$\Delta y / y$', fontsize=13.5)
    plt.title(f"Relative error, plot range ${error_range}$",
              fontsize=13.5, y=1.035)
    plt.ylim(error_range)

    plt.tight_layout()
    plt.subplots_adjust(wspace=.35)
    plt.show()
    return y_error


def ticks_range(x_plot, h):
    return num.arange(min(x_plot), max(x_plot) + h, h)


def compare_error(y_dict, title: str, y_range) -> None:
    """ Compare relative error of `interpol_function()`,
        :param y_dict: relative error of each interpolation method,
            comined in dict; e.g. 'Method': y_data_list;
        :param title: displayed title of graph;
        :param y_range: plot range of y.
    """
    fig, ax = plt.subplots(dpi=100)
    plt_dict = {}
    for key, y_list in y_dict.items():
        plt_dict.update({key: plt.plot(x_plot, y_list, label=key)})
    plt.setp(plt_dict.values(), linewidth=1.2)
    plt.setp(plt_dict['Neville'], linewidth=.8, linestyle='--')
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$\Delta y / y$', fontsize=13.5)
    plt.title(title, fontsize=13.5, y=1.035)
    plt.ylim(y_range)

    plt.title(f"Compare {title}", fontsize=13.5, y=1.025)
    ax.set_xticks(ticks_range(x_plot, .2))
    ax.set_xticks(ticks_range(x_plot, .1), minor=True)
    plt.grid(linestyle='dotted')
    ax.xaxis.grid(False)
    plt.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(1.28, 1))
    plt.show()
    return None


x_out = num.arange(-1., 1.05, .05).tolist()
fx_out = [runge(x) for x in x_out]


def interp_test_output(interpol_function, name: str, filename: str):
    """ Interpolation test data output;
        :params name: displayed function name of `interpol_function`,
            math-styled;
        :params filename: output csv filename.
    """
    interp_out = [interpol_function(x) for x in x_out]
    error_out = list(map(lambda x, y: abs(x - y), *[fx_out, interp_out]))
    data = pd.DataFrame(
        transpose( [x_out, fx_out, interp_out, error_out] ),
        columns=['$x$', '$f(x)$', f'${name}(x)$', f'$|\,{name} - f\,|$']
    )
    display(data[:10])
    data.columns = ['x', 'f(x)', f'{name}(x)', f'|{name}-f|']
    data.to_csv(f'csv/{filename}.csv')
    display(Markdown(' ...... *more in* '
                     f'[`csv/{filename}.csv`](csv/{filename}.csv) '))
    return None


def cardioid(phi: float):
    return 1. - num.cos(phi)


def phi_t(t):
    return t * num.pi / 4


t_range = mrange(0, 8)
t_list = list(t_range)

xt_list = [ cardioid(phi_t(t)) * num.cos(phi_t(t)) for t in t_list]
yt_list = [ cardioid(phi_t(t)) * num.sin(phi_t(t)) for t in t_list]


def cardioid_graph(title: str, sx, sy):
    t_plot = num.arange(0., 8. + .01, .01)
    xt_ref = [ cardioid(phi_t(t)) * num.cos(phi_t(t)) for t in t_plot]
    yt_ref = [ cardioid(phi_t(t)) * num.sin(phi_t(t)) for t in t_plot]

    xt_interp = [ sx(t) for t in t_plot ]
    yt_interp = [ sy(t) for t in t_plot ]

    plt.subplots(figsize=(4.5, 4.5), dpi=100)

    plt.plot(xt_ref, yt_ref)
    plt.plot(xt_interp, yt_interp)
    plt.scatter(xt_list, yt_list, s=16)
    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$y$', fontsize=16)
    plt.title(title, fontsize=13.5, y=1.035)

    plt.tight_layout()
    plt.show()
    return None


q2_plot = num.arange(.001, 3.01, .01).tolist()
size = len(q2_plot)
zeta_zero = find_root_secant(
    lambda x: zeta(x), 0.3, 0.6,
    accuracy=10**(-16))


def machine_limit_at(x):
    return abs(num.nextafter(x, 1) - x)


def zeta_eps_plot(n_max_alpha, n_max_beta,
                  k_max, int_order,
                  y_range='auto', error_range=[1e-20, 1e2]):

    params = [n_max_alpha, n_max_beta, k_max, int_order]
    zeta_values = [zeta(q2, *params) for q2 in q2_plot]
    machine_limit_list = [ machine_limit_at(value)
                           for value in zeta_values ]
    digits_6 = [ abs(zeta_values[i]) * 10**(-6) for i in range(size) ]
    digits_12 = [ abs(zeta_values[i]) * 10**(-12) for i in range(size) ]

    # Absolute error, via eps()
    zeta_eps_details = [eps(q2, *params, opt='detailed') for q2 in q2_plot]
    zeta_eps_partial = (lambda part:
        [ abs(zeta_eps_details[i][1].get(part))
          for i in range(size) ] )   # noqa: E128

    # Truncate value list
    # AFTER data preparation
    upper = 1000.
    lower = -1000.
    for i in range(size):
        if zeta_values[i] > upper:
            zeta_values[i] = num.inf
        if zeta_values[i] < lower:
            zeta_values[i] = -num.inf

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.75), dpi=100)

    plt.subplot(121)
    plt.plot(q2_plot, zeta_values)
    plt.grid(linestyle='dotted')
    plt.xlabel('$q^2$', fontsize=12)
    plt.title(r'$\mathcal{Z}_{00}\,(1;q^2)$', fontsize=13.5, y=1.035)
    if y_range != 'auto':
        plt.ylim(y_range)

    plt.subplot(122)
    plt.semilogy(q2_plot, machine_limit_list,
                 label='Machine limit', linewidth=.8,
                 color='tab:gray')
    plt.semilogy(q2_plot, digits_6,
                 label='6 digits', linewidth=.8,
                 linestyle='--', color='tab:brown')
    plt.semilogy(q2_plot, digits_12,
                 label='12 digits', linewidth=.8,
                 linestyle='--', color='tab:purple')
    plt.plot((zeta_zero, zeta_zero), (lower, upper),
             color='tab:purple', linewidth=1.,
             label='First zero', linestyle='dotted')

    for key in ['$k$', r'$\alpha$', r'$\beta$']:
        plt.semilogy(q2_plot, zeta_eps_partial(key), label=key)

    plt.xlabel('$q^2$', fontsize=12)
    plt.title(r"Various contributions of $\tilde{\epsilon}$\\[.5ex] "
              fr"\small for $m = {n_max_alpha},\, "
              fr"m' = {n_max_beta},\, k = {k_max}$, "
              fr"integral order {int_order}",
              fontsize=13.5, y=1.035)
    plt.ylim(*error_range)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.035))

    plt.tight_layout()
    plt.subplots_adjust(wspace=.25)
    plt.show()
    return None


q2_plot_mini = num.arange(.001, .302, .002).tolist()
size_mini = len(q2_plot_mini)


def zeta_eps_plot_mini(n_max_alpha, n_max_beta,
                       k_max, int_order,
                       y_range='auto', error_range=[1e-20, 1e2]):

    params = [n_max_alpha, n_max_beta, k_max, int_order]
    zeta_values = [zeta(q2, *params) for q2 in q2_plot_mini]
    machine_limit_list = [ machine_limit_at(value)
                           for value in zeta_values ]
    digits_6 = [ abs(zeta_values[i]) * 10**(-6) for i in range(size_mini) ]
    digits_12 = [ abs(zeta_values[i]) * 10**(-12) for i in range(size_mini) ]

    # Absolute error, via eps()
    zeta_eps_details = [eps(q2, *params, opt='detailed')
                        for q2 in q2_plot_mini]
    zeta_eps_partial = (lambda part:
        [ abs(zeta_eps_details[i][1].get(part))
          for i in range(size_mini) ] )   # noqa: E128

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), dpi=100)

    plt.subplot(121)
    plt.plot(q2_plot_mini, zeta_values)
    plt.grid(linestyle='dotted')
    plt.xlabel('$q^2$', fontsize=12)
    plt.title(r'$\mathcal{Z}_{00}\,(1;q^2)$', fontsize=13.5, y=1.035)
    if y_range != 'auto':
        plt.ylim(y_range)

    plt.subplot(122)

    plt.semilogy(q2_plot_mini, machine_limit_list,
                 label='Machine limit', linewidth=.8,
                 color='tab:gray')
    plt.semilogy(q2_plot_mini, digits_6,
                 label='6 digits', linewidth=.8,
                 linestyle='--', color='tab:brown')
    plt.semilogy(q2_plot_mini, digits_12,
                 label='12 digits', linewidth=.8,
                 linestyle='--', color='tab:purple')

    for key in ['$k$', r'$\alpha$', r'$\beta$']:
        plt.semilogy(q2_plot_mini, zeta_eps_partial(key), label=key)

    plt.xlabel('$q^2$', fontsize=12)
    plt.title(r"Various contributions of $\tilde{\epsilon}$\\[.5ex] "
              fr"\small for $m = {n_max_alpha},\, "
              fr"m' = {n_max_beta},\, k = {k_max}$, "
              fr"integral order {int_order}",
              fontsize=13.5, y=1.035)
    plt.ylim(*error_range)
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.035))

    plt.tight_layout()
    plt.subplots_adjust(wspace=.25)
    plt.show()
    return None


q2_plot_slim = q2_plot[round(size * .12):round(size * .415)]
pre_found_root = find_root_secant(
    lambda x: zeta(x) - phase_shift(x), 0.001, 0.999,
    accuracy=10**(-16))


def zeta_root_plot(n_max_alpha, n_max_beta,
                   k_max, int_order,
                   y_range='auto'):

    params = [n_max_alpha, n_max_beta, k_max, int_order]
    zeta_values = [zeta(q2, *params) for q2 in q2_plot]

    phase_shift_list = [ phase_shift(q2) for q2 in q2_plot_slim]

    # Truncate value list
    # AFTER data preparation
    upper = 1000.
    lower = -1000.
    for i in range(size):
        if zeta_values[i] > upper:
            zeta_values[i] = num.inf
        if zeta_values[i] < lower:
            zeta_values[i] = -num.inf

    plt.subplots(dpi=80)

    plt.plot(q2_plot, zeta_values)
    phase_shift_plot = plt.plot(q2_plot_slim, phase_shift_list,
                                label='Phase shift')

    plt.setp(phase_shift_plot, linewidth=1.2, linestyle='--',
             color='tab:gray')
    plt.scatter([pre_found_root], [phase_shift(pre_found_root)],
                s=24, color='tab:orange', zorder=10)

    plt.grid(linestyle='dotted')
    plt.xlabel('$q^2$', fontsize=12)
    plt.title(r'$\mathcal{Z}_{00}\,(1;q^2)$, '
              'compared with phase shift', fontsize=13.5, y=1.035)
    if y_range != 'auto':
        plt.ylim(y_range)

    plt.legend(loc='upper right', bbox_to_anchor=(1.28, 1))
    plt.tight_layout()
    plt.show()
    return None
