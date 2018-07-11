#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# <codecell>
from toolkit.pde import Advection1D
from backstage import PDESolve
from IPython.display import display, Markdown


# <codecell>
def square_input(center=-.15, half_width=.15):
    def distribution(x):
        return 1. if abs(x - center) <= half_width else 0.
    return distribution


# <codecell>
class UpwindTest(PDESolve):
    def __init__(self, working_dir='csv/', memory_saving=False):
        super().__init__(
            working_dir=working_dir,
            memory_saving=memory_saving
        )

        self.problem = Advection1D(
            velocity_function=lambda x=0., t=0.: -1.,
            initial_distributions=square_input,
            supp_domain=(-10., 0.)
        )
        self.params = {
            'dt': .02,
            'dx': .05,
            't_end': 10.
        }  # Default params
        self._pathname_builder()

    def _pathname_builder(self):
        self.name = (lambda arg: (
            'advection'
            + f"_dt{int(arg['dt'] * 100)}"
            + f"_dx{int(arg['dx'] * 100)}"
        ))(self.params)
        self.plot_dir = self.working_dir + 'figs/' + self.name + '/'
        self.path = self.working_dir + self.name + '.csv'

    def solve_with_settings(self, output_enabled=False, **upwind_solve_params):
        super().solve_with_settings(output_enabled, **upwind_solve_params)

        self.x_grid, self.u_values_list \
            = self.problem.pde_solve_upwind(
                **self.params
            )

        if output_enabled:
            self.csv_output()
        super().post_solving_cleanup()

    def visualize(self,
                  t_sample_size=100, pointsize=None,
                  linewidth=2, gif_gen=False):
        try:
            super().visualize(
                x_range=[-10., 0.], y_range=[0., 1.05],
                t_sample_size=t_sample_size,
                pointsize=pointsize, linewidth=linewidth,
                gif_gen=gif_gen
            )
            display(Markdown(
                '$'
                fr'\Delta t = {self.params["dt"]},\ '
                fr'\Delta x = {self.params["dx"]},\ '
                '$'
            ))
            display(Markdown('![](' + self.plot_dir.strip("/") + '.gif' + ')'))
        except AttributeError:
            raise AttributeError('PDE not solved yet, '
                                 'please invoke `solve_with_settings` '
                                 'before visualization. ')


if __name__ == '__main__':
    pass
