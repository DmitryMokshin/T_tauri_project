import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from bayes_framework.models.isochrone_bin_stars import Isochrone_Bin_Stars_Model
from bayes_framework.likehood.gaussian import GaussianLikelihoodBinStars
from bayes_framework.distributions.stars_prior import PriorBinStars


class InferenceBaseClassicApproach:
    """

    Класс для классического байес подхода по работе с апостериорным распределением, через прямое построение карт.
    А затем определение максимума апостериорного распределения.


    """

    def __init__(self, model, prior, observed_parameters, err_observed_parameters):

        self.model = model
        self.prior = prior

        self.observed_parameters = observed_parameters
        self.err_observed_parameters = err_observed_parameters

        self.age_grid = None
        self.mass_grid = None

        self.posterior_grid = None

        self.p_age_m1_2d = None
        self.p_age_m2_2d = None
        self.p_m1_m2_2d = None

        self.P_age = None
        self.P_m1 = None
        self.P_m2 = None

    def compute_posterior(self, input_parameters_grid):
        self.age_grid, self.mass_grid = input_parameters_grid

        age = self.age_grid
        mass = self.mass_grid

        dt = age[1] - age[0]
        dm = mass[1] - mass[0]

        print('Begin computing likelihood')

        likelihood = GaussianLikelihoodBinStars(
            self.observed_parameters,
            self.err_observed_parameters,
            self.model
        )([age, mass])

        print('End computing likelihood')

        print('Begin computing prior')

        A, M1, M2 = np.meshgrid(age, mass, mass, indexing='ij')

        prior_grid = self.prior(A, M1, M2)

        print('End computing prior')

        # --- posterior ---
        self.posterior_grid = normalize_posterior_grid(
            likelihood * prior_grid,
            age,
            mass
        )

        print('End computing posterior')

        # --- максимум ---
        res = find_max_3d(self.posterior_grid, age, mass)
        print(res)

        # --- маргинализация ---
        (
            self.p_age_m1_2d,
            self.p_age_m2_2d,
            self.p_m1_m2_2d,
            self.P_age,
            self.P_m1,
            self.P_m2
        ) = marginalize(self.posterior_grid, dt, dm)

        # --- графики ---
        self._plot_1d_distributions()

        self.plot_maps()

    def _plot_1d_distributions(self):

        plots = [
            (self.age_grid, self.P_age, 'Age', 'Marginal_Age_Distribution.png'),
            (self.mass_grid, self.P_m1, 'Primary Mass', 'Primary_Mass_Distribution.png'),
            (self.mass_grid, self.P_m2, 'Secondary Mass', 'Secondary_Mass_Distribution.png'),
        ]

        for x, y, label, fname in plots:
            plt.figure()
            plt.plot(x, y)
            plt.xlabel(label, fontsize=16)
            plt.ylabel('Probability', fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()

    def plot_maps(self):

        # ---- единая шкала цветов ----
        vmin = min(
            self.p_age_m2_2d.min(),
            self.p_m1_m2_2d.min(),
            self.p_age_m1_2d.min()
        )
        vmax = max(
            self.p_age_m2_2d.max(),
            self.p_m1_m2_2d.max(),
            self.p_age_m1_2d.max()
        )

        # ---- figure + gridspec ----
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=fig, hspace=0.0, wspace=0.0)

        ax_age_m2 = fig.add_subplot(gs[0, 0])
        ax_m1_m2 = fig.add_subplot(gs[1, 0], sharex=ax_age_m2)
        ax_age_m1 = fig.add_subplot(gs[1, 1], sharey=ax_m1_m2)

        # ---- карты ----
        im1 = ax_age_m2.imshow(
            self.p_age_m2_2d,
            origin='lower',
            aspect='auto',
            cmap='gray_r',
            vmin=vmin,
            vmax=vmax,
            extent=[
                self.mass_grid.min(), self.mass_grid.max(),
                self.age_grid.min(), self.age_grid.max()
            ]
        )

        im2 = ax_m1_m2.imshow(
            self.p_m1_m2_2d,
            origin='lower',
            aspect='auto',
            cmap='gray_r',
            vmin=vmin,
            vmax=vmax,
            extent=[
                self.mass_grid.min(), self.mass_grid.max(),
                self.mass_grid.min(), self.mass_grid.max()
            ]
        )

        im3 = ax_age_m1.imshow(
            self.p_age_m1_2d,
            origin='lower',
            aspect='auto',
            cmap='gray_r',
            vmin=vmin,
            vmax=vmax,
            extent=[
                self.age_grid.min(), self.age_grid.max(),
                self.mass_grid.min(), self.mass_grid.max()
            ]
        )

        # ---- сетка ----
        for ax in (ax_age_m2, ax_m1_m2, ax_age_m1):
            ax.grid(True, linestyle=':', alpha=0.4)

        # ---- подписи ----
        ax_m1_m2.set_xlabel(r'$M_2\,[M_\odot]$')
        ax_m1_m2.set_ylabel(r'$M_1\,[M_\odot]$')

        ax_age_m2.set_ylabel(r'Log Age', fontsize=20)
        ax_age_m1.set_xlabel(r'Log Age', fontsize=20)

        # ---- скрываем лишние тики ----
        ax_age_m2.tick_params(labelbottom=False)
        ax_age_m1.tick_params(labelleft=False)

        # ---- общий colorbar ----
        cbar = fig.colorbar(
            im1,
            ax=(ax_age_m2, ax_m1_m2, ax_age_m1),
            shrink=0.8,
            pad=0.02
        )
        cbar.set_label('Posterior probability', fontsize=20)

        plt.savefig('Maps_distributions.png', dpi=150, format='png')
        plt.close()


def find_max_3d(P, age_grid, mass_grid):
    """
    Поиск максимума в 3D сетке постериора.

    Parameters
    ----------
    P : ndarray, shape (N_age, N_mass, N_mass)
        Апостериор
    age_grid : ndarray
        Сетка возрастов
    mass_grid : ndarray
        Сетка масс (общая для M1 и M2)

    Returns
    -------
    result : dict
        Индексы, аргументы и значение максимума
    """

    # индекс максимума
    flat_index = np.argmax(P)
    i_age, i_m1, i_m2 = np.unravel_index(flat_index, P.shape)

    result = {
        "indices": {
            "age": i_age,
            "M1": i_m1,
            "M2": i_m2
        },
        "values": {
            "age": age_grid[i_age],
            "M1": mass_grid[i_m1],
            "M2": mass_grid[i_m2]
        },
        "posterior_max": P[i_age, i_m1, i_m2]
    }

    return result


def normalize_posterior_grid(posterior, age_grid, mass_grid):
    dt = age_grid[1] - age_grid[0]
    dM = mass_grid[1] - mass_grid[0]

    Z = np.sum(posterior)
    return posterior / Z


def marginalize(post, dt, dm):
    P_age_m1 = post.sum(axis=2)
    P_m1_m2 = post.sum(axis=0)
    P_age_m2 = post.sum(axis=1)

    P_age = post.sum(axis=(1, 2))
    P_m1 = post.sum(axis=(0, 2))
    P_m2 = post.sum(axis=(0, 1))

    return P_age_m1, P_age_m2, P_m1_m2, P_age, P_m1, P_m2


if __name__ == "__main__":
    age_grid_input = np.arange(7.0, 9.0, 0.01)
    mass_grid_input = np.arange(1.0, 7.0, 0.01)

    observ_mf = 0.1373
    observ_f = 1.91

    err_mf = 0.0002
    err_f = 0.05

    core_method = InferenceBaseClassicApproach(Isochrone_Bin_Stars_Model, PriorBinStars(age_grid_input),
                                               [observ_mf, observ_f],
                                               [err_mf, err_f])

    core_method.compute_posterior([age_grid_input, mass_grid_input])
