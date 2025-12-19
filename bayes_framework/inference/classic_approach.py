import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

from bayes_framework.models.isochrone_bin_stars import Isochrone_Bin_Stars_Model
from bayes_framework.likehood.gaussian import GaussianLikelihoodBinStars
from bayes_framework.distributions.stars_prior import PriorBinStars


class InferenceBaseClassicApproach:
    """

    Класс для классического байес подхода по работе с апостериорным распределением, через прямое построение карт.
    А затем определение максимума апостериорного распределения.


    """

    def __init__(self, model, prior, observed_parameters, err_observed_parameters):
        self.P_m2 = None
        self.P_m1 = None
        self.P_age = None
        self.p_m1_m2_2d = None
        self.p_age_m2_2d = None
        self.p_age_m1_2d = None
        self.posterior_grid = None
        self.model = model
        self.prior = prior

        self.observed_parameters = observed_parameters
        self.err_observed_parameters = err_observed_parameters

    def compute_posterior(self, input_parameters_grid):
        age_grid = input_parameters_grid[0]
        mass_grid = input_parameters_grid[1]

        dt = age_grid[1] - age_grid[0]
        dm = mass_grid[1] - mass_grid[0]

        num_mass = len(mass_grid)
        num_age = len(age_grid)

        likelihood_grid = GaussianLikelihoodBinStars(self.observed_parameters, self.err_observed_parameters,
                                                     self.model)(
            [age_grid, mass_grid])

        posterior_grid = np.zeros([num_age, num_mass, num_mass])

        for k in range(num_age):
            for l in range(num_mass):
                for m in range(num_mass):
                    posterior_grid[k, l, m] = likelihood_grid[k, l, m] * self.prior(age_grid[k], mass_grid[l],
                                                                                    mass_grid[m])

        self.posterior_grid = normalize_posterior_grid(posterior_grid, age_grid, mass_grid)

        res = find_max_3d(self.posterior_grid, age_grid, mass_grid)

        print(res)

        maps = marginalize(self.posterior_grid, dt, dm)

        self.p_age_m1_2d = maps[0]
        self.p_age_m2_2d = maps[1]
        self.p_m1_m2_2d = maps[2]

        self.P_age = maps[3]
        self.P_m1 = maps[4]
        self.P_m2 = maps[5]

        plt.plot(age_grid, self.P_age)
        plt.show()

        plt.plot(mass_grid, self.P_m1)
        plt.show()

        plt.plot(mass_grid, self.P_m2)
        plt.show()

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
    age_grid_input = np.arange(5.0, 9.0, 0.01)
    mass_grid_input = np.arange(1.0, 7.0, 0.05)

    observ_mf = 0.1373
    observ_f = 1.21

    err_mf = 0.0002
    err_f = 0.05

    core_method = InferenceBaseClassicApproach(Isochrone_Bin_Stars_Model, PriorBinStars(age_grid_input),
                                               [observ_mf, observ_f],
                                               [err_mf, err_f])

    core_method.compute_posterior([age_grid_input, mass_grid_input])


