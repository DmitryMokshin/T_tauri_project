import numpy as np
from bayes_framework.distributions.standard_distributions import UniformDistributions


class PriorBinStars:
    def __init__(self, age_grid):
        self.min_age = min(age_grid)
        self.max_age = max(age_grid)

    def __call__(self, age, mass_1, mass_2):
        kroupa_general = KroupaInitialMassFunction()

        kroupa_1 = kroupa_general.prob(mass_1)
        kroupa_2 = kroupa_general.prob(mass_2)

        return UniformDistributions(self.min_age, self.max_age).pdf(age) * kroupa_1 * kroupa_2


class KroupaInitialMassFunction:
    def __init__(self):
        self.mass_c = 0.079
        self.sigma_lm = 0.69
        self.k_kroupa = 4.53
        self.a_coef = 0.2791

    def prob(self, mass):
        if 0.1 <= mass <= 1.0:
            return self.k_kroupa * 1.0 / mass * np.exp(
                -np.power(np.log10(mass) - np.log10(self.mass_c), 2.0) / 2.0 / np.power(self.sigma_lm, 2.0))
        elif 1.0 < mass <= 150:
            return self.k_kroupa * self.a_coef * np.power(mass, -2.35)
        else:
            return None


if __name__ == '__main__':
    print('Test')
