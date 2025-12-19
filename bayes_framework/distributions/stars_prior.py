import numpy as np
from bayes_framework.distributions.standard_distributions import UniformDistributions


class PriorBinStars:
    def __init__(self, age_grid):
        self.min_age = age_grid.min()
        self.max_age = age_grid.max()

        self.imf = KroupaInitialMassFunction()
        self.age_norm = 1.0 / (self.max_age - self.min_age)

    def __call__(self, age, mass_1, mass_2):
        """
        Vectorized prior:
        age, mass_1, mass_2 can be scalars or numpy arrays
        """

        age = np.asarray(age)

        # --- uniform age prior ---
        p_age = np.zeros_like(age, dtype=float)
        mask_age = (age >= self.min_age) & (age <= self.max_age)
        p_age[mask_age] = self.age_norm

        # --- IMF ---
        p_m1 = self.imf.prob(mass_1)
        p_m2 = self.imf.prob(mass_2)

        return p_age * p_m1 * p_m2


class KroupaInitialMassFunction:
    def __init__(self):
        self.mass_c = 0.079
        self.sigma_lm = 0.69
        self.k_kroupa = 4.53
        self.a_coef = 0.2791

    def prob(self, mass):
        """
        Vectorized Kroupa IMF.
        mass : float or np.ndarray
        """

        mass = np.asarray(mass)
        p = np.zeros_like(mass, dtype=float)

        mask_low = (mass >= 0.1) & (mass <= 1.0)
        mask_high = (mass > 1.0) & (mass <= 150.0)

        # log-normal part
        p[mask_low] = (
            self.k_kroupa
            / mass[mask_low]
            * np.exp(
                - (np.log10(mass[mask_low]) - np.log10(self.mass_c))**2
                / (2.0 * self.sigma_lm**2)
            )
        )

        # power-law part
        p[mask_high] = (
            self.k_kroupa
            * self.a_coef
            * mass[mask_high]**(-2.35)
        )

        return p


if __name__ == '__main__':
    print('Test')
