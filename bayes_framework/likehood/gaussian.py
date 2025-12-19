import numpy as np
from bayes_framework.models.isochrone_bin_stars import Isochrone_Bin_Stars_Model
from scipy.interpolate import RegularGridInterpolator

class GaussianLikelihoodBinStars:
    """

    Класс создания функции правдоподобия в виде суммы гауссиан, сделано чисто для двойных звезд.
    На вход воспринимается массив данных и соответствующих ошибок. Пока, что это только функция масс и отношение потоков.
    Структура Likehood

    init: данные, ошибки, модель, которая дает наблюдения по параметрам (в данном случае возраст). Но среди неизвестных
    параметров есть массы.

    Наблюдательные данные: Функция масс, отношениие светимостей

    Функция вызываема: зависит сетки по логарифму возраста

    На выход dict[age]

    В каждой ячейке словаря массив квадратный по массам. То есть итого это массив 3-х мерный.

    """

    def __init__(self, data, err, model):
        self.observ_data = data
        self.observ_err = err
        self.predict_model = model

    def __call__(self, param_grid):
        age_grid = param_grid[0]
        mass_grid = param_grid[1]

        pred = self.predict_model(age_grid).forward_model()
        model_data = pred[2]

        num_of_age = len(age_grid)
        num_of_mass = len(mass_grid)

        likelihood_grid = np.zeros((num_of_age, num_of_mass, num_of_mass))

        for k, age in enumerate(age_grid):
            mass_grid_local = pred[0][age]
            mf_grid_local = model_data[age][0]
            flux_grid_local = model_data[age][1]

            interp_mf = RegularGridInterpolator(
                (mass_grid_local, mass_grid_local),
                mf_grid_local,
                bounds_error=False,
                fill_value=np.nan
            )

            interp_flux = RegularGridInterpolator(
                (mass_grid_local, mass_grid_local),
                flux_grid_local,
                bounds_error=False,
                fill_value=np.nan
            )

            m1, m2 = np.meshgrid(mass_grid, mass_grid, indexing="ij")
            pts = np.column_stack([m1.ravel(), m2.ravel()])

            mf_pred = interp_mf(pts).reshape(num_of_mass, num_of_mass)
            f_pred = interp_flux(pts).reshape(num_of_mass, num_of_mass)

            valid = np.isfinite(mf_pred) & np.isfinite(f_pred)

            likelihood_grid_k = np.zeros((num_of_mass, num_of_mass))

            likelihood_grid_k[valid] = (
                    1.0 / (2.0 * np.pi * self.observ_err[0] * self.observ_err[1])
                    * np.exp(
                -0.5 * ((self.observ_data[0] - mf_pred[valid]) / self.observ_err[0]) ** 2
                - 0.5 * ((self.observ_data[1] - f_pred[valid]) / self.observ_err[1]) ** 2
            )
            )

            likelihood_grid[k] = likelihood_grid_k

        return likelihood_grid


if __name__ == '__main__':
    age_grid_input = np.arange(7.5, 8.7, 0.1)
    mass_grid_input = np.arange(1.0, 7.0, 0.05)

    model = Isochrone_Bin_Stars_Model(age_grid_input)

    likehood = GaussianLikelihoodBinStars([0.1373, 1.32], [0.0002, 0.05], Isochrone_Bin_Stars_Model)(
        [age_grid_input, mass_grid_input])

