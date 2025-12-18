import numpy as np
from bayes_framework.models.isochrone_bin_stars import Isochrone_Bin_Stars_Model


class GaussianLikelihoodBinStars:
    """

    Класс создания функции правдоподобия в виде суммы гауссиан, сделано чисто для двойных звезд

    """

    def __init__(self, data, err, model):
        self.observ_data = data
        self.observ_err = err
        self.predict_model = model

    def __call__(self, param_grid):
        pred = self.predict_model(param_grid).forward_model()

        age_grid = pred[0].keys()

        model_data = pred[2]

        likelihood_grid = {}

        for age in age_grid:
            print(round(min(pred[1][age]), 2), round(max(pred[1][age]), 2))
            likelihood_grid[age] = 1.0 / 2.0 / np.pi * 1.0 / np.sqrt(self.observ_err[0] * self.observ_err[1]) * np.exp(
                -np.power(self.observ_data[0] - model_data[age][0], 2.0) / 2.0 / np.power(self.observ_err[0], 2.0)) * np.exp(
                -np.power(self.observ_data[1] - model_data[age][1], 2.0) / 2.0 / np.power(self.observ_err[1], 2.0))

        return likelihood_grid


if __name__ == '__main__':

    age_grid = np.arange(7.5, 8.7, 0.1)

    model = Isochrone_Bin_Stars_Model(np.arange(7.5, 8.7, 0.1))

    likehood = GaussianLikelihoodBinStars([0.1373, 1.32], [0.0002, 0.05], Isochrone_Bin_Stars_Model)(age_grid)

