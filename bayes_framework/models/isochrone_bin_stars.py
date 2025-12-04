import matplotlib.pyplot as plt
import numpy as np
from bayes_framework.utils.mesa_utils_module import load_tracks, extract_model_grid, get_one_isochrone


class Isochrone_Bin_Stars_Model:
    def __init__(self, input_grid_age, dir_path_name='/home/dmitrii/PycharmProjects/T_tau_project/Mesa_tracks_res/'):
        self.tracks = None
        self.isos = {}
        self.grid_age = input_grid_age

        self.read_tracks_grid(dir_path_name)
        self.compute_isochrone()

    def read_tracks_grid(self, dir_path_name):
        mesa_tracks = load_tracks(dir_path_name)

        self.tracks = mesa_tracks

    def compute_isochrone(self):
        for age in self.grid_age:
            self.isos[age] = get_one_isochrone(age, self.tracks[0.00][0.0147]).sort_values('mass', ignore_index=True)

    def forward_model(self):

        observe_data_grid = {}

        data_mass_grid = {}
        data_flux_grid = {}

        for age in self.grid_age:
            mass_grid = self.isos[age]['mass']
            flux_grid = self.isos[age]['log_TESS']

            num_of_points = len(mass_grid)

            mass_function_grid = np.zeros((num_of_points, num_of_points))
            fraction_flux_grid = np.zeros((num_of_points, num_of_points))

            for i in range(num_of_points):
                for j in range(num_of_points):
                    mass_function_grid[i, j] = mass_function(mass_grid[i], mass_grid[j], 90.0 * np.pi / 180.0)

                    fraction_flux_grid[i, j] = flux_grid[i] - flux_grid[j]

            observe_data_grid[age] = [mass_function_grid, fraction_flux_grid]

            data_flux_grid[age] = flux_grid
            data_mass_grid[age] = mass_grid

        return data_mass_grid, data_flux_grid, observe_data_grid


def mass_function(mass_1, mass_2, incline):
    return np.power(mass_2 * np.sin(incline), 3.0) / np.power(mass_1 + mass_2, 2.0)


if __name__ == '__main__':
    tracks_folder = '../../Mesa_tracks_res/'

    Isochrone_Bin_Stars_Model(np.arange(7.0, 8.7, 0.1), tracks_folder).forward_model()
