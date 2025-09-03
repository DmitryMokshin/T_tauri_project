import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import Akima1DInterpolator, RegularGridInterpolator

from MESA_product_module import read_mesa_track


def get_one_isochrone(target_age, input_tracks):
    age_target_linear = 10.0 ** target_age

    isochrone = {
        'star_mass': [],
        'log_age': [],
        'mass': [],
        'log_g': [],
        'log_R': [],
        'log_Teff': [],
        'log_L': []
    }

    for m, track in input_tracks.items():
        ages = track['star_age'].values

        # Проверка диапазона возраста
        if not (ages[0] <= age_target_linear <= ages[-1]):
            continue

        # Быстрый поиск индексов (предыдущий и следующий)
        idx = np.searchsorted(ages, age_target_linear)
        if idx == 0 or idx == len(ages):
            continue

        # Две ближайшие точки
        row1, row2 = track.iloc[idx - 1], track.iloc[idx]

        log_age_array = np.log10([row1['star_age'], row2['star_age']])
        values = {
            'mass': [row1['star_mass'], row2['star_mass']],
            'log_g': [row1['log_g'], row2['log_g']],
            'log_R': [row1['log_R'], row2['log_R']],
            'log_Teff': [row1['log_Teff'], row2['log_Teff']],
            'log_L': [row1['log_L'], row2['log_L']]
        }

        # Интерполяция
        isochrone['star_mass'].append(m)
        isochrone['log_age'].append(target_age)

        for key, arr in values.items():
            interpolator = Akima1DInterpolator(log_age_array, arr)
            isochrone[key].append(interpolator(target_age))

    result_iso = pd.DataFrame(isochrone)

    plt.plot(result_iso['log_Teff'], result_iso['log_L'], label=f'Age: {target_age}')
    return result_iso

def read_one_tess_flux(file_path):
    """
    Читает файл формата tess_flux_xxxx.dat и возвращает pandas DataFrame.
    """
    df = pd.read_csv(
        file_path,
        sep='\s+',
        comment="#",
        header=None,
        names=["logg", "flux", "flux_with_extinction"]
    )
    return df

def read_all_tess_flux(path_dir_tess_flux):
    str_list_tess_temperature = sorted(os.listdir(path_dir_tess_flux))

    temp_list = [
        float(f.replace("tess_flux_", "").replace(".dat", ""))
        for f in str_list_tess_temperature
    ]

    dfs = [read_one_tess_flux(os.path.join(path_dir_tess_flux, f))
           for f in str_list_tess_temperature]

    logg_list = dfs[0]["logg"].values

    grid_flux_without_ext = np.vstack([df["flux"].values for df in dfs])
    grid_flux_with_ext = np.vstack([df["flux_with_extinction"].values for df in dfs])

    flux_interpolator = RegularGridInterpolator(
        (temp_list, logg_list), grid_flux_without_ext
    )
    flux_with_ext_interpolator = RegularGridInterpolator(
        (temp_list, logg_list), grid_flux_with_ext
    )

    return flux_interpolator, flux_with_ext_interpolator

if __name__ == '__main__':
    file_path_tess_flux = './tess-tables/'

    t_eff = 3900

    file_name = f'tess_flux_{t_eff:.0f}.dat'

    read_all_tess_flux(path_dir_tess_flux=file_path_tess_flux)

    # print(read_one_tess_flux(file_path_tess_flux + file_name))
    # dir_with_track = '/home/dmitrii/Science/T_tauri_project/MESA_TRACK/mesa_res/'
    # masses = np.arange(0.75, 7.1, 0.25)
    #
    # tracks = {
    #     mass: read_mesa_track(f'{dir_with_track}ROT{0.00:.2f}Z{0.0147}M{mass:.2f}.dat')
    #     for mass in masses
    # }
    #
    # for age in np.arange(5.5, 9.1, 0.5):
    #     get_one_isochrone(age, tracks)
    #
    # plt.grid()
    # plt.gca().invert_xaxis()
    # plt.legend()
    # plt.show()
