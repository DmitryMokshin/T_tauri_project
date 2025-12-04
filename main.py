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
        'log_L': [],
        'TESS_flux': [],
        'TESS_flux_w_ex': []
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
            'log_L': [row1['log_L'], row2['log_L']],
            'TESS_flux': [row1['TESS_flux'], row2['TESS_flux']],
            'TESS_flux_w_ex': [row1['TESS_flux_w_ex'], row2['TESS_flux_w_ex']]
        }

        # Интерполяция
        isochrone['star_mass'].append(m)
        isochrone['log_age'].append(target_age)

        for key, arr in values.items():
            interpolator = Akima1DInterpolator(log_age_array, arr)
            isochrone[key].append(float(interpolator(target_age)))

    result_iso = pd.DataFrame(isochrone)

    return result_iso

def read_one_tess_flux(file_path):
    """
    Читает файл формата tess_flux_xxxx.dat и возвращает pandas DataFrame.
    """
    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        comment="#",
        header=None,
        names=["logg", "flux", "flux_with_extinction"]
    )
    return df

def read_all_tess_flux(path_dir_tess_flux='./tess-tables/'):
    str_list_tess_temperature = os.listdir(path_dir_tess_flux)

    temp_file_pairs = sorted(
        [(float(f.replace("tess_flux_", "").replace(".dat", "")), f)
         for f in str_list_tess_temperature],
        key=lambda x: x[0]
    )

    temp_list = [t for t, _ in temp_file_pairs]
    dfs = [read_one_tess_flux(os.path.join(path_dir_tess_flux, f))
           for _, f in temp_file_pairs]

    logg_list = dfs[0]["logg"].values

    grid_flux_without_ext = np.vstack([df["flux"].values for df in dfs])
    grid_flux_with_ext = np.vstack([df["flux_with_extinction"].values for df in dfs])

    flux_interpolator = RegularGridInterpolator(
        (temp_list, logg_list), grid_flux_without_ext, bounds_error=False, fill_value=None
    )
    flux_with_ext_interpolator = RegularGridInterpolator(
        (temp_list, logg_list), grid_flux_with_ext, bounds_error=False, fill_value=None
    )

    return flux_interpolator, flux_with_ext_interpolator

if __name__ == '__main__':
    file_path_tess_flux = './tess-tables/'

    fl_interpolator, fl_with_ext_interpolator = read_all_tess_flux(path_dir_tess_flux=file_path_tess_flux)

    dir_with_track = '/home/dmitrii/Science/T_tauri_project/MESA_TRACK/mesa_tracks_2/'
    masses = np.arange(2.25, 7.1, 0.25)

    tracks = {
        mass: read_mesa_track(f'{dir_with_track}ROT{0.00:.2f}Z{0.0147}M{mass:.2f}.dat')
        for mass in masses
    }

    for m, track in tracks.items():
        # print(m, np.log10(track['star_age'].iloc[-1]), track['model_number'].iloc[-1])
        points = np.column_stack((track["Teff"].values, track["log_g"].values))
        track['TESS_flux'] = fl_interpolator(points) * np.power(np.power(10.0, track['log_R']), 2.0).to_numpy().astype(float)
        track['TESS_flux_w_ex'] = fl_with_ext_interpolator(points) * np.power(np.power(10.0, track['log_R']), 2.0).to_numpy().astype(float)
        # plt.plot(track['log_Teff'], track['log_g'])


    # for age in np.arange(0.5, 9.0, 0.1):
    #     iso = get_one_isochrone(age, tracks)
    #     plt.plot(iso['log_Teff'], iso['log_g'], label=f'Age: {age: 0.2f}')
    # for mass in masses:
    #     if 4 <= mass <= 7:
    #         plt.plot(tracks[mass]['log_Teff'], tracks[mass]['log_L'], label=f'M: {mass: 2.2f}', linestyle='--', alpha=0.9)
    track_4 = read_mesa_track('ROT0.00Z0.0147M4.00.dat')

    plt.plot(np.log10(track_4['star_age']), track_4['log_g'])
    plt.errorbar(np.log10(track_4['star_age']), 3.3 * np.ones(len(track_4)), color='red')
    plt.errorbar(np.log10(track_4['star_age']), 2.1 * np.ones(len(track_4)), color='red')
    # plt.errorbar(np.log10(4800), 2.5, 0.2, np.log(10) / 4800.0 * 100.0, color='red')

        # iso_filter = iso[
        #     (iso['log_g'].between(2.3, 2.7)) &
        #     (iso['log_Teff'].between(np.log10(4700), np.log10(4900)))
        #     ]
        # if len(iso_filter) > 0.0:
        #     print(iso_filter)


        # plt.plot(iso['log_Teff'], np.log10(iso['TESS_flux_w_ex']), label=f'With ex Age: {age}', linestyle='--')
    # plt.xlim(np.log10(4700.0), np.log10(4900))
    # plt.ylim(2.3, 2.7)
    # plt.gca().invert_xaxis()
    plt.grid()
    plt.legend()
    plt.show()
