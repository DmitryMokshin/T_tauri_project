import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import Akima1DInterpolator

from MESA_product_module import read_mesa_track

def get_one_isochrone(target_age, input_tracks):

    input_masses = list(input_tracks.keys())

    isochrone = {'star_mass': [], 'log_age': [], 'log_g': [], 'log_R': [], 'log_Teff': [], 'log_L': [], 'mass': []}

    for m in input_masses:

        track = input_tracks[m]

        min_age = np.min(track['star_age'])
        max_age = np.max(track['star_age'])

        if np.log10(min_age) <= target_age <= np.log10(max_age):

            track_filter_1 = track[track['star_age'] < np.power(10.0, target_age)].iloc[-1]
            track_filter_2 = track[track['star_age'] >= np.power(10.0, target_age)].iloc[0]

            log_age_array = np.array([np.log10(track_filter_1['star_age']), np.log10(track_filter_2['star_age'])])

            mass_array = np.array([track_filter_1['star_mass'], track_filter_2['star_mass']])
            log_g_array = np.array([track_filter_1['log_g'], track_filter_2['log_g']])
            log_R_array = np.array([track_filter_1['log_R'], track_filter_2['log_R']])
            log_Teff_array = np.array([track_filter_1['log_Teff'], track_filter_2['log_Teff']])
            log_L_array = np.array([track_filter_1['log_L'], track_filter_2['log_L']])

            isochrone['star_mass'].append(m)
            isochrone['log_age'].append(target_age)

            isochrone['mass'].append(Akima1DInterpolator(log_age_array, mass_array)(target_age))
            isochrone['log_g'].append(Akima1DInterpolator(log_age_array, log_g_array)(target_age))
            isochrone['log_R'].append(Akima1DInterpolator(log_age_array, log_R_array)(target_age))
            isochrone['log_L'].append(Akima1DInterpolator(log_age_array, log_L_array)(target_age))
            isochrone['log_Teff'].append(Akima1DInterpolator(log_age_array, log_Teff_array)(target_age))
        else:
            pass

    result_iso = pd.DataFrame(isochrone)

    plt.plot(result_iso['log_Teff'], result_iso['log_L'], label=f'Age: {target_age}')

    return result_iso

if __name__ == '__main__':
    dir_with_track = '/home/dmitrii/Science/T_tauri_project/MESA_TRACK/mesa_res/'

    masses = np.arange(0.75, 7.1, 0.25)

    tracks = {}

    for mass in masses:
        file_name = f'ROT{0.00:.2f}Z{0.0147}M{mass:.2f}.dat'

        tracks[mass] = read_mesa_track(dir_with_track + file_name)

    for age in np.arange(5.5, 9.1, 0.5):
        print(get_one_isochrone(age, tracks))

    plt.grid()
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()