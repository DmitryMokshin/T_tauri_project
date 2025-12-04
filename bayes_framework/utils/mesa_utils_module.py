from itertools import islice

import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator

def read_mesa_track(filename, num_for_skip=5):
    with open(filename, 'r') as file:
        i = 1
        result_table = {}
        for line in islice(file, num_for_skip, None):
            new_line = line.split()
            if i == 1:
                for j in range(len(new_line)):
                    if new_line[j].find('\n') > 0:
                        result_table[new_line[j].replace('\n', '')] = []
                    else:
                        result_table[new_line[j]] = []
                i += 1
            else:
                column = list(result_table.keys())
                for j in range(len(new_line)):
                    if new_line[j].find('\n') > 0:
                        result_table[column[j]].append(float(new_line[j].replace('\n', '')))
                    else:
                        result_table[column[j]].append(float(new_line[j]))

        return pd.DataFrame(result_table)

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
        'log_TESS': [],
        'log_TESS_noext': []
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
            'log_TESS': [row1['log_TESS'], row2['log_TESS']],
            'log_TESS_noext': [row1['log_TESS_noext'], row2['log_TESS_noext']]
        }

        # Интерполяция
        isochrone['star_mass'].append(m)
        isochrone['log_age'].append(target_age)

        for key, arr in values.items():
            interpolator = Akima1DInterpolator(log_age_array, arr)
            isochrone[key].append(float(interpolator(target_age)))

    result_iso = pd.DataFrame(isochrone)

    return result_iso


def load_tracks(folder_path, recursive=False, round_decimals=6):
    NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

    pat_rot = re.compile(r'ROT[\s_\-:]*(' + NUM + r')', flags=re.IGNORECASE)
    pat_z   = re.compile(r'Z[\s_\-:]*('   + NUM + r')', flags=re.IGNORECASE)
    pat_m   = re.compile(r'M[\s_\-:]*('   + NUM + r')', flags=re.IGNORECASE)

    tracks = {}

    iterator = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
    for root, _, files in iterator:
        for fname in files:
            if not fname.lower().endswith('.csv'):
                continue

            mr = pat_rot.search(fname)
            mz = pat_z.search(fname)
            mm = pat_m.search(fname)

            if not (mr and mz and mm):
                continue

            r = round(float(mr.group(1).replace(',', '.')), round_decimals)
            z = round(float(mz.group(1).replace(',', '.')), round_decimals)
            m = round(float(mm.group(1).replace(',', '.')), round_decimals)

            path = os.path.join(root, fname)
            df = pd.read_csv(path)

            tracks.setdefault(r, {}).setdefault(z, {})[m] = df

    return tracks


def extract_model_grid(folder_path, recursive=False, round_decimals=6):
    NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'

    pat_rot = re.compile(r'ROT[\s_\-:]*(' + NUM + r')', flags=re.IGNORECASE)
    pat_z   = re.compile(r'Z[\s_\-:]*('   + NUM + r')', flags=re.IGNORECASE)
    pat_m   = re.compile(r'M[\s_\-:]*('   + NUM + r')', flags=re.IGNORECASE)

    ROTS, ZS, MASSES = set(), set(), set()

    iterator = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]
    for root, _, files in iterator:
        for fname in files:
            if not fname.lower().endswith('.csv'):
                continue

            mr = pat_rot.search(fname)
            mz = pat_z.search(fname)
            mm = pat_m.search(fname)

            if not (mr and mz and mm):
                continue

            r = round(float(mr.group(1).replace(',', '.')), round_decimals)
            z = round(float(mz.group(1).replace(',', '.')), round_decimals)
            m = round(float(mm.group(1).replace(',', '.')), round_decimals)

            ROTS.add(r)
            ZS.add(z)
            MASSES.add(m)

    return (
        np.array(sorted(ROTS)),
        np.array(sorted(ZS)),
        np.array(sorted(MASSES)),
    )