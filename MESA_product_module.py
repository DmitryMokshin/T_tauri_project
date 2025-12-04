from itertools import islice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


if __name__ == '__main__':

    dir_mesa_track_pre = '/home/dmitrii/Science/T_tauri_project/MESA_TRACK/mesa_tracks/'

    log_beg = 'LOGS_start/'
    log_int = 'LOGS_ms/'
    log_end = 'LOGS_end/'

    file_name = 'history.data'

    pattern_dir_name = lambda m, z, w: f'track_M{m:.2f}_Z{z}_ROT{w:.2f}/'

    rot = 0.00
    metal = 0.0147

    masses = np.arange(3.5, 4.66, 0.05)

    for mass in masses:

        name_dir = dir_mesa_track_pre + pattern_dir_name(mass, metal, rot)

        name_track_1 = name_dir + log_beg + file_name
        name_track_2 = name_dir + log_int + file_name
        name_track_3 = name_dir + log_end + file_name

        track_1 = read_mesa_track(name_track_1)
        track_2 = read_mesa_track(name_track_2)
        track_3 = read_mesa_track(name_track_3)

        track = pd.concat([track_1, track_2, track_3], ignore_index=True)

        track.to_csv('./Mesa_tracks_res/' + f'ROT{rot:.2f}M{mass:.2f}Z{metal}.csv', index=False, sep=',')
