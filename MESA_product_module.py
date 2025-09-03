from itertools import islice
import pandas as pd

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