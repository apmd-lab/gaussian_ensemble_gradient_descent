import os
directory = os.path.dirname(os.path.realpath(__file__))

import numpy as np

def read_txt(filename):
    with open(filename + '.txt', 'r') as f:
        labels = f.readline()
        row = f.readline()
        n_col = np.size(row.split())
        line_count = 1
        for line in f:
            line_count += 1
        f.seek(0)
        f.readline()
        raw = np.zeros((line_count, n_col))
        count = 0
        for line in f:
            temp_array = np.array([line.split()])
            raw[count,:] = np.asarray(temp_array, dtype=np.float64)
            count += 1
        
        return raw
            
if __name__ == '__main__':
    raw = read_txt('\\\\f0\\smin\\Python Scripts\\Radiative Cooling\\AM1.5_SMARTS295')