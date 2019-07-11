import argparse

import matplotlib.pyplot as plt
import numpy as np
import tables


def plot_pose(path):
    h5file = tables.open_file(path, mode='r')
    column = 'yaw_fa'
    rows = h5file.root['train'].where('%s < 180' % column)
    angles_fa = [row[column] for row in rows]
    angles_fa = np.array(angles_fa)
    plt.hist(angles_fa, bins=180)
    plt.xlabel(column)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    plot_pose(args.data)
