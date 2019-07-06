import argparse

import matplotlib.pyplot as plt
import numpy as np
import tables


def plot_pose(path):
    h5file = tables.open_file(path, mode='r')
    rows = h5file.root['train'].where('yaw < 180')
    angles = [row['yaw'] for row in rows]
    angles = np.array(angles)
    plt.hist(angles, bins=180)
    plt.xlabel('Yaw')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    plot_pose(args.data)
