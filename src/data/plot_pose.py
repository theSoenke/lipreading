import argparse

import matplotlib.pyplot as plt
import numpy as np
import tables


def accuracy(pred, label):
    correct = 0
    for i, view in enumerate(label):
        angle = pred[i]
        if angle - 15 <= view and angle + 15 >= view:
            correct += 1

    acc = correct / len(pred)
    print("Accuracy: %f" % acc)


def plot_pose(path):
    h5file = tables.open_file(path, mode='r')
    column = 'yaw'
    rows = h5file.root['train'].where('%s < 180' % column)
    # angles = [[abs(row[column]), row['view']] for row in rows]
    angles = [row[column] for row in rows]
    angles = np.array(angles)
    # accuracy(angles[:, 0], angles[:, 1])

    plt.hist(angles, bins=180)
    plt.xlabel(column)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    plot_pose(args.data)
