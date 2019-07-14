import argparse

import matplotlib.pyplot as plt
import numpy as np
import tables


def accuracy(preds, labels, files):
    correct = 0
    for i, view in enumerate(labels):
        angle = preds[i]
        if angle - 15 <= view and angle + 15 >= view:
            correct += 1
        else:
            print("Expected: %.2f, Actual: %.2f, File: %s" % (view, angle, files[i]))

    acc = correct / len(preds)
    print("Accuracy: %f" % acc)


def plot_pose(path):
    h5file = tables.open_file(path, mode='r')
    column = 'yaw'
    rows = h5file.root['train'].where('%s < 180' % column)
    samples = [[abs(row[column]), row['view'], row['file']] for row in rows]
    angles = np.array([[sample[0], sample[1]] for sample in samples])
    files = [sample[2] for sample in samples]
    accuracy(angles[:, 0], angles[:, 1], files)

    plt.hist(angles[:, 0], bins=180)
    plt.xlabel(column)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    plot_pose(args.data)
