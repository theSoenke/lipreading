import argparse

import matplotlib.pyplot as plt
import numpy as np
import tables


def accuracy(preds, labels, files, degree=15):
    correct = 0
    total_samples = {0: 0, 30: 0, 45: 0, 60: 0, 90: 0}
    correct_samples = {0: 0, 30: 0, 45: 0, 60: 0, 90: 0}

    for i, view in enumerate(labels):
        angle = preds[i]
        total_samples[view] += 1
        if angle - degree <= view and angle + degree >= view:
            correct += 1
            correct_samples[view] += 1
        else:
            print("Expected: %.2f, Actual: %.2f, File: %s" % (view, angle, files[i]))

    acc = correct / len(preds)
    print("Accuracy: %f" % acc)
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")


def plot_pose_lrw(path):
    h5file = tables.open_file(path, mode='r')
    column = 'yaw'
    rows = h5file.root['train'].where('%s < 180' % column)
    samples = [row[column] for row in rows]
    angles = np.array(samples)
    plt.figure(figsize=(9.5, 5))
    plt.hist(angles, bins=180, ec='white')
    plt.xlabel("Yaw angle")
    plt.show()


def plot_ouluvs2(path):
    h5file = tables.open_file(path, mode='r')
    column = 'yaw'
    rows = h5file.root['train'].where('%s < 180' % column)
    samples = [[abs(row[column]), row['view'], row['file']] for row in rows]
    angles = np.array([[sample[0], sample[1]] for sample in samples])
    files = [sample[2] for sample in samples]
    accuracy(angles[:, 0], angles[:, 1], files)
    plt.figure(figsize=(9.5, 5))
    plt.hist(angles[:, 0], bins=180, ec='white')
    plt.xlabel("Yaw angle")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('set', type=str)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    if args.set == 'lrw':
        plot_pose_lrw(args.data)
    else:
        plot_ouluvs2(args.data)
