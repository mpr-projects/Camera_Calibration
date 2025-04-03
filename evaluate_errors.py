import os
import sys
import yaml
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

from target_functions import get_state, extract_points_options, plot_options
from validation_functions import find_pose_fn


# Todo: it would be more efficient to apply ignore_source_indices before computing errors, I've only
#       added that as an afterthought when making the video, still have to update the code ...


def project_and_compute_errors(camera_matrix, dist, src_path, settings, state, extract_points_fns, plot_fns):
    fnames = [f for f in os.listdir(src_path) if f.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']]
    corners_, errors = [], []

    for fid, fname in enumerate(fnames):
        print(f' File {fid+1}/{len(fnames)}\033[F')
        fname = os.path.join(src_path, fname)
        img = cv.imread(fname, flags=cv.IMREAD_COLOR + cv.IMREAD_IGNORE_ORIENTATION)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, rvec, tvec, corners, objpoints = find_pose_fn(
            img, gray, camera_matrix, dist, settings, state, extract_points_fns, plot_fns)
        
        corners_.append(corners)

        assert ret, 'This error can\'t really happen ...'

        proj_corners, _ = cv.projectPoints(objpoints, rvec, tvec, camera_matrix, dist)

        errs = ((corners - proj_corners)**2).sum(axis=-1)**0.5
        errors.append(errs.flatten())

    corners_ = np.concatenate(corners_).reshape(-1, 2)
    return corners_, np.concatenate(errors)


def plot_individual(labels, corners, errors):
    maxv = max([np.amax(e) for e in errors])
    cmap = plt.get_cmap('YlOrRd')

    for l, c, e in zip(labels, corners, errors):
        p = plt.scatter(c[:, 0], c[:, 1], vmin=0, vmax=maxv, c=e, cmap=cmap)
        plt.colorbar(p, orientation='horizontal')
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.axis('off')
        plt.title(l)
        plt.show()


def plot_combined(labels, corners, errors):
    # return plot_individual(corners, errors)
    maxv = max([np.amax(e) for e in errors])
    cmap = plt.get_cmap('YlOrRd')

    fig, axs = plt.subplot_mosaic([
        ['upper left', 'upper right'],
        ['lower', 'lower']], layout="constrained", height_ratios=[15, 1])
    
    for l, c, e, loc in zip(labels, corners, errors, ['upper left', 'upper right']):
        ax = axs[loc]
        p = ax.scatter(c[:, 0], c[:, 1], vmin=0, vmax=maxv, c=e, cmap=cmap)
        ax.invert_yaxis()
        ax.axis('equal')
        ax.axis('off')
        ax.set_title(l)

    plt.colorbar(p, cax=axs['lower'], orientation='horizontal')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate reprojection errors.')
    parser.add_argument('calibration_file', type=str, help='Path to the calibration results file (pickle).')
    parser.add_argument('--labels', nargs='+', help='Labels to use in plot. If not given then the source folders are used.')
    parser.add_argument('--ignore_source_indices', nargs='+', type=int,
                        help='Indices of sources to ignore in plotting. This is applied at the end so if you'
                             ' provide labels then you have to provide them for all sources.')
    args = parser.parse_args()

    # Load calibration data
    with open(args.calibration_file, 'rb') as f:
        settings, state = pickle.load(f)

    # need a new state to get the detectors
    state_new = get_state(settings)

    for k, v in state_new.items():
        if k.startswith('detector'):
            state[k] = state_new[k]
            print('copied', k)

    # set up global functions
    pattern_types = settings['pattern_type']

    extract_points_fns = [extract_points_options[pattern_type] for pattern_type in pattern_types]
    plot_fns = [plot_options[pattern_type] for pattern_type in pattern_types]

    # compute errors
    val_paths = [p[1] for p in settings['src_folders'].values()]
    results = state['results']

    if args.labels is None:
        labels = [p[0] for p in settings['src_folders'].values()]
    else:
        assert len(args.labels) == len(val_paths), f'Invalid number of labels, expected {len(val_paths)}.'
        labels = args.labels

    c, e = list(), list()

    for label, val_path, result in zip(labels, val_paths, results):
        print(f'\n\n{label}: {val_path}')

        # using the result from the most number of images, could add a command-line
        # argument to use a different result
        result = result[-1]

        camera_matrix = result[1]
        dist = result[2]

        print('camera_matrix:', camera_matrix)
        print('dist:', dist)

        corners, errors = project_and_compute_errors(
            camera_matrix, dist, val_path, settings, state,
            extract_points_fns, plot_fns
        )

        c.append(corners)
        e.append(errors)

    # remove unwanted sources
    if args.ignore_source_indices is not None:
        for idx in args.ignore_source_indices:
            del labels[idx]
            del c[idx]
            del e[idx]

    # plot results
    if len(e) == 2:
        plot_combined(labels, c, e)

    else:
        plot_individual(labels, c, e)