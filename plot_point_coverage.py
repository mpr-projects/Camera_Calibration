import os
import sys
import yaml
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from target_functions import get_state, extract_points_options, plot_options, settings_update_options
from validation_functions import find_pose_fn


def extract_points(src_path, settings, state, extract_points_fns):
    fnames = [f for f in os.listdir(src_path) if f.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']]
    corners_ = []
    w, h = 0, 0

    for fid, fname in enumerate(fnames):
        print(f' File {fid+1}/{len(fnames)}\033[F')
        fname = os.path.join(src_path, fname)
        img = cv.imread(fname, flags=cv.IMREAD_COLOR + cv.IMREAD_IGNORE_ORIENTATION)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        for idx, extract_points_fn in enumerate(extract_points_fns):
            ret, corners = extract_points_fn(img, gray, settings, state, idx)

            if ret is False:
                print(f'Couldn\'t find image corners of image {fname}.')
                return False, None, None, None
            
            corners_.append(corners)

    h, w = img.shape[:2]
    return np.concatenate(corners_).reshape(-1, 2), w, h


def get_distortion_image(w, h, dist):
    k1, k2, k3 = dist
    cx, cy = 0, 0  # assuming no offset
    xs = np.arange(w) - w/2 - cx
    ys = np.arange(h) - h/2 - cy

    xys = np.meshgrid(xs, ys)
    xys = np.stack(xys, axis=-1)

    rs = (xys**2).sum(axis=-1)**0.5

    ds = 1 + k1 * rs**2 + k2 * rs**4 + k3 * rs**6
    return ds


def plot_individual(labels, corners, w, h, dist=None):

    for l, c in zip(labels, corners):
        if dist is not None:
            ds = get_distortion_image(w, h, dist)
            print('ds:', ds.shape)
        else:
            ds = np.zeros((h, w))
            print('fake ds:', ds.shape)

        plt.imshow(ds, cmap=plt.get_cmap('Grays'))
        plt.gca().invert_yaxis()
            
        p = plt.scatter(c[:, 0], c[:, 1])
        plt.axis('equal')
        plt.axis('off')
        plt.title(l)
        rect = patches.Rectangle((0, 0), w, h, linewidth=1, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)
        plt.show()


def plot_combined(labels, corners, w, h, dist):
    _, axs = plt.subplots(ncols=2, sharex=True)

    for ax, l, c in zip(axs, labels, corners):
        if dist is not None:
            ds = get_distortion_image(w, h, dist)
            ax.imshow(ds, cmap=plt.get_cmap('Grays'))
        else:
            ax.invert_yaxis()

        p = ax.scatter(c[:, 0], c[:, 1])
        ax.axis('equal')
        ax.axis('off')
        ax.set_title(l)
        rect = patches.Rectangle((0, 0), w, h, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Point Coverage.')
    parser.add_argument('settings_file', type=str)
    parser.add_argument('--labels', nargs='+', help='Labels to use in plot. If not given then the source folders are used.')
    parser.add_argument('--plot_distortion_intensity', type=float, nargs=3,
                        help='If provided then the intensity of radial distortion is plotted underneath the points.')
    args = parser.parse_args()

    # Load settings
    with open(args.settings_file) as f:
        settings = yaml.safe_load(f)

    # set up global functions
    src_paths = settings['src_folders']
    pattern_types = settings['pattern_type']

    # we may want to use multiple calibration patterns, so we create a list of
    # calibration patterns even if we just use one
    if isinstance(pattern_types, str):
        settings_update_fn = settings_update_options[pattern_types]
        settings_update_fn(settings)
        pattern_types = settings['pattern_type'] = [pattern_types,]
        settings['pattern_size'] = [settings['pattern_size'],]
        settings['pattern_pose'] = [[0, 0, 0],]

    extract_points_fns = [extract_points_options[pattern_type] for pattern_type in pattern_types]
    state = get_state(settings)

    # visualize points
    src_paths = [p[0] for p in settings['src_folders'].values()]
    results = state['results']

    if args.labels is None:
        labels = src_paths
    else:
        assert len(args.labels) == len(src_paths), f'Invalid number of labels, expected {len(src_paths)}.'
        labels = args.labels

    c = list()

    for label, src_path in zip(labels, src_paths):
        print(f'\n\n{label}: {src_path}')

        corners, w, h = extract_points(src_path, settings, state, extract_points_fns)
        print(f'corners {label}:', corners.shape)
        c.append(corners)

    # plot results
    if len(c) == 2:
        plot_combined(labels, c, w, h, args.plot_distortion_intensity)

    else:
        plot_individual(labels, c, w, h, args.plot_distortion_intensity)