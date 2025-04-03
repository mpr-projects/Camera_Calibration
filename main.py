import os
import yaml
import time
import chime
import pickle
import random
import datetime
import cv2 as cv
import numpy as np
import argparse


from target_functions import (
    get_state, settings_update_options, plot_options, extract_points_options)
from validation_functions import validate
from calibrate_functions import calibrate_options
import plot_results
from plot_helper import plot_target_layout



def calibrate(fnames, settings, state, points=None, extended=False):
    # passing points allows for efficient incremental calibration
    if points is None:
        objpoints = []
        imgpoints = []
        
    else:
        objpoints, imgpoints = points

    n = 0

    for fname in fnames:
        n += 1

        if len(objpoints) >= n:  # skip files that have already been covered
            continue

        fname = os.path.join(src_path, fname)
        print('fname:', fname)
        img = cv.imread(fname, flags=cv.IMREAD_COLOR+cv.IMREAD_IGNORE_ORIENTATION)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        imgpoints_ = list()
        objpoints_ = list()

        for idx, extract_points_fn in enumerate(extract_points_fns):
            ret, corners = extract_points_fn(img, gray, settings, state, idx)
            assert ret, f'Couldn\'t find corners (#{idx}) in {fname}.'

            imgpoints_.append(corners)
            objpoints_.append(state[f'objp_{idx}'])

            if settings['plot']:
                plot_fns[idx](img, settings, state, corners, ret, idx)

        imgpoints_ = np.concatenate(imgpoints_, axis=0, dtype=np.float32)
        objpoints_ = np.concatenate(objpoints_, axis=0, dtype=np.float32)

        objpoints.append(objpoints_)
        imgpoints.append(imgpoints_)

    rv = calibrate_fn(settings, state, gray, objpoints, imgpoints, extended=extended)
    # rv == [ret, camera_matrix, dist] or [ret, camera_matrix, dist, stdDeviationsIntrinsics]
    return objpoints, imgpoints, rv


def run_once(settings, state, fnames, src_path_validation=None):
    extended = settings['compute_standard_deviation']
    min_ns = settings.get('minimum_number_of_pictures', 3)

    rvs, val_errs, val_errs_max = list(), list(), list()
    ns = list(range(min_ns, len(fnames)+1))

    objpoints = []
    imgpoints = []

    for l in ns:
        print(f'Using {l}/{max(ns)} images')
        objpoints, imgpoints, rv = calibrate(
            fnames[:l], settings, state, points=[objpoints, imgpoints], extended=extended)
        rvs.append(rv)

        print(f'Reprojection Error: {rv[0]:.4e}')

        if src_path_validation not in [None, 'None', 'none']:
            camera_matrix, dist = rv[1:3]
            val_err, val_err_max = validate(
                camera_matrix, dist, src_path_validation, settings, state,
                extract_points_fns, plot_fns)
            val_errs.append(val_err)
            val_errs_max.append(val_err_max)
            print(f'Validation Error: {val_err:.3e}, {val_err_max:.3e}')

        print('----------------------------------------\n')

    # Todo: if I want to parallelize this code then the lines below should go to run_once
    # ----------------------------------------------------------------
    state['results'].append(rvs)
    state['val_errs'].append(val_errs)
    state['val_errs_max'].append(val_errs_max)
    state['nss'].append(ns)

    # save all adjusted objpoints in one array, no need to save them separately
    finalObjPoints_ = list()
    
    for idx in range(len(settings['pattern_type'])):
        finalObjPoints_.append(state[f'adjObjp_{idx}'])

    finalObjPoints_ = np.concatenate(finalObjPoints_, axis=0)
    state['finalObjPoints'].append(finalObjPoints_)

    if settings['plot']:
        plot_results.plot(settings, state)


if __name__ == '__main__':
    start_time = time.time()

    # get arguments
    parser = argparse.ArgumentParser('CalibrateCamera')
    parser.add_argument('settings_file', help='File containing the settings.')
    parser.add_argument('--dont_save', action='store_true', help='Don\'t save the results.')
    args = parser.parse_args()

    # load settings
    with open(args.settings_file) as f:
        settings = yaml.safe_load(f)

    src_paths = settings['src_folders']
    pattern_types = settings['pattern_type']
    calibrate_type = settings['calibration_type']
    clamera_model = settings['camera_model']

    seed = settings['seed']
    random.seed(seed)

    # the code is set up for using multiple calibration patterns, so we create
    # a list of calibration patterns even if we just use one
    if isinstance(pattern_types, str):
        settings_update_fn = settings_update_options[pattern_types]
        settings_update_fn(settings)
        pattern_types = settings['pattern_type'] = [pattern_types,]
        settings['pattern_size'] = [settings['pattern_size'],]
        settings['pattern_pose'] = [[0, 0, 0],]

    # set up global functions
    calibrate_fn = calibrate_options[calibrate_type]
    extract_points_fns = [extract_points_options[pattern_type] for pattern_type in pattern_types]
    plot_fns = [plot_options[pattern_type] for pattern_type in pattern_types]

    # set up state
    state = get_state(settings)

    # visualize layout of patterns
    if len(pattern_types) > 1:
        plot_target_layout(settings, state)

    # run tests
    for idx, (src_path, src_path_validation) in src_paths.items():
        fnames = [f for f in os.listdir(src_path) if f.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']]
        fnames.sort()

        if settings.get('shuffle_images', True):
            random.shuffle(fnames)
        
        run_once(settings, state, fnames, src_path_validation=src_path_validation)

    print(f'Runtime: {time.time()-start_time:.2f}sec')

    # save/plot results
    if not args.dont_save:
        timecode = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pattern_type = '_'.join(pattern_types)
        fname = f'results_{clamera_model}_{pattern_type}_{calibrate_type}_{timecode}.pkl'

        output_folder = os.path.join(os.path.dirname(args.settings_file), 'outputs')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, fname)

        with open(output_file, 'wb') as f:
            for idx in range(len(pattern_types)):
                state[f'detector_{idx}'] = None
            pickle.dump([settings, state], f)

        print(f'Saved output to {output_file}.')

    chime.success()

    plot_results.plot(settings, state)
    cv.destroyAllWindows()