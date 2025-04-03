import os
import sys
import chime
import cv2 as cv
import numpy as np


def find_pose_fn(img, gray, camera_matrix, dist, settings, state, extract_points_fns, plot_fns):
    corners_, objpoints_ = list(), list()

    for idx, extract_points_fn in enumerate(extract_points_fns):
        ret, corners = extract_points_fn(img, gray, settings, state, idx)

        if ret is False:
            print('Couldn\'t find image corners.')
            return False, None, None, None
        
        corners_.append(corners)
        objpoints_.append(state[f'adjObjp_{idx}'])

        if settings['plot']:
            plot_fns[idx](img, settings, state, corners, ret, idx)

    corners = np.concatenate(corners_, axis=0, dtype=np.float32)
    objpoints = np.concatenate(objpoints_, axis=0, dtype=np.float32)

    # Todo: check if this is really necessary ...
    # https://github.com/opencv/opencv/issues/8813
    ret, rvec, tvec = cv.solvePnP(objpoints, corners, camera_matrix, dist,
                                  flags=cv.SOLVEPNP_SQPNP)

    if ret is False:
        print('Couldn\'t find pose.')
        return False, None, None, None

    return ret, rvec, tvec, corners, objpoints



def validate(camera_matrix, dist, src_path, settings, state, extract_points_fns, plot_fns):
        fnames = [f for f in os.listdir(src_path) if f.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']]
        mean_err, n = 0, 0
        max_err = 0

        # undistorting the image will lead to either lot's of black pixels where we don't have
        # information (alpha=1) or it will lead to an image without black pixels but where parts
        # of the image have been cut away (alpha=0); whichever one we choose, we should only
        # compute our validation error on those target points that will end up in the final image
        w, h = settings['sensor_width_px'], settings['sensor_height_px']

        for fname in fnames:
            fname = os.path.join(src_path, fname)
            img = cv.imread(fname, flags=cv.IMREAD_COLOR+cv.IMREAD_IGNORE_ORIENTATION)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, rvec, tvec, corners, objpoints = find_pose_fn(
                img, gray, camera_matrix, dist, settings, state,
                extract_points_fns, plot_fns)

            if ret is False:
                print(f'Couldn\'t find pose of validation file {fname}.')
                chime.error()
                sys.exit()

            # this is how calibrateCamera/-RO compute ret (https://stackoverflow.com/questions/29628445/meaning-of-the-retval-return-value-in-cv2-calibratecamera)
            proj_corners, _ = cv.projectPoints(objpoints, rvec, tvec, camera_matrix, dist)

            err = cv.norm(corners, proj_corners, cv.NORM_L2)
            n += len(proj_corners)
            mean_err += err*err

            max_err_ = ((corners - proj_corners)**2).sum(axis=-1).max()
            max_err = max(max_err, max_err_)

        mean_err = (mean_err / n)**0.5
        return mean_err, max_err**0.5
        