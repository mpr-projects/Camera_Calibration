import cv2 as cv
import numpy as np
from target_functions import get_ro_idx_options


def get_initial_camera_matrix(settings):
    focal_length = settings['focal_length']
    w, h = settings['sensor_width_px'], settings['sensor_height_px']
    W, H = settings['sensor_width_mm'], settings['sensor_height_mm']

    camera_matrix = np.array([
        [focal_length * w / W, 0, w / 2],
        [0, focal_length * h / H, h / 2],
        [0, 0, 1]
    ])

    return camera_matrix


def get_calibration_flags(settings):
    flags = cv.CALIB_USE_INTRINSIC_GUESS

    if not settings.get('calibrate_tangential_distortion', False):
        flags += cv.CALIB_ZERO_TANGENT_DIST

    if not settings.get('calibrate_radial_k3', False):
        flags += cv.CALIB_FIX_K3

    if settings.get('calibrate_radial_k4_k5_k6', False):
        flags += cv.CALIB_RATIONAL_MODEL

    if settings.get('calibrate_tilt', False):
        flags += cv.CALIB_TILTED_MODEL

    return flags


def calibrate_std(settings, state, gray, objpoints, imgpoints, extended=False):
    camera_matrix = get_initial_camera_matrix(settings)
    flags = get_calibration_flags(settings)

    if extended:
        rv = cv.calibrateCameraExtended(
            objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=flags)
        # ret, camera_matrix, dist, rvecs, tvecs, stdDeviationsIntrinsics, _, _ = rv
        rv = rv[:3] + rv[5:6]

    else:
        rv = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=flags)
        # ret, camera_matrix, dist, rvecs, tvecs = rv
        rv = rv[:3]
    
    return rv


def calibrate_ro(settings, state, gray, objpoints, imgpoints, extended=False):
    camera_matrix = get_initial_camera_matrix(settings)
    flags = get_calibration_flags(settings)

    # for the release_object method we need three points that are taken as the base
    # plane of the coordinate system, the first and the last entry are automatically
    # chosen by OpenCV; the docs recommend to use the top right point as the third
    # point (Todo: find out why, when I read the paper it didn't seem like it would matter)
    # if there are multiple patterns then the first one is picked for the reference
    # # coordinate system
    pattern_types = settings['pattern_type']
    get_ro_idx_fn = get_ro_idx_options[pattern_types[0]]
    idx = get_ro_idx_fn(settings, 0)

    if extended:    
        rv = cv.calibrateCameraROExtended(
            objpoints, imgpoints, gray.shape[::-1], idx, camera_matrix, None, flags=flags)
        # ret, camera_matrix, dist, rvecs, tvecs, newObjPoints, stdDeviationsIntrinsics = rv[:7]
        objpoints = rv[5]
        rv = rv[:3] + rv[6:7]

    else:
        rv = cv.calibrateCameraRO(
            objpoints, imgpoints, gray.shape[::-1], idx, camera_matrix, None, flags=flags)
        # ret, camera_matrix, dist, rvecs, tvecs, newObjPoints = rv[:6]
        objpoints = rv[5]
        rv = rv[:3]

    print('Camera Matrix:', rv[1])
    print('Distortion:', rv[2])

    # update the adjusted object points for each pattern type
    n_ = 0
    objpoints = objpoints[0]

    for idx in range(len(settings['pattern_type'])):
        n = state[f'adjObjp_{idx}'].shape[0]
        state[f'adjObjp_{idx}'] = objpoints[n_:n_+n]
        n_ += n

    # rv == [ret, camera_matrix, dist, stdDeviationsIntrinsics]
    return rv


calibrate_options = dict(
    standard=calibrate_std,
    release_object=calibrate_ro,
)