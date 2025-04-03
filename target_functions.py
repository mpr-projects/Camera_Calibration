import copy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def init_base(settings):
    state = dict(
        results=list(), val_errs=list(), val_errs_max=list(), nss=list(),
        finalObjPoints=list())
    return state


aruco_dict_mapping = dict(
    DICT_5X5_100=cv.aruco.DICT_5X5_100,
    DICT_4X4_100=cv.aruco.DICT_4X4_100,
)


def settings_update_Charuco(settings):
    settings['predefined_dict'] = [settings['predefined_dict']]
    settings['square_length'] = [settings['square_length']]
    settings['marker_length'] = [settings['marker_length']]


def init_Charuco(settings, idx):
    pattern_size = settings['pattern_size'][idx]
    predefined_dict_name = settings['predefined_dict'][idx]
    square_length = settings['square_length'][idx]
    marker_length = settings['marker_length'][idx]

    predefined_dict = aruco_dict_mapping[predefined_dict_name]
    dictionary = cv.aruco.getPredefinedDictionary(predefined_dict)

    board = cv.aruco.CharucoBoard(pattern_size, square_length, marker_length, dictionary)

    charucoParams = cv.aruco.CharucoParameters()
    charucoParams.tryRefineMarkers = True

    detector = cv.aruco.CharucoDetector(board, charucoParams)

    d0, d1 = pattern_size[0] - 1, pattern_size[1] - 1
    objp = np.zeros((d0*d1, 3), np.float32) 
    objp[:,:2] = np.mgrid[0:d0, 0:d1].T.reshape(-1,2)

    # Todo: apply rotation and offset

    state = init_base(settings)
    state[f'detector_{idx}'] = detector
    state[f'objp_{idx}'] = objp
    state[f'adjObjp_{idx}'] = objp
    return state


def extract_points_Charuco(img, gray, settings, state, idx, plot=False):
    detector = state[f'detector_{idx}']
    corners, ids, markers, marker_ids = detector.detectBoard(gray)
    state[f'ids_{idx}'] = ids

    if plot and marker_ids is not None:
        cv.aruco.drawDetectedMarkers(img, markers, marker_ids)
        cv.imshow("Detected Markers", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    return ids is not None, corners


def plot_Charuco(img, settings, state, corners, ret, idx):
    ids = state[f'ids_{idx}']
    cv.aruco.drawDetectedCornersCharuco(img, corners, ids)
    cv.imshow('Charuco Corners', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_ro_idx_charuco(settings, idx):
    pattern_size = settings['pattern_size'][idx]
    return pattern_size[0] - 2


def settings_update_Chessboard(settings):
    settings['square_size_mm'] = [settings['square_size_mm']]


def init_Chessboard(settings, idx):
    pattern_size = settings['pattern_size'][idx]
    square_size_mm = settings['square_size_mm'][idx]

    objp = np.zeros((np.prod(pattern_size), 3), np.float32) 
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size_mm

    # adjust pose for multiple patterns
    angle, ox, oy = settings['pattern_pose'][idx]

    if angle != 0:
        rotMat = get_rotation_matrix(angle)
        objp = np.inner(rotMat, objp).T

    objp += np.array([ox, oy, 0])

    state = init_base(settings)
    state[f'objp_{idx}'] = objp
    state[f'adjObjp_{idx}'] = objp
    return state


def extract_points_Chessboard(img, gray, settings, state, idx, plot=False):
    pattern_size = settings['pattern_size'][idx]
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    if ret is True:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = settings.get('subPix_window_size', default=(11, 11))
        zeroZone = settings.get('subPix_zero_zone', default=(-1, -1))
        corners = cv.cornerSubPix(gray,corners, winSize, zeroZone, criteria)

    return ret, corners if ret else None


def plot_generic(img, settings, corners, ret, idx, title, proj_corners=None):
    pattern_size = settings['pattern_size'][idx]
    img = copy.copy(img)
    cv.drawChessboardCorners(img, pattern_size, corners, ret)

    if proj_corners is not None:
        cv.drawChessboardCorners(img, pattern_size, proj_corners, ret)

    img = img[1580:2100, 3780:4250]

    fig, ax = plt.subplots()
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
    plt.show()


def plot_Chessboard(img, settings, state, corners, ret, idx, proj_corners=None, title='Chessboard Corners'):
    plot_generic(img, settings, corners, ret, idx, title=title, proj_corners=proj_corners)


def find_pose_Chessboard(img, gray, camera_matrix, dist, settings, state, idx, use_adjusted_obj=True):
    ret, corners = extract_points_Chessboard(img, gray, settings, state, idx)

    if ret is False:
        print('Couldn\'t find image corners.')
        return False, None, None, None
    
    objp = state[f'adjObjp_{idx}'] if use_adjusted_obj else state[f'objp_{idx}']
    ret, rvec, tvec = cv.solvePnP(objp, corners, camera_matrix, dist)
    return ret, rvec, tvec, corners


def get_ro_idx_generic(settings, idx):
    pattern_size = settings['pattern_size'][idx]
    return pattern_size[0] - 1


def settings_update_ACircles(settings):
    settings['square_size_mm'] = [settings['square_size_mm'],]
    settings['filterByArea'] = [settings['filterByArea']]
    settings['minArea'] = [settings['minArea']]
    settings['maxArea'] = [settings['maxArea']]
    # settings['diameter'] = [settings['diameter'],]
    # settings['gap'] = [settings['gap'],]


def init_ACircles(settings, idx):
    pattern_size = settings['pattern_size'][idx]
    square_size_mm = settings['square_size_mm'][idx]

    # adjust blob detector, default settings are not suitable for images with high resolution
    blobParams = cv.SimpleBlobDetector_Params()

    blobParams.filterByColor = True
    blobParams.blobColor = 0

    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.95

    blobParams.filterByInertia = False
    blobParams.filterByCircularity = False

    blobParams.filterByArea = settings['filterByArea'][idx]
    blobParams.minArea = settings['minArea'][idx]
    blobParams.maxArea = settings['maxArea'][idx]

    detector = cv.SimpleBlobDetector_create(blobParams)

    # location of circles on pattern
    # circ_diameter = settings['diameter'][idx]
    # circ_gap = settings['gap'][idx]

    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    cnt = 0

    for i in range(pattern_size[1]):
        i_ = (i+0) * square_size_mm
        o = 0 if i % 2 == 0 else 1

        for j in range(pattern_size[0]):
            j_ = (2*j+o+0) * square_size_mm
            objp[cnt] = np.array([j_, i_, 0])
            cnt += 1

    # adjust pose for multiple patterns
    angle, ox, oy = settings['pattern_pose'][idx]

    if angle != 0:
        rotMat = get_rotation_matrix(angle)
        objp = np.inner(rotMat, objp).T

    objp += np.array([ox, oy, 0])

    state = init_base(settings)
    state[f'detector_{idx}'] = detector
    state[f'objp_{idx}'] = objp
    state[f'adjObjp_{idx}'] = objp
    return state


def extract_points_ACircles(img, gray, settings, state, idx, plot=False):
    detector = state[f'detector_{idx}']
    pattern_size = settings['pattern_size'][idx]

    ret, corners = cv.findCirclesGrid(gray, pattern_size, flags=cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING, blobDetector=detector)

    if ret is False:
        keypoints = detector.detect(gray)
        print('n_keypoints:', len(keypoints))
        im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        fig, ax = plt.subplots()
        ax.imshow(cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2RGB), interpolation='bicubic')
        fig.suptitle('Extracted Keypoints')
        plt.show()

    return ret, corners if ret else None


def plot_ACircles(img, settings, state, corners, ret, idx, proj_corners=None, title='ACircles Corners'):
    plot_generic(img, settings, corners, ret, idx, title=title, proj_corners=proj_corners)


def find_pose_ACircles(img, gray, camera_matrix, dist, settings, state, idx, use_adjusted_obj=True):
    ret, corners = extract_points_ACircles(img, gray, settings, state, idx)

    if ret is False:
        print('Couldn\'t find image corners.')
        return False, None, None, None
    
    # Todo: this may not work
    objp = state[f'adjObjp_{idx}'] if use_adjusted_obj else state[f'objp_{idx}']
    print('find pose acircles')
    print('objp:', objp.shape)
    print('corners:', corners.shape)
    ret, rvec, tvec = cv.solvePnP(objp, corners, camera_matrix, dist)

    return ret, rvec, tvec, corners


# we support using multiple calibration patterns in the same image, but it is also possible
# to provide just one; if one is provided then the variables used by the respective pattern
# types must be converted to lists so they are compatible with using multiple pattern types
settings_update_options = dict(
    charuco=settings_update_Charuco,
    chessboard=settings_update_Chessboard,
    acircles=settings_update_ACircles,
)


init_options = dict(
    charuco=init_Charuco,
    chessboard=init_Chessboard,
    acircles=init_ACircles,
)


extract_points_options = dict(
    charuco=extract_points_Charuco,
    chessboard=extract_points_Chessboard,
    acircles=extract_points_ACircles,
)

plot_options = dict(
    charuco=plot_Charuco,
    chessboard=plot_Chessboard,
    acircles=plot_ACircles,
)

# Todo: write function for charuco
find_pose_options = dict(
    charuco=None,
    chessboard=find_pose_Chessboard,
    acircles=find_pose_ACircles,
)

get_ro_idx_options = dict(
    charuco=get_ro_idx_charuco,
    chessboard=get_ro_idx_generic,
    acircles=get_ro_idx_generic,
)


def get_state(settings):
    pattern_types = settings['pattern_type']
    init_fns = [init_options[pattern_type] for pattern_type in pattern_types]

    state = init_base(settings)

    for idx, init_fn in enumerate(init_fns):
        state.update(init_fn(settings, idx))

    return state