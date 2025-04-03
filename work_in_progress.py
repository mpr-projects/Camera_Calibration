import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from calibrate_functions import get_initial_camera_matrix


# This function is not currently used because it's not finished, it's intended to be used with multiple
# calibration targets; right now we have to manually specify their position and rotation, relative to
# each other; this function should find the relative positions automatically

# Todo: first set up states without rotation/translation, use the detectors to find keypoints with extract_points_fns,
#  assuming a simple camMat find the estimated pose with solvePnP, use resulting poses to align grids
def estimate_pattern_layout(settings, state, extract_points_fns):
    print('estimate pattern layout')
    camera_matrix = get_initial_camera_matrix(settings)
    dist = None  # np.zeros(5)  # Todo:

    src_folder = settings['src_folders']
    src_folder = src_folder[next(iter(src_folder))][0]
    fnames = [f for f in os.listdir(src_folder) if f.split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']]
    fnames.sort()
    fname = os.path.join(src_folder, fnames[0])

    print('fname:', fname)

    img = cv.imread(fname, flags=cv.IMREAD_COLOR+cv.IMREAD_IGNORE_ORIENTATION)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cmaps = [
        plt.get_cmap('Blues'),
        plt.get_cmap('Reds'),
        plt.get_cmap('Greens'),
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    first_camPos = None

    for idx, extract_points_fn in enumerate(extract_points_fns):
        ret, corners = extract_points_fn(img, gray, settings, state, idx)
        print('corners.shape:', corners.shape)
        print('objpoints.shape:', state[f'objp_{idx}'].shape)
        ret, rvec, tvec = cv.solvePnP(state[f'objp_{idx}'], corners, camera_matrix, None)
        print('ret:', ret)
        print('rvec:', rvec)
        print('tvec:', tvec)
        rotMat, _ = cv.Rodrigues(rvec)
        print('rotMat:', rotMat)

        T = np.eye(4)
        T[:3, :3] = rotMat
        T[:3, 3] = tvec[:, 0]
        print('T:', T)

        # Todo: I know how to invert this analytically ...
        T_inv = np.linalg.inv(T)
        print('T_inv:', T_inv)
        camPos = T_inv[:3, -1]
        print('camPos:', camPos)

        if first_camPos is None:
            first_camPos = camPos
            offset = np.zeros(3)

        else:
            offset = first_camPos - camPos
            print('offset:', offset)

        obj_coords = state[f'objp_{idx}']
        # obj_coords = obj_coords[:20]
        oxc = obj_coords[:, 0]
        oyc = obj_coords[:, 1]

        minv = 0.2
        colors = cmaps[idx](minv + (1-minv)*(len(obj_coords) - np.arange(len(obj_coords))) / len(obj_coords))

        # plt.scatter(oxc, oyc, color=colors)
        # plt.axis('equal')
        # plt.gca().invert_yaxis()
        # plt.show()

        ax.scatter(oxc, oyc, 0, color=colors)  # alpha=alpha)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.scatter(camPos[0], camPos[1], -camPos[2], color=colors[0])

    plt.show()
    
