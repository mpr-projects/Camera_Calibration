import sys
import random
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_single_result(settings, ns, rvs, val_errs, val_errs_max, axs, no_validation=False, markers=False):
    # prepare data
    rets = [rv[0] for rv in rvs]
    camera_matrices = [rv[1] for rv in rvs]
    dists = [rv[2] for rv in rvs]

    fxs_px = [cm[0, 0] for cm in camera_matrices]
    fys_px = [cm[1, 1] for cm in camera_matrices]
    u0s_px = [cm[0, 2] for cm in camera_matrices]
    v0s_px = [cm[1, 2] for cm in camera_matrices]
    k1s_px = [dist[0, 0] for dist in dists]
    k2s_px = [dist[0, 1] for dist in dists]
    p1s_px = [dist[0, 2] for dist in dists]
    p2s_px = [dist[0, 3] for dist in dists]
    k3s_px = [dist[0, 4] for dist in dists]

    w, h = settings['sensor_width_px'], settings['sensor_height_px']
    W, H = settings['sensor_width_mm'], settings['sensor_height_mm']  

    fxs_mm = [fx * W / w for fx in fxs_px]
    fys_mm = [fy * H / h for fy in fys_px]

    u0s_adj_px = [u0 - w/2 for u0 in u0s_px]
    v0s_adj_px = [v0 - h/2 for v0 in v0s_px]

    # prepare colors
    cmap = plt.get_cmap('viridis')
    cmap_blue = plt.get_cmap('Blues')
    cmap_red = plt.get_cmap('Reds')
    cmap_green = plt.get_cmap('Greens')

    random_color = cmap(random.random())

    min_i = 0.25  # minimum intensity to avoid too light colors
    random_blue = cmap_blue(min_i + (1-min_i)*random.random())
    random_red = cmap_red(min_i + (1-min_i)*random.random())
    random_green = cmap_green(min_i + (1-min_i)*random.random())

    # plot data
    marker = 'o' if markers else None
    axs[0, 0].plot(ns, rets, color=random_color, marker=marker)

    if len(val_errs) == len(ns) and no_validation is False:
        axs[0, 0].plot(ns, val_errs, color=random_color, ls='--', marker=marker)

    if len(val_errs_max) == len(ns) and no_validation is False:
        axs[0, 0].plot(ns, val_errs_max, color=random_color, ls=':', marker=marker)

    axs[0, 1].plot(ns, u0s_adj_px, color=random_blue, marker=marker)
    axs[0, 1].plot(ns, v0s_adj_px, color=random_red, marker=marker)

    axs[0, 2].plot(ns, fxs_px, color=random_blue, marker=marker)
    axs[0, 2].plot(ns, fys_px, color=random_red, marker=marker)

    axs[1, 0].plot(ns, k1s_px, color=random_blue, marker=marker)
    axs[1, 0].plot(ns, k2s_px, color=random_red, marker=marker)
    axs[1, 0].plot(ns, k3s_px, color=random_green, marker=marker)

    axs[1, 1].plot(ns, p1s_px, color=random_blue, marker=marker)
    axs[1, 1].plot(ns, p2s_px, color=random_red, marker=marker)

    axs[1, 2].plot(ns, fxs_mm, color=random_blue, marker=marker)
    axs[1, 2].plot(ns, fys_mm, color=random_red, marker=marker)

    # if we don't compute the standard deviation then we're done
    if settings['compute_standard_deviation'] is False:
        return

    # prepare standard deviation data
    stds_fx = [rv[3][0][0] for rv in rvs]
    stds_fy = [rv[3][1][0] for rv in rvs]
    stds_u0 = [rv[3][2][0] for rv in rvs]
    stds_v0 = [rv[3][3][0] for rv in rvs]
    stds_k1 = [rv[3][4][0] for rv in rvs]
    stds_k2 = [rv[3][5][0] for rv in rvs]
    stds_p1 = [rv[3][6][0] for rv in rvs]
    stds_p2 = [rv[3][7][0] for rv in rvs]
    stds_k3 = [rv[3][8][0] for rv in rvs]

    stds_fx_mm = [std * W / w for std in stds_fx]
    stds_fy_mm = [std * H / h for std in stds_fy]

    z = settings['z_factor']
    fx_px_min = [fx - z*std for fx, std in zip(fxs_px, stds_fx)]
    fx_px_max = [fx + z*std for fx, std in zip(fxs_px, stds_fx)]
    fy_px_min = [fy - z*std for fy, std in zip(fys_px, stds_fy)]
    fy_px_max = [fy + z*std for fy, std in zip(fys_px, stds_fy)]
    fx_mm_min = [fx - z*std for fx, std in zip(fxs_mm, stds_fx_mm)]
    fx_mm_max = [fx + z*std for fx, std in zip(fxs_mm, stds_fx_mm)]
    fy_mm_min = [fy - z*std for fy, std in zip(fys_mm, stds_fy_mm)]
    fy_mm_max = [fy + z*std for fy, std in zip(fys_mm, stds_fy_mm)]
    u0_min = [u0 - z*std for u0, std in zip(u0s_adj_px, stds_u0)]
    u0_max = [u0 + z*std for u0, std in zip(u0s_adj_px, stds_u0)]
    v0_min = [v0 - z*std for v0, std in zip(v0s_adj_px, stds_v0)]
    v0_max = [v0 + z*std for v0, std in zip(v0s_adj_px, stds_v0)]
    k1_min = [k1 - z*std for k1, std in zip(k1s_px, stds_k1)]
    k1_max = [k1 + z*std for k1, std in zip(k1s_px, stds_k1)]
    k2_min = [k2 - z*std for k2, std in zip(k2s_px, stds_k2)]
    k2_max = [k2 + z*std for k2, std in zip(k2s_px, stds_k2)]
    k3_min = [k3 - z*std for k3, std in zip(k3s_px, stds_k3)]
    k3_max = [k3 + z*std for k3, std in zip(k3s_px, stds_k3)]
    p1_min = [p1 - z*std for p1, std in zip(p1s_px, stds_p1)]
    p1_max = [p1 + z*std for p1, std in zip(p1s_px, stds_p1)]
    p2_min = [p2 - z*std for p2, std in zip(p2s_px, stds_p2)]
    p2_max = [p2 + z*std for p2, std in zip(p2s_px, stds_p2)]

    # plot standard deviation data
    alpha = 0.1
    axs[0, 1].fill_between(ns, u0_min, u0_max, color=random_blue, alpha=alpha)
    axs[0, 1].fill_between(ns, v0_min, v0_max, color=random_red, alpha=alpha)
    axs[0, 2].fill_between(ns, fx_px_min, fx_px_max, color=random_blue, alpha=alpha)
    axs[0, 2].fill_between(ns, fy_px_min, fy_px_max, color=random_red, alpha=alpha)
    axs[1, 0].fill_between(ns, k1_min, k1_max, color=random_blue, alpha=alpha)
    axs[1, 0].fill_between(ns, k2_min, k2_max, color=random_red, alpha=alpha)
    axs[1, 0].fill_between(ns, k3_min, k3_max, color=random_green, alpha=alpha)
    axs[1, 1].fill_between(ns, p1_min, p1_max, color=random_blue, alpha=alpha)
    axs[1, 1].fill_between(ns, p2_min, p2_max, color=random_red, alpha=alpha)
    axs[1, 2].fill_between(ns, fx_mm_min, fx_mm_max, color=random_blue, alpha=alpha)
    axs[1, 2].fill_between(ns, fy_mm_min, fy_mm_max, color=random_red, alpha=alpha)


def plot(settings, state, no_validation=False, ignore=[], markers=False, image_inds=None):
    # set up axes
    fig, axs = plt.subplots(2, 3, sharex=True)
    axs[0, 0].set_title('Reprojection error [px]')
    axs[0, 1].set_title('Principal Point [px]')
    axs[0, 2].set_title('Focal Lengths [px]')
    axs[1, 0].set_title('Distortion $k_1$, $k_2$, $k_3$')
    axs[1, 1].set_title('Distortion $p_1$, $p_2$')
    axs[1, 2].set_title('Focal Lengths [mm]')

    # plot single entry
    use_validation = False
    max_ns = 0

    _val_errs_max = state.get('val_errs_max', [list()] * len(state['nss']))  # older results don't contain val_errs_max

    for idx, (ns, rvs, val_errs, val_errs_max) in enumerate(zip(state['nss'], state['results'], state['val_errs'], _val_errs_max)):
        if idx in ignore:
            continue

        max_ns = max(max_ns, max(ns))

        if image_inds is not None:  # only use data from runs with specified numbers of images
            ns_, rvs_, val_errs_, val_errs_max_ = list(), list(), list(), list()

            for idx in image_inds:
                if idx not in ns:
                    continue

                idx_ = ns.index(idx)
                ns_.append(idx)
                rvs_.append(rvs[idx_])
                val_errs_.append(val_errs[idx_] if len(val_errs) > 0 else None)
                val_errs_max_.append(val_errs_max[idx_] if len(val_errs_max) > 0 else None)
                
            ns, rvs, val_errs, val_errs_max = ns_, rvs_, val_errs_, val_errs_max_

        plot_single_result(settings, ns, rvs, val_errs, val_errs_max, axs,
                           no_validation=no_validation, markers=markers)

        if len(val_errs) == len(ns) and no_validation is False:
            use_validation = True

    # create labels
    axs[0, 0].plot([], [], color='black', label='Fitting')

    if use_validation:
        axs[0, 0].plot([], [], color='black', ls='--', label='Validation Mean')
        axs[0, 0].plot([], [], color='black', ls=':', label='Validation Max')
        
    axs[0, 0].set_xlim(0, max_ns+3)
    # axs[0, 0].set_ylim(0, 1.5)
    axs[0, 0].legend()

    blue_patch = mpatches.Patch(color='blue', label='$c_x$')
    red_patch = mpatches.Patch(color='red', label='$c_y$')
    axs[0, 1].legend(handles=[blue_patch, red_patch])
    axs[0, 1].set_xlabel('Number of Images')

    blue_patch = mpatches.Patch(color='blue', label='$f_x$')
    red_patch = mpatches.Patch(color='red', label='$f_y$')
    axs[0, 2].legend(handles=[blue_patch, red_patch])
    axs[1, 2].legend(handles=[blue_patch, red_patch])
    axs[1, 2].set_ylim(22.5, 24)
    axs[1, 2].set_xscale('linear')

    axs[1, 0].plot([], [], color='blue', label='$k_1$')
    axs[1, 0].plot([], [], color='red', label='$k_2$')
    axs[1, 0].plot([], [], color='green', label='$k_3$')
    axs[1, 0].legend()

    axs[1, 1].plot([], [], color='blue', label='$p_1$')
    axs[1, 1].plot([], [], color='red', label='$p_2$')
    axs[1, 1].legend()

    xticks = list(range(max_ns+4))

    if len(xticks) > 5:
        xticks = xticks[::len(xticks) // 5]

    for i in range(2):
        for j in range(3):
            axs[i, j].set_xticks(xticks)

    plt.show()


def plot_3d_target_points(settings, state, ignore=list()):
    objPoints = state['finalObjPoints']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('viridis')
    min_z, max_z = float('inf'), float('-inf')

    for i, obj_points in enumerate(objPoints):
        if i in ignore:
            continue
        
        print(f'Run {i+1} object points: {obj_points.shape}')
        obj_points = obj_points.reshape(-1, 3)
        print('adjusted shape:', obj_points.shape)
        color = cmap(1 / max(1, (len(objPoints) - 1)) * i)

        x = obj_points[:, 0]
        y = obj_points[:, 1]
        z = obj_points[:, 2]

        min_z = min(min_z, min(z))
        max_z = max(max_z, max(z))

        ax.scatter(x, y, z, color=color, marker='o', label=f'Run {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    factor = 4
    ax.set_zlim(factor*min_z, factor*max_z)
    ax.set_title('Plot of Estimated Target Points\n(z-axis not to scale)')

    ax.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot calibration results.')
    parser.add_argument('fpath', type=str, help='Path to the file containing the calibration results.')
    parser.add_argument('--no_validation', action='store_true', help='Don\'t plot validation reprojection error.')
    parser.add_argument('--plot_markers', action='store_true')
    parser.add_argument('--select_images', type=int, nargs='+', help='Indices of image to use in plotting.')
    parser.add_argument(
        '--ignore_srcs', type=int, nargs='+', default=[], help='Indices of source folders to ignore in plotting.'
    )
    parser.add_argument(
        '--plot_target', action='store_true',
        help='Plot the estimated 3D target points (only useful if the release_object method was used).')
    args = parser.parse_args()

    with open(args.fpath, 'rb') as f:
        settings, state = pickle.load(f)

    for idx, (_, src) in enumerate(settings['src_folders'].items()):
        xx = '[-] ' if idx in args.ignore_srcs else ''
        print(f'{idx}: {xx}{src}')
        
    plot(settings, state,
         no_validation=args.no_validation, ignore=args.ignore_srcs,
         markers=args.plot_markers, image_inds=args.select_images)

    if args.plot_target:
        plot_3d_target_points(settings, state, ignore=args.ignore_srcs)