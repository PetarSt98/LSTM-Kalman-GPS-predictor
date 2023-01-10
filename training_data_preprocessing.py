try:
    import sys

    if '/opt/project/common' not in sys.path:
        sys.path.append('common')
except:  # noqa: E722
    raise

import pandas as pd
import numpy as np
from trajectory_interpolator_inference import TrajectoryInterpolatorInference
import matplotlib
import pickle
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy


def interpolate_points_and_accel(old_tstamps, old_points, accl_x, accl_y, repeats=8):
    ti = TrajectoryInterpolatorInference(old_points, old_tstamps)

    interpolated_tstamps = np.zeros([8])
    interpolated_points = np.zeros([8, 2])
    theta = np.zeros([8])
    for t in range(repeats):
        i = t
        # t = t * 50 * 1e6
        t = t * 200 * 1e6
        tstamp = old_tstamps[-1] - t
        interpolated_tstamps[-i - 1] = tstamp
        interpolated_points[-i - 1, :] = ti.get_point(tstamp)

        point1 = ti.get_point(tstamp)
        point2 = ti.get_point(tstamp + 1 * 1e6)
        theta[-i - 1] = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])

    accl = np.zeros([len(accl_x), 2])
    accl[:, 0] = accl_x
    accl[:, 1] = accl_y
    ti = TrajectoryInterpolatorInference(accl, old_tstamps)

    interpolated_accl = np.zeros([8, 2])
    for t in range(repeats):
        i = t
        # t = t * 50 * 1e6
        t = t * 200 * 1e6
        tstamp = old_tstamps[-1] - t
        interpolated_tstamps[-i - 1] = tstamp
        interpolated_accl[-i - 1, :] = ti.get_point(tstamp)
    accl_x = interpolated_accl[:, 0]
    accl_y = interpolated_accl[:, 1]

    return interpolated_tstamps, interpolated_points, accl_x, accl_y, theta


def generate_ground_truth(old_points, old_tstamps, future_points, future_tstamps, accl_x, accl_y, accl_future):
    accl = np.zeros([len(accl_x), 2])
    accl[:, 0] = accl_x
    accl[:, 1] = accl_y

    cat_points = np.concatenate((old_points, future_points), 0)
    cat_tstamps = np.concatenate((old_tstamps, future_tstamps), 0)
    cat_accl = np.concatenate((accl, accl_future), 0)

    ti = TrajectoryInterpolatorInference(cat_points, cat_tstamps)

    ground_truth_tstamp = old_tstamps[-1]
    # ground_truth_tstamps = np.zeros([4])
    # ground_truth_points = np.zeros([4, 2])
    ground_truth_tstamps = np.zeros([1])
    ground_truth_points = np.zeros([1, 2])
    #for n in range(4):
    for n in range(1):
        #ground_truth_tstamp = ground_truth_tstamp + 50 * 1e6
        ground_truth_tstamp = ground_truth_tstamp + 200 * 1e6
        ground_truth_tstamps[n] = ground_truth_tstamp
        ground_truth_points[n, :] = ti.get_point(ground_truth_tstamp)

    angle_tstamps_2 = np.array(ground_truth_tstamps) + 1 * 1e6
    angle_tstamps_1 = np.array(ground_truth_tstamps)
    #ground_truth_theta = np.zeros([4])
    ground_truth_theta = np.zeros([1])
    for n, (t1, t2) in enumerate(zip(angle_tstamps_1, angle_tstamps_2)):
        point1 = ti.get_point(t1)
        point2 = ti.get_point(t2)
        ground_truth_theta[n] = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])

    ti = TrajectoryInterpolatorInference(cat_accl, cat_tstamps)

    #ground_truth_accl = np.zeros([4, 2])
    ground_truth_accl = np.zeros([1, 2])
   # for n, tstamps in enumerate(ground_truth_tstamps[:4]):
    for n, tstamps in enumerate(ground_truth_tstamps[:1]):
        ground_truth_accl[n, :] = ti.get_point(tstamps)

    ground_truth_accl_x = ground_truth_accl[:, 0]
    ground_truth_accl_y = ground_truth_accl[:, 1]

    return ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y, ground_truth_theta


def mlp_model(df):
    for i in range(len(df)):
        if i == 0:
            continue
        old_tstamps = df[i]['old_tstamps']
        tstamps = df[i]['tstamps']
        old_points = df[i]['ground_truth_old']
        accl_t = np.int_([df[i]['ACCEL'][k][0] for k in range(64)])
        accl_x = [df[i]['ACCEL'][k][1] for k in range(64)]
        accl_y = [df[i]['ACCEL'][k][2] for k in range(64)]
        accl = np.zeros([64, 2])
        accl[:, 0] = accl_x
        accl[:, 1] = accl_y

        ti = TrajectoryInterpolatorInference(accl, accl_t)

        tstamps_interpolate = np.linspace(old_tstamps[0], old_tstamps[-1], 20)

        ACCEL = []
        for n, t in enumerate(old_tstamps):
            # PTS[n, :] = ti.get_point(t)
            a_x, a_y = ti.get_point(t)
            ACCEL.append((a_x, a_y))

        df[i]['ACCEL'] = ACCEL
        df[i]['ACCEL_future'] = ti.get_point(tstamps[-1])

        ti_theta = TrajectoryInterpolatorInference(old_points, old_tstamps)
        angle_tstamps_2 = np.array(old_tstamps) + 1 * 1e6
        angle_tstamps_1 = np.array(old_tstamps)
        theta = []
        for t1, t2 in zip(angle_tstamps_1, angle_tstamps_2):
            point1 = ti_theta.get_point(t1)
            point2 = ti_theta.get_point(t2)
            theta.append(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        df[i]['theta'] = theta

    return df


def plot_trajectory(
        old_tstamps,
        future_tstamps,
        future_points,
        ground_truth_tstamps,
        old_points,
        old_points_interpolated,
        ground_truth_points,
        old_tstamps_interpolated,
        accl_y,
        ground_truth_accl_y,
        label,
        i
        ):

    if not (os.path.exists(f'/model_ckpt/interpolated_tracks/{label}')):
        os.mkdir(f'/model_ckpt/interpolated_tracks/{label}')

    all_points = np.zeros([16, 2])
    all_tstamps = np.zeros([16])
    all_tstamps[:8] = old_tstamps
    all_tstamps[8:] = future_tstamps
    all_points[:8, :] = old_points
    all_points[8:, :] = future_points

    ti_old = TrajectoryInterpolatorInference(old_points, old_tstamps)
    ti_all = TrajectoryInterpolatorInference(all_points, all_tstamps)

    curve_old = []
    for t in np.linspace(old_tstamps[0], old_tstamps[-1], 30):
        curve_old.append(ti_old.get_point(t))
    curve_old = np.array(curve_old)

    curve_all = []
    for t in np.linspace(old_tstamps[0], future_tstamps[-1], 30):
        curve_all.append(ti_all.get_point(t))
    curve_all = np.array(curve_all)

    old_tstamps_interpolated = (np.array(old_tstamps_interpolated) - old_tstamps[0])/1e6
    ground_truth_tstamps = (np.array(ground_truth_tstamps) - old_tstamps[0]) / 1e6
    if i - 8 == 57:
        a=11
    fig = plt.figure(figsize=(20, 20))
    # for index in range(len(old_tstamps_interpolated)):
    #     plt.text(old_points_interpolated[index, 1], old_points_interpolated[index, 0], old_tstamps_interpolated[index], size=7)
    for index in range(len(ground_truth_tstamps)):
        plt.text(ground_truth_points[index, 1], ground_truth_points[index, 0], ground_truth_tstamps[index], size=7)
    plt.plot(curve_old[:, 1], curve_old[:, 0], 'm')
    plt.plot(curve_all[:, 1], curve_all[:, 0], 'g')
    plt.plot(old_points[:, 1], old_points[:, 0], 'm*')
    plt.plot(old_points_interpolated[:, 1], old_points_interpolated[:, 0], 'rx')
    plt.plot(ground_truth_points[:, 1], ground_truth_points[:, 0], 'gx')
    plt.plot(future_points[:, 1], future_points[:, 0], 'mo')
    for index in range(len(future_tstamps)):
        plt.text(future_points[index, 1], future_points[index, 0], (future_tstamps[index] - old_tstamps[0])/1e6, size=7)
    for index in range(len(old_tstamps)):
        plt.text(old_points[index, 1], old_points[index, 0], (old_tstamps[index] - old_tstamps[0]) / 1e6,
                 size=7)

    plt.legend(['old curve', 'future curve', 'old_points', 'old_points interpolated', 'ground_truth_points', 'future_points'])
    plt.savefig(f'/model_ckpt/interpolated_tracks/{label}/frame{i - 8}_t.png')
    plt.close(fig)

    fig = plt.figure()
    for index in range(len(old_tstamps_interpolated)):
        plt.text(old_points_interpolated[index, 1], old_points_interpolated[index, 0], accl_y[index], size=7)
    for index in range(len(ground_truth_tstamps)):
        plt.text(ground_truth_points[index, 1], ground_truth_points[index, 0], ground_truth_accl_y[index], size=7)
    plt.plot(curve_old[:, 1], curve_old[:, 0], 'm')
    plt.plot(curve_all[:, 1], curve_all[:, 0], 'g')
    plt.plot(old_points[:, 1], old_points[:, 0], 'm*')
    plt.plot(old_points_interpolated[:, 1], old_points_interpolated[:, 0], 'rx')
    plt.plot(ground_truth_points[:, 1], ground_truth_points[:, 0], 'gx')
    plt.legend(['old curve', 'future curve', 'old_points', 'old_points interpolated', 'ground_truth_points'])
    plt.savefig(f'/model_ckpt/interpolated_tracks/{label}/frame{i - 8}_a.png')
    plt.close(fig)
    print(f'Frame {i - 8} finished')


def match_accel_tstamps(df, accl, accl_t, old_tstamps, accl_future, accl_t_future, future_tstamps, i):
    ti = TrajectoryInterpolatorInference(accl, accl_t)

    ACCEL = []
    for n, t in enumerate(old_tstamps):
        a_x, a_y = ti.get_point(t)
        ACCEL.append((a_x, a_y))

    df[i - 8]['ACCEL'] = ACCEL
    cat_accl = np.concatenate((accl, accl_future), 0)
    cat_accl_tstamps = np.concatenate((accl_t, accl_t_future), 0)
    ti = TrajectoryInterpolatorInference(cat_accl, cat_accl_tstamps)

    df[i - 8]['ACCEL_future'] = np.array([ti.get_point(t) for t in future_tstamps])
    return df


def calculate_theta(df, old_points, old_tstamps, i):
    ti_theta = TrajectoryInterpolatorInference(old_points, old_tstamps)
    angle_tstamps_2 = np.array(old_tstamps) + 1 * 1e6
    angle_tstamps_1 = np.array(old_tstamps)
    theta = []
    for t1, t2 in zip(angle_tstamps_1, angle_tstamps_2):
        point1 = ti_theta.get_point(t1)
        point2 = ti_theta.get_point(t2)
        theta.append(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
    df[i - 8]['theta'] = theta
    return df


def lstm_model(df, label, plot=True):
    for i in range(11+8, len(df)):
        old_tstamps = df[i-8]['old_tstamps']
        future_tstamps = df[i]['old_tstamps']
        future_points = df[i]['ground_truth_old']
        accl_future = np.zeros([64, 2])
        accl_future[:, 0] = np.array([df[i]['ACCEL'][k][1] for k in range(64)])
        accl_future[:, 1] = np.array([df[i]['ACCEL'][k][2] for k in range(64)])
        accl_t_future = np.int_([df[i]['ACCEL'][k][0] for k in range(64)])
        old_points = df[i-8]['ground_truth_old']
        accl_t = np.int_([df[i-8]['ACCEL'][k][0] for k in range(64)])
        accl_x = [df[i-8]['ACCEL'][k][1] for k in range(64)]
        accl_y = [df[i-8]['ACCEL'][k][2] for k in range(64)]
        accl = np.zeros([64, 2])
        accl[:, 0] = accl_x
        accl[:, 1] = accl_y

        df = match_accel_tstamps(df, accl, accl_t, old_tstamps, accl_future, accl_t_future, future_tstamps, i)

        df = calculate_theta(df, old_points, old_tstamps, i)

        accl_x = [df[i-8]['ACCEL'][k][0] for k in range(8)]
        accl_y = [df[i-8]['ACCEL'][k][1] for k in range(8)]
        accl_future = df[i-8]['ACCEL_future']

        old_tstamps_interpolated, old_points_interpolated, accl_x, accl_y, theta = interpolate_points_and_accel(old_tstamps, old_points, accl_x,
                                                                                      accl_y)

        ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y, ground_truth_theta = \
            generate_ground_truth(old_points, old_tstamps, future_points, future_tstamps, accl_x, accl_y, accl_future) # ovo mozda mora da zameni mesto sa prvom interpolacijom

        if plot:
            plot_trajectory(
                            old_tstamps,
                            future_tstamps,
                            future_points,
                            ground_truth_tstamps,
                            old_points,
                            old_points_interpolated,
                            ground_truth_points,
                            old_tstamps_interpolated,
                            accl_y,
                            ground_truth_accl_y,
                            label,
                            i
                            )

        ACCEL = []
        for a_x, a_y in zip(accl_x, accl_y):
            ACCEL.append((a_x, a_y))
        df[i-8]['old_tstamps'] = old_tstamps_interpolated
        df[i-8]['ground_truth_old'] = old_points_interpolated
        df[i-8]['ACCEL'] = ACCEL
        df[i-8]['theta'] = theta
        df[i-8]['ground_truth_points'] = ground_truth_points
        df[i-8]['ground_truth_tstamps'] = ground_truth_tstamps
        df[i-8]['ground_truth_accl_x'] = ground_truth_accl_x
        df[i-8]['ground_truth_accl_y'] = ground_truth_accl_y
        df[i-8]['ground_truth_theta'] = ground_truth_theta
    return df


class dictionary:
    def __init__(self):
        self.df = dict()
        self.cnt = 0

    def append(self, df_temp):
        for i in range(len(df_temp)):
            self.df[i + self.cnt] = df_temp[i]
        self.cnt = len(self.df)


if __name__ == '__main__':
    df = dictionary()
    labels = [
            "aslan-20220409114513",
            "aslan-20220502091737",
            "aslan-20220502092513",
            "aslan-20220504083146",
            "aslan-20220409115940",
            "aslan-20220504082807",
            "aslan-20220504082134",
            "aslan-20220629142208",
            "aslan-20220413125957"
            ] # TRAIN
    # labels = ['aslan-normal-20220104141948', 'aslan-20220408131739'] # VAL
    for i, label in enumerate(labels):
        df_temp = pd.read_pickle(f'/model_ckpt/pickled_bags/{label}.pickle')
        # df = mlp_model(df)
        df_temp = lstm_model(df_temp, label=label, plot=True)
        df_temp2 = dict()
        for j in range(11, len(df_temp)-9):
            df_temp2[j-11] = df_temp[j]
        df.append(df_temp2)
        print(f'Interpolation for {label} file finished, length of file {len(df.df)}')
    with open('/model_ckpt/npi_accel_fitted_one_pred_val.pickle', 'wb') as handle:
        pickle.dump(df.df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Pickle file saved')
