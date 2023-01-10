import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import torch

from data_loaders import lstm_custom_data_loader, custom_data_loader
from trajectory_interpolator_inference import TrajectoryInterpolatorInference


class KalmanFilter:
    def __init__(self, initial_pos, initial_vel, process_noise, measurement_noise):
        self.state = np.array([initial_pos[0], initial_pos[1], initial_vel[0], initial_vel[1]])
        self.covariance = np.eye(4)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, accel, dt):
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[dt**2, 0], [0, dt**2], [dt, 0], [0, dt]])
        self.state = A @ self.state + B @ accel
        self.covariance = A @ self.covariance @ A.T + self.process_noise

    def correct(self, measurement):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        measurement_prediction = H @ self.state
        residual = measurement - measurement_prediction
        S = H @ self.covariance @ H.T + self.measurement_noise
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ residual
        self.covariance = (np.eye(4) - K @ H) @ self.covariance

    def get_state(self):
        return self.state[:2]


def plotting(old_points, j, pred, point_gt):
    fig = plt.figure()
    plt.plot(old_points[1, :], old_points[0, :], 'xr')
    plt.plot(point_gt[1], point_gt[0], 'xk')
    plt.plot(pred[1], pred[0], 'xy')
    plt.legend(['Observed points', 'Ground truth point', 'Predictions'])
    plt.savefig(f'./kalman_plots/frame_{j}.png')
    plt.close(fig)



parser_gps = argparse.ArgumentParser()
parser_gps.add_argument('--dataset_pth', type=str, required=True)
args = parser_gps.parse_args()
if __name__ == '__main__':
    with open(args.dataset_pth, "rb") as pth:
        df_val = pickle.load(pth)

    # Set the time step (dt) and the number of time steps to simulate (num_steps)
    dt = 0.05

    # Initialize lists to store the true and predicted positions of the object
    true_positions = []
    predicted_positions = []

    data_loader = lstm_custom_data_loader(df_val, 1)
    for j, (batch, ground_truth) in enumerate(zip(data_loader.batches, data_loader.ground_truth_batches)):
        batch = batch.numpy()
        ground_truth = ground_truth.numpy()

        old_points = batch[0, :2, :]
        accel = batch[0, 2:4, :]
        theta = batch[0, 4, :]
        old_tstmps = batch[0, 5, :]

        point_gt = ground_truth[0, :2, 0]
        accel_gt = ground_truth[0, 2:4, 0]
        theta_gt = ground_truth[0, 4, 0]
        tstmps_gt = ground_truth[0, 5, 0]

        velocity = (old_points[:, 1] - old_points[:, 0]) / dt

        # Add the current position of the object to the true positions list
        true_positions.append(point_gt)

        # Initialize the Kalman filter with the initial position, velocity, and acceleration of the object
        kf = KalmanFilter(initial_pos=old_points[:, 0], initial_vel=velocity, process_noise=np.eye(4) * 0.05,
                          measurement_noise=np.eye(2) * 0.05)

        for i in range(1, old_tstmps.shape[-1]):

            position = old_points[:, i]
            kf.predict(accel[:, i], dt)
            kf.correct(position)

        # Predict the future position of the object using the Kalman filter
        kf.predict(accel_gt, dt)
        predicted_positions.append(kf.get_state())
        print(f'Estimated position: {kf.get_state()}\tReal position: {point_gt}')

        # if 410 > j > 400:
        #     plotting(old_points, j, kf.get_state(), point_gt)

    fig = plt.figure()
    plt.plot(np.array(true_positions)[300:600, 0], np.array(true_positions)[300:600, 1], label='True position')
    plt.plot(np.array(predicted_positions)[300:600, 0], np.array(predicted_positions)[300:600, 1],label='Predicted position')
    plt.legend()
    plt.savefig('Kalman_example.png')


    print(f'Average error: {np.mean(np.sqrt(np.sum(np.square(np.array(true_positions) - np.array(predicted_positions)), axis=1)))}m')
    plt.show()
