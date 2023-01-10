
import numpy as np
from trajectory_interpolator_inference import TrajectoryInterpolatorInference
from gps_predictor_models import ImuGpsCombinedPredictor
from data_loaders import lstm_custom_data_loader, custom_data_loader
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import time
import onnx, onnxruntime
import os
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EpochRunner:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion[0]
        self.L1loss = criterion[1]
        self.optimizer = optimizer
        self.avg_loss = 0
        self.avg_loss_train = 0

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def run_epoch_lstm(self, batches, ground_truth_batches):
        self.optimizer.zero_grad()
        out, out_full = self.model(batches)
        loss1 = self.criterion(out_full, torch.cat((batches[:, :, 1:], ground_truth_batches), 2))
        loss2 = self.L1loss(out[:, :2, :], ground_truth_batches[:, :2, :].to(device))
        loss = loss1 + loss2 * 10
        loss.backward()
        self.optimizer.step()
        self.avg_loss_train += loss.item()

    def run_val_epoch_lstm(self, batches, ground_truth_batches):
        out, out_full = self.model(batches)
        loss1 = self.criterion(out_full, torch.cat((batches[:, :, 1:], ground_truth_batches), 2))
        loss2 = self.L1loss(out[:, :2, :], ground_truth_batches[:, :2, :].to(device))
        loss = loss1 + loss2 * 10
        self.avg_loss += loss.item()

    def print_loss(self, epoch, len):
        print(f'Epoch: {epoch} \t Loss_val: {self.avg_loss / len} \t Loss_train: {self.avg_loss_train / len}')
        self.avg_loss = 0
        self.avg_loss_train = 0

    def generate_lstm_preditions(self, data_loader):

        def predict(batch, ground_truth):
            out, _ = self.model(batch.to(device))
            loss = self.L1loss(out[:, :2, :], ground_truth[:, :2, :].to(device))
            return out.detach().to('cpu').numpy(), loss.item()

        output = [predict(batch, ground_truth) for batch, ground_truth in
                  zip(data_loader.batches, data_loader.ground_truth_batches)]

        pred = []
        pred_time = []
        pred_accl = []
        pred_theta = []
        ground_truth_pts = []
        loss = []
        for tuples_out, gt in zip(output, data_loader.ground_truth_batches):
            for predictions, gt_pts in zip(tuples_out[0], gt):
                pred.append(np.array([predictions[0], predictions[1]]))
                pred_accl.append(np.array([predictions[2], predictions[3]]))
                pred_theta.append(np.array(predictions[4]))
                pred_time.append(np.array(predictions[5]))
                ground_truth_pts.append(np.array([gt_pts[0].numpy(), gt_pts[1].numpy()]))
            loss.append(tuples_out[1])
        return pred, pred_accl, pred_theta, pred_time, ground_truth_pts, loss

    def scheduler_step(self, scheduler, data_loader, epoch):
        scheduler.step(self.avg_loss / data_loader.cnt)
        if scheduler.state_dict()['num_bad_epochs'] == 0:
            scheduler.state_dict()['best'] = self.avg_loss / data_loader.cnt
            scheduler.state_dict()['last_epoch'] = epoch


def generate_data_for_plot(df, i):
    points = df[i]['ground_truth']
    tstamps = df[i]['tstamps']
    old_points = df[i]['ground_truth_old']
    old_tstamps = df[i]['old_tstamps']
    X = df[i]['X']
    Y = df[i]['Y']
    old_ti = TrajectoryInterpolatorInference(old_points, old_tstamps)
    ti = TrajectoryInterpolatorInference(points, tstamps)

    tstamps_interpolate = np.linspace(tstamps[0], tstamps[-1], 20)
    tstamps_interpolate_old = np.linspace(old_tstamps[0], old_tstamps[-1] + 200000000, 20)
    pts = np.zeros([20, 2])
    for n in range(20):
        pts[n, :] = ti.get_point(tstamps_interpolate[n])
    old_pts = np.zeros([20, 2])
    for n in range(20):
        old_pts[n, :] = old_ti.get_point(tstamps_interpolate_old[n])

    return old_pts, pts, X, Y, points


def lstm_plotting(df,i, pred, ground_truth_pts, pred_time):
    old_pts, pts, X, Y, points = generate_data_for_plot(df, i)

    old_points = df[i]['ground_truth_old']
    old_tstamps = df[i]['old_tstamps']
    ground_turth_tstamps = df[i]['ground_truth_tstamps']
    # ground_turth_tstamps = (np.array(ground_turth_tstamps) - old_tstamps[0]) / 1e9
    old_tstamps = (np.array(old_tstamps) - old_tstamps[0]) / 1e9

    fig = plt.figure()
    plt.plot(old_pts[:, 1], old_pts[:, 0], 'm')
    plt.plot(pts[:, 1], pts[:, 0], 'g')
    plt.plot(Y, X, 'bx')
    plt.plot(points[:, 1], points[:, 0], 'rx')
    plt.plot(pred[i - 0][1], pred[i - 0][0], 'yx')
    for index in range(len(pred[i - 0][1])):
        plt.text(pred[i - 0][1][index], pred[i - 0][0][index], pred_time[i - 0][index], size=7)
    plt.plot(ground_truth_pts[i - 0][1], ground_truth_pts[i - 0][0], 'kx')
    # for index in range(len(ground_truth_pts[i - 0][0])):
    #     plt.text(ground_truth_pts[i - 0][1][index], ground_truth_pts[i - 0][0][index], ground_turth_tstamps[index], size=7)
    plt.plot(old_points[:, 1], old_points[:, 0], 'gx')
    for index in range(len(old_points[:, 1])):
        plt.text(old_points[index, 1], old_points[index, 0], old_tstamps[index], size=7)
    plt.legend(['Ground truth fitted curve', 'Ground truth fitted curve + future', 'Original Predictions', 'Points',
                'Lstm predictions', 'Ground truth for predictions', 'interpolated points'])
    plt.savefig(f'./kalman_plots/frame_{i}.png')
    plt.close(fig)


def lstm_model(epoch_num):
    model = ImuGpsCombinedPredictor().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {pytorch_total_params}\n')

    model.train()

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, threshold=0.0001,
                                                     threshold_mode='rel', min_lr=5e-5, verbose=True)

    with open(args.train_dataset, "rb") as pth:
        df_train = pickle.load(pth)

    with open(args.train_dataset, "rb") as pth:
        df_val = pickle.load(pth)

    data_loader = lstm_custom_data_loader(df_train, 1024)
    data_loader_val = lstm_custom_data_loader(df_val, 1024)
    # data_loader.annotate(threshold=0.6, repeat=7)

    epoch_runner = EpochRunner(model, (criterion1, criterion2), optimizer)
    if os.path.exists(args.model_pth):
        epoch_runner.load_model(args.model_pth)

    min_loss = 99999
    for epoch in range(epoch_num):
        data_loader.shuffle()

        model.train()
        for batch, ground_truth in zip(data_loader.batches, data_loader.ground_truth_batches):
            epoch_runner.run_epoch_lstm(batch.to(device), ground_truth.to(device))

        model.eval()
        for batch, ground_truth in zip(data_loader_val.batches, data_loader_val.ground_truth_batches):
            epoch_runner.run_val_epoch_lstm(batch.to(device), ground_truth.to(device))

        epoch_runner.scheduler_step(scheduler, data_loader, epoch)

        if min_loss > epoch_runner.avg_loss:
            min_loss = epoch_runner.avg_loss
            model.eval()
            torch.save(model.state_dict(), args.model_pth)
            torch.onnx.export(model,  # model being run
                              torch.randn(1024, 6, 8, requires_grad=True).to(device),  # model input (or a tuple for multiple inputs)
                              args.onnx_model_pth,  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=10,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input'],  # the model's input names
                              output_names=['output'],  # the model's output names
                              dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                            'output': {0: 'batch_size'}})

        epoch_runner.print_loss(epoch, data_loader.cnt)

    print('================ Finished training ================\n')
    model.eval()
    torch.save(model.state_dict(), f'/model_ckpt/gps_interpolation_model_over_fitted.pt')
    torch.onnx.export(model,  # model being run
                      torch.randn(1024, 6, 8, requires_grad=True).to(device),  # model input (or a tuple for multiple inputs)
                      f'/model_ckpt/gps_interpolation_model_over_fitted.onnx', # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print('================ Model saved ================\n')

    model.eval()
    data_loader = lstm_custom_data_loader(df_val, 1)
    pred, pred_accl, pred_theta, pred_time, ground_truth_pts, loss = epoch_runner.generate_lstm_preditions(data_loader)

    print('================ Generating predictions ================\n')
    for i, l in enumerate(loss):
        print(f'Frame: {i} \t Error: {l}')

    print('================= Compare ONNX and Pytorch ===================\n')
    x = data_loader_val.batches[0]

    epoch_runner.load_model(args.model_pth)
    model = epoch_runner.model
    onnx_model = onnx.load(args.onnx_model_pth)
    onnx.checker.check_model(onnx_model)

    # Give providers (GPU and CPU) some operators will be done by CPU instead of GPU
    providers = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(args.onnx_model_pth, providers=providers)

    ort_inputs = {ort_session.get_inputs()[0].name: x.cpu().numpy()}
    ort_outs, ort_outs_full = ort_session.run(None, ort_inputs)

    out, out_full = model(x.to(device))

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(out.cpu().detach().numpy(), ort_outs, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(out_full.cpu().detach().numpy(), ort_outs_full, rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    print('================ Plotting ================\n')
    for i in range(0, len(df_val)-0):
        lstm_plotting(df_val, i, pred, ground_truth_pts, pred_time)

    print('================ Finished plotting ================')


def kalman_filter(pos, speed, accel, dt):
    # Initialize state matrix and covariance matrix
    x = np.array([[pos[0]], [pos[1]], [speed[0]], [speed[1]]])
    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Initialize transition matrix
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Initialize measurement matrix
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    # Initialize measurement noise
    R = np.array([[1, 0], [0, 1]])

    # Initialize process noise
    Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]])

    # Predict step
    x_pred = np.dot(F, x) + np.array([[0], [0], [accel[0]], [accel[1]]]) * dt
    P_pred = np.dot(F, P) + Q

    # Update step
    K = np.dot(P_pred, H.T) / (np.dot(H, np.dot(P_pred, H.T)) + R)
    x = x_pred + np.dot(K, (np.array([[pos[0]], [pos[1]]]) - np.dot(H, x_pred)))
    P = P_pred - np.dot(K, np.dot(H, P_pred))

    return x[0][0], x[1][0], x[2][0], x[3][0]


parser_gps = argparse.ArgumentParser()
parser_gps.add_argument('--model_pth', type=str, required=True)
parser_gps.add_argument('--onnx_model_pth', type=str, required=True)
parser_gps.add_argument('--train_dataset', type=str, required=True)
parser_gps.add_argument('--val_dataset', type=str, required=True)
args = parser_gps.parse_args()
if __name__ == '__main__':
    print(device)
    torch.cuda.empty_cache()
    epoch_num = 1000
    lstm_model(epoch_num)
