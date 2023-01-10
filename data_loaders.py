import torch
import numpy as np
import copy
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


class custom_data_loader:
    def __init__(self, data, ref_time, batch_size=16):
        self.len = batch_size
        self.cnt = 0
        self.ref_time = ref_time
        self.batches = []
        self.batch = []
        self.ground_truth_batch = []
        self.ground_truth_batches = []
        self.data = data
        self.create_batches()

    def create_batches(self):
        for j in range(0, len(self.data)-0):
            future_points = self.data[j]['ground_truth'][-1, :]
            tstamps = self.data[j]['tstamps']
            old_points = self.data[j]['ground_truth_old']
            old_tstamps = self.data[j]['old_tstamps']
            accl_x = [self.data[j]['ACCEL'][k][0] for k in range(8)]
            accl_y = [self.data[j]['ACCEL'][k][1] for k in range(8)]
            theta = self.data[j]['theta'][-1]

            old_tstamps = copy.deepcopy(old_tstamps)
            future_tstamps = tstamps[-1]
            old_tstamps = (np.array(old_tstamps) - self.ref_time) / 1e9
            future_tstamps = (future_tstamps - self.ref_time) / 1e9

            future_points, old_tstamps, future_tstamps, old_points, accl_x, accl_y, theta = self.convert2tensor(
                future_points, old_tstamps, future_tstamps, old_points, accl_x, accl_y, theta
            )

            self.append_batch(old_tstamps, future_tstamps, old_points, theta, accl_x, accl_y)

            self.append_ground_truth(future_points)

        if len(self.batch) > 0:
            self.batches.append(torch.cat(self.batch, 0))
            self.batch = []
            self.ground_truth_batches.append(torch.cat(self.ground_truth_batch, 0))
            self.ground_truth_batch = []

    @staticmethod
    def convert2tensor(future_points, old_tstamps, future_tstamps, old_points, accl_x, accl_y, theta):
        future_points = torch.flatten(torch.Tensor(future_points))
        old_tstamps = torch.Tensor(old_tstamps)
        future_tstamps = torch.Tensor(np.array(future_tstamps)).reshape(1)
        old_points = torch.flatten(torch.Tensor(old_points))
        accl_x = torch.flatten(torch.Tensor(accl_x))
        accl_y = torch.flatten(torch.Tensor(accl_y))
        theta = torch.Tensor(np.array(theta)).reshape(1)
        return future_points, old_tstamps, future_tstamps, old_points, accl_x, accl_y, theta

    def append_batch(self, *kwargs):
        sample = torch.unsqueeze(torch.cat(kwargs, 0), 0)
        self.batch.append(sample)
        self.cnt += 1
        if self.cnt % self.len == 0:
            self.batches.append(torch.cat(self.batch, 0))
            self.batch = []

    def append_ground_truth(self, future_points):
        self.ground_truth_batch.append(torch.unsqueeze(future_points, 0))
        if self.cnt % self.len == 0:
            self.ground_truth_batches.append(torch.cat(self.ground_truth_batch, 0))
            self.ground_truth_batch = []

    def shuffle(self):
        all_data = torch.cat(self.batches)
        all_ground_truth = torch.cat(self.ground_truth_batches)

        indices = np.arange(all_data.size()[0])
        indices = np.random.permutation(indices)

        all_data = all_data[indices]
        all_ground_truth = all_ground_truth[indices]

        for j in range(1, all_data.size()[0]//self.len + 1):
            self.batches[j-1] = all_data[(j-1)*self.len:j*self.len]
            self.ground_truth_batches[j - 1] = all_ground_truth[(j - 1) * self.len:j * self.len]

        self.batches[-1] = all_data[j*self.len:]
        self.ground_truth_batches[-1] = all_ground_truth[j * self.len:]

    def annotate(self, threshold=1.0, repeat=2):
        all_data = torch.cat(self.batches)
        all_ground_truth = torch.cat(self.ground_truth_batches)
        all_data_annotated = torch.cat(self.batches)
        all_ground_truth_annotated = torch.cat(self.ground_truth_batches)

        for j in range(len(all_ground_truth)):
            if any(abs(all_data[j][-8:]) >= threshold):
                for _ in range(repeat):
                    all_data_annotated = torch.cat([all_data_annotated, torch.unsqueeze(all_data[j], 0)])
                    all_ground_truth_annotated = torch.cat([all_ground_truth_annotated, torch.unsqueeze(all_ground_truth[j], 0)])

        self.batches = []
        self.ground_truth_batches = []
        load_data_batch = []
        load_ground_truth_batch = []
        for j in range(len(all_ground_truth_annotated)):
            load_data_batch.append(torch.unsqueeze(all_data_annotated[j], 0))
            load_ground_truth_batch.append(torch.unsqueeze(all_ground_truth_annotated[j], 0))

            if len(load_ground_truth_batch) % self.len == 0:
                load_data_batch = torch.cat(load_data_batch)
                load_ground_truth_batch = torch.cat(load_ground_truth_batch)
                self.batches.append(load_data_batch)
                self.ground_truth_batches.append(load_ground_truth_batch)
                load_data_batch = []
                load_ground_truth_batch = []

        if len(load_ground_truth_batch) > 0:
            load_data_batch = torch.cat(load_data_batch)
            load_ground_truth_batch = torch.cat(load_ground_truth_batch)
            self.batches.append(load_data_batch)
            self.ground_truth_batches.append(load_ground_truth_batch)


class lstm_custom_data_loader:
    def __init__(self, data, batch_size=16):
        self.len = batch_size
        self.cnt = 0
        self.ref_time = 0
        self.batches = []
        self.batch = []
        self.ground_truth_batch = []
        self.ground_truth_batches = []
        self.data = data
        self.create_batches()

    def create_batches(self):
        for j in range(0, len(self.data)-0):

            old_points = self.data[j]['ground_truth_old']
            old_tstamps = self.data[j]['old_tstamps']
            accl_x = [self.data[j]['ACCEL'][k][0] for k in range(8)]
            accl_y = [self.data[j]['ACCEL'][k][1] for k in range(8)]
            theta = self.data[j]['theta']
            ground_truth_points = self.data[j]['ground_truth_points']
            ground_truth_tstamps = self.data[j]['ground_truth_tstamps']
            ground_truth_accl_x = self.data[j]['ground_truth_accl_x']
            ground_truth_accl_y = self.data[j]['ground_truth_accl_y']
            ground_truth_theta = self.data[j]['ground_truth_theta']

            self.ref_time = old_tstamps[0]
            old_tstamps = copy.deepcopy(old_tstamps)
            old_tstamps = (np.array(old_tstamps) - self.ref_time) / 1e9
            ground_truth_tstamps = copy.deepcopy(ground_truth_tstamps)
            ground_truth_tstamps = (np.array(ground_truth_tstamps) - self.ref_time) / 1e9

            ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y,\
            old_tstamps, old_points, accl_x, accl_y, theta, ground_truth_theta = self.convert2tensor(
                ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y,
                old_tstamps, old_points, accl_x, accl_y, theta, ground_truth_theta
            )

            self.append_batch(old_points[:, 0], old_points[:, 1], accl_x, accl_y, theta, old_tstamps)
            self.append_ground_truth(ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y, ground_truth_theta)

        if len(self.batch) > 0:
            self.batches.append(torch.cat(self.batch, 0))
            self.batch = []
            self.ground_truth_batches.append(torch.cat(self.ground_truth_batch, 0))
            self.ground_truth_batch = []

    @staticmethod
    def convert2tensor(
            ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y,
             old_tstamps, old_points, accl_x, accl_y, theta, ground_truth_theta
            ):
        ground_truth_points = torch.Tensor(ground_truth_points)
        ground_truth_tstamps = (torch.Tensor(ground_truth_tstamps))
        ground_truth_accl_x = torch.Tensor(ground_truth_accl_x)
        ground_truth_accl_y = (torch.Tensor(ground_truth_accl_y))
        old_tstamps = torch.Tensor(old_tstamps)
        old_points = (torch.Tensor(old_points))
        accl_x = torch.Tensor(accl_x)
        accl_y = torch.Tensor(accl_y)
        theta = torch.Tensor(np.array(theta))
        ground_truth_theta = torch.Tensor(ground_truth_theta)
        return ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y,\
               old_tstamps, old_points, accl_x, accl_y, theta, ground_truth_theta

    def append_batch(self, *kwargs):
        items = []
        for item in kwargs:
            items.append(torch.unsqueeze(item, 0))

        sample = torch.unsqueeze(torch.cat(items, 0), 0)
        self.batch.append(sample)
        self.cnt += 1
        if self.cnt % self.len == 0:
            self.batches.append(torch.cat(self.batch, 0))
            self.batch = []

    def append_ground_truth(self, ground_truth_points, ground_truth_tstamps, ground_truth_accl_x, ground_truth_accl_y, ground_truth_theta):
        ground_truth_list = [
            torch.unsqueeze(ground_truth_points[:, 0], 0),
            torch.unsqueeze(ground_truth_points[:, 1], 0),
            torch.unsqueeze(ground_truth_accl_x, 0),
            torch.unsqueeze(ground_truth_accl_y, 0),
            torch.unsqueeze(ground_truth_theta, 0),
            torch.unsqueeze(ground_truth_tstamps, 0),
        ]

        self.ground_truth_batch.append(torch.unsqueeze(torch.cat(ground_truth_list, 0), 0))

        if self.cnt % self.len == 0:
            self.ground_truth_batches.append(torch.cat(self.ground_truth_batch, 0))
            self.ground_truth_batch = []

    def shuffle(self):
        all_data = torch.cat(self.batches)
        all_ground_truth = torch.cat(self.ground_truth_batches)

        indices = np.arange(all_data.size()[0])
        indices = np.random.permutation(indices)

        all_data = all_data[indices]
        all_ground_truth = all_ground_truth[indices]

        for j in range(1, all_data.size()[0]//self.len + 1):
            self.batches[j-1] = all_data[(j-1)*self.len:j*self.len]
            self.ground_truth_batches[j - 1] = all_ground_truth[(j - 1) * self.len:j * self.len]

        self.batches[-1] = all_data[j*self.len:]
        self.ground_truth_batches[-1] = all_ground_truth[j * self.len:]

    def annotate(self, threshold=1.0, repeat=2):
        all_data = torch.cat(self.batches)
        all_ground_truth = torch.cat(self.ground_truth_batches)
        all_data_annotated = torch.cat(self.batches)
        all_ground_truth_annotated = torch.cat(self.ground_truth_batches)

        for j in range(len(all_ground_truth)):
            if any(abs(all_data[j, 3, :]) >= threshold):
                for _ in range(repeat):
                    all_data_annotated = torch.cat([all_data_annotated, torch.unsqueeze(all_data[j], 0)])
                    all_ground_truth_annotated = torch.cat([all_ground_truth_annotated, torch.unsqueeze(all_ground_truth[j], 0)])

        self.batches = []
        self.ground_truth_batches = []
        load_data_batch = []
        load_ground_truth_batch = []
        for j in range(len(all_ground_truth_annotated)):
            load_data_batch.append(torch.unsqueeze(all_data_annotated[j], 0))
            load_ground_truth_batch.append(torch.unsqueeze(all_ground_truth_annotated[j], 0))

            if len(load_ground_truth_batch) % self.len == 0:
                load_data_batch = torch.cat(load_data_batch)
                load_ground_truth_batch = torch.cat(load_ground_truth_batch)
                self.batches.append(load_data_batch)
                self.ground_truth_batches.append(load_ground_truth_batch)
                load_data_batch = []
                load_ground_truth_batch = []

        if len(load_ground_truth_batch) > 0:
            load_data_batch = torch.cat(load_data_batch)
            load_ground_truth_batch = torch.cat(load_ground_truth_batch)
            self.batches.append(load_data_batch)
            self.ground_truth_batches.append(load_ground_truth_batch)
