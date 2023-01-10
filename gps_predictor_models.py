from torch import nn
import torch
import copy
#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class imuGpsCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(42, 30),
            nn.LeakyReLU(0.2, True),
            nn.Linear(30, 30),
            nn.LeakyReLU(0.2, True),
            nn.Linear(30, 30),
            nn.LeakyReLU(0.2, True),
            nn.Linear(30, 30),
            #nn.Dropout(0.1),
            nn.LeakyReLU(0.2, True),
            nn.Linear(30, 2)
        )

    def forward(self, x):
        return self.fc(x)


class ImuGpsCombinedPredictor(nn.Module):
    def __init__(self, hidden_dim=100):
        super(ImuGpsCombinedPredictor, self).__init__()
        self.n_hidden = hidden_dim

        self.linear1 = nn.Linear(6, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 6)

        # self.lstm1 = nn.LSTMCell(hidden_dim, self.n_hidden)
        # self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        # self.lstm3 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lstm1 = nn.LSTM(hidden_dim, self.n_hidden, 2, batch_first=True)
        # self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden, 1, batch_first=True)
        # self.lstm3 = nn.LSTM(self.n_hidden, self.n_hidden, 1, batch_first=True)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, future=1):
        outputs_full = []
        outputs = []
        batch_size = x.size(0)
        h_t = torch.zeros([2, batch_size, self.n_hidden], dtype=torch.float32).to(device)
        c_t = torch.zeros([2, batch_size, self.n_hidden], dtype=torch.float32).to(device)
        #h_t_2 = torch.zeros(batch_size, self.n_hidden, dtype=torch.float32).to(device)
        #c_t_2 = torch.zeros(batch_size, self.n_hidden, dtype=torch.float32).to(device)
        output_t = []
        mean_v = torch.mean(x, dim=2)
        std_v = torch.std(x, dim=2)

        for i in range(8):
            input_t = (x[:, :, i] - mean_v) / std_v
            input_t = self.linear1(input_t)
            input_t = torch.unsqueeze(self.leaky_relu(input_t), 1)
            h_t_last, (h_t, c_t) = self.lstm1(input_t, (h_t, c_t))
            output_t = self.linear4(h_t_last[:, 0, :])
            outputs_full.append(torch.unsqueeze((output_t * std_v) + mean_v, 2))
        outputs.append(torch.unsqueeze((output_t * std_v) + mean_v, 2))

        # for i in range(8):
        #     input_t = (x[:, :, i] - mean_v) / std_v
        #     input_t = self.linear1(input_t)
        #     input_t = self.leaky_relu(input_t)
        #     input_t = torch.unsqueeze(input_t, 1)
        #     h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        #     #h_t_2, c_t_2 = self.lstm2(h_t, (h_t_2, c_t_2))
        #     output_t = self.linear4(h_t[0, :, :])
        #     outputs_full.append(torch.unsqueeze((output_t * std_v) + mean_v, 2))
        # outputs.append(torch.unsqueeze((output_t * std_v) + mean_v, 2))

        # for i in range(future-1):
        #     output_t = self.linear1(output_t)
        #     output_t = self.leaky_relu(output_t)
        #     h_t, c_t = self.lstm1(output_t, (h_t, c_t))
        #     h_t_2, c_t_2 = self.lstm2(h_t, (h_t_2, c_t_2))
        #     output_t = self.linear4(h_t_2)
        #     outputs_full.append(torch.unsqueeze(output_t, 2))
        #     outputs.append(torch.unsqueeze(output_t, 2))

        outputs_full = torch.cat(outputs_full, dim=2)
        outputs = torch.cat(outputs, dim=2)
        return outputs, outputs_full
