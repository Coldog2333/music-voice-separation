import torch

# 神经网络架构
class DLSTM(torch.nn.Module):
    def __init__(self, frequence_range, max_len):
        super(DLSTM, self).__init__()
        self.max_len = max_len
        self.frequence_range = frequence_range

        self.lstm1 = torch.nn.LSTM(
            input_size=frequence_range,
            hidden_size=1000,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.lstm2 = torch.nn.LSTM(
            input_size=1000,
            hidden_size=2000,
            num_layers=1,
            batch_first=True,
        )
        self.lstm3 = torch.nn.LSTM(
            input_size=2000,
            hidden_size=frequence_range*2,
            num_layers=1,
            batch_first=True,
        )

    # 暂时先做定time_step的神经网络
    def forward(self, x):
        # x (time_step=max_time, input_size=frequence_range)
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out3, _ = self.lstm3(out2)

        # state (num_layers=1, batch, hidden_size)
        # out (batch, time_step=max_time, hidden_size)
        y1 = out3[:, :, :self.frequence_range]
        y2 = out3[:, :, self.frequence_range:]

        mask = torch.abs(y1) / (torch.abs(y1) + torch.abs(y2))
        bgm = mask * x
        vocal = (1 - mask) * x
        return bgm, vocal

class Discriminative_loss(torch.nn.Module):
    def __init__(self):
        super(Discriminative_loss, self).__init__()

    def forward(self, y1, y2, pred_y1, pred_y2):
        return torch.nn.functional.mse_loss(y1, pred_y1) - 0.05 * torch.nn.functional.mse_loss(y1, pred_y2) + \
               torch.nn.functional.mse_loss(y2, pred_y2) - 0.05 * torch.nn.functional.mse_loss(y2, pred_y1)

class StdMSELoss(torch.nn.Module):
    def __init__(self):
        super(StdMSELoss, self).__init__()

    def forward(self, y1, y2, pred_y1, pred_y2):
        return torch.nn.functional.mse_loss(y1, pred_y1) + torch.nn.functional.mse_loss(y2, pred_y2)