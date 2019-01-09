import os
import datetime
import numpy as np
import torch
import psutil
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
from network import DLSTM, Discriminative_loss, StdMSELoss
from load_data import Loader, MemoryFriendlyLoader

# ------------------------------------------------------
torch.cuda.set_device(0)
plt.switch_backend('agg')
# ------------------------------------------------------
# Hyper parameters
EPOCH = 100
LR = 1e-5
BATCH_SIZE = 64
LR_strategy = []
# Global
workplace = '.'
dataset_dir = './8sMIR/train'
# ------------------------------------------------------
Dataset = MemoryFriendlyLoader(wavdir=dataset_dir, multi=50)
train_loader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
max_len = Dataset.max_len
frequence_range = Dataset.frequence_range
total_music = Dataset.count

model_name = 'net_' + str(EPOCH) + 'epoch_' + str(int(total_music / 1000)) + 'K_' + 'MIR'
model_information_txt = model_name + '.txt'
print('Training on %d samples...' % total_music)

# --------------------------------------------------------------
# some functions
def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s


def delta_time(datetime1, datetime2):
    if datetime1 > datetime2:
        datetime1, datetime2 = datetime2, datetime1
    second = 0
    # second += (datetime2.year - datetime1.year) * 365 * 24 * 3600
    # second += (datetime2.month - datetime1.month) * 30 * 24 * 3600
    second += (datetime2.day - datetime1.day) * 24 * 3600
    second += (datetime2.hour - datetime1.hour) * 3600
    second += (datetime2.minute - datetime1.minute) * 60
    second += (datetime2.second - datetime1.second)
    return second
# --------------------------------------------------------------
net = DLSTM(frequence_range=frequence_range, max_len=max_len)
net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
# loss_func = StdMSELoss()
loss_func = Discriminative_loss()

prev_time = datetime.datetime.now()  # 当前时间
plotx = []
ploty = []
check_loss = 9999999999999999
state = None
for epoch in range(EPOCH):
    losses = []
    count = 0
    for step, (x, y1, y2) in enumerate(train_loader):
        x = x.cuda()
        y1 = y1.cuda()
        y2 = y2.cuda()

        pred_y1, pred_y2 = net(x)

        pred_y1 = pred_y1.cuda()
        pred_y2 = pred_y2.cuda()
        # loss = loss_func(y1, y2, pred_y1, pred_y2)    # 这样会导致loss很大很大
        loss = loss_func(pred_y1, pred_y2, y1, y2)      # 这样就不会. 原因暂未知

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        count += len(x)
        if count / 500 == count // 500:
            print('%s  Processed %0.2f%% triples.\tMemory used %0.2f%%.\tCpu used %0.2f%%.' %
                  (show_time(datetime.datetime.now()), count / total_music * 100, psutil.virtual_memory().percent,
                   psutil.cpu_percent(1)))

    plotx.append(epoch)
    ploty.append(np.mean(losses))
    print('---\tepoch %d: Average_loss=%f\t---' % (epoch + 1, np.mean(losses)))
    if epoch // 1 == epoch / 1:  # 每个epoch打印一次误差折线图
        plt.plot(plotx, ploty)
        plt.savefig(os.path.join(workplace, model_name + '_result.jpg'))  # Linux 保存路径
    if epoch in LR_strategy:
        optimizer.param_groups[0]['lr'] /= 10

    if np.mean(losses) < check_loss:
        print('\nSaving model temporarily...')
        if not os.path.exists(os.path.join(workplace, 'models')):
            os.mkdir(os.path.join(workplace, 'models'))
        torch.save(net.state_dict(), os.path.join(workplace, 'models', model_name + '_best_params.pkl'))  # 保存参数
        check_loss = np.mean(losses)

plt.plot(plotx, ploty)
plt.savefig(os.path.join(workplace, model_name + '_result.jpg'))  # Linux 保存路径

cur_time = datetime.datetime.now()  # 训练后此时时间
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
print("Training costs %02d:%02d:%02d" % (h, m, s))

print('\nSaving model...')
if not os.path.exists(os.path.join(workplace, 'models')):
    os.mkdir(os.path.join(workplace, 'models'))
torch.save(net.state_dict(), os.path.join(workplace, 'models', model_name + '_final_params.pkl'))  # 保存参数

print('\nCollecting some information...')
fp = open(os.path.join(workplace, 'models', model_information_txt), 'w')
fp.write('Model Path:%s\n' % os.path.join(workplace, 'models', model_name + '.pkl'))
fp.write('\nModel Structure:\n')
print(net, file=fp)
fp.write('\nModel Hyper Parameters:\n')
fp.write('\tEpoch = %d\n' % EPOCH)
fp.write('\tBatch size = %d\n' % BATCH_SIZE)
fp.write('\tLearning rate = %f\n' % LR)
print('\tLR strategy = %s' % str(LR_strategy), file=fp)
fp.write('Train on %dK_%s\n' % (int(total_music / 1000), 'MIR'))
print("Totally costs %02d:%02d:%02d" % (h, m, s), file=fp)
fp.close()

cur_time = datetime.datetime.now()  # 训练后此时时间
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
print("Totally costs %02d:%02d:%02d" % (h, m, s))
print('All done.')
