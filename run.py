import os
import wave
import utils
import numpy as np
import torch
from network import DLSTM

# -------------------------------------------------------
dataset_dir = r'./8sMIR/test'
# -------------------------------------------------------
music_filename = []
for filename in os.listdir(dataset_dir):
    if filename.split('.')[-1] != 'wav':  # 跳过非wav文件
        continue
    else:
        music_filename.append(os.path.join(dataset_dir, filename))

music_list = []
musicname_info = dict()

for filename in music_filename:
    music_info = dict()
    f = wave.open(filename, 'rb')
    params = f.getparams()
    # 声道数，量化位数(byte单位)，采样频率(帧速率)，采样点数
    nchannels, _, _, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data.shape = -1, nchannels
    wave_data = wave_data.T

    # music_info['bgm'] = wave_data[0]
    # music_info['vocal'] = wave_data[1]
    music_info['mix'] = wave_data[0] + wave_data[1]
    music_info['path'] = os.path.split(filename)[0]
    music_info['filename'] = os.path.split(filename)[1]
    music_info['magnitude'] = utils.Signal_to_Magnitude_and_Phase(music_info['mix'], window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T
    music_info['phase'] = utils.Signal_to_Magnitude_and_Phase(music_info['mix'], window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[1]
    music_list.append(music_info)

temp= utils.Signal_to_Magnitude_and_Phase(music_list[0]['mix'], window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=True)[0].T
windows = temp.shape[0]
frequence_range = temp.shape[1]

count = 0

net = DLSTM(frequence_range=513, max_len=251)
net = net.eval().cuda()
net.load_state_dict(torch.load('./models/net_100epoch_12K_MIR_best_params.pkl'))

for music in music_list:
    bgm_magnitude, vocal_magnitude = net(torch.FloatTensor(music['magnitude']).view(1, music['magnitude'].shape[0], music['magnitude'].shape[1]).cuda())
    bgm_magnitude = bgm_magnitude.cpu().detach().numpy()
    vocal_magnitude = vocal_magnitude.cpu().detach().numpy()
    bgm = utils.Magnitude_and_Phase_to_Signal(bgm_magnitude[0, :, :].T, music['phase'])
    vocal = utils.Magnitude_and_Phase_to_Signal(vocal_magnitude[0, :, :].T, music['phase'])
    if not os.path.exists('./evaluation'):
        os.mkdir('./evaluation')
    utils.save_wave(bgm, os.path.join('./evaluation', music['filename'].split('.')[0] + '_bgm.wav'))
    utils.save_wave(vocal, os.path.join('./evaluation', music['filename'].split('.')[0] + '_vocal.wav'))
    count += 1
    print('Processed %d/%d\r' % (count, len(music_list)), end='')