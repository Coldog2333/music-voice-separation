import os
import wave
import utils
import numpy as np

dataset_dir = './8sMIR/test'
pred_dir = './evaluation'

music_filename = []
for filename in os.listdir(dataset_dir):
    if filename.split('.')[-1] != 'wav':  # 跳过非wav文件
        continue
    else:
        music_filename.append(filename)

music_list = []
musicname_info = dict()

def read_wav(filename):
    f = wave.open(filename, 'rb')
    params = f.getparams()
    nchannels, _, _, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    if nchannels == 2:
        wave_data = np.frombuffer(str_data, dtype=np.short)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T
        return wave_data[0], wave_data[1]
    elif nchannels == 1:
        wave_data = np.frombuffer(str_data, dtype=np.short)
        return wave_data

SDR, SIR, SAR = [], [], []
count = 0
for filename in music_filename:
    pred_bgm = read_wav(os.path.join(pred_dir, filename.split('.')[0] + '_bgm.wav'))
    pred_vocal = read_wav(os.path.join(pred_dir, filename.split('.')[0] + '_vocal.wav'))
    bgm, vocal = read_wav(os.path.join(dataset_dir, filename))

    sdr = utils.SDR(bgm, vocal, pred_bgm, pred_vocal)
    sir = utils.SIR(bgm, vocal, pred_bgm, pred_vocal)
    sar = utils.SAR(bgm, vocal, pred_bgm, pred_vocal)

    SDR.append((sdr[0] + sdr[1]) / 2)
    SIR.append((sir[0] + sir[1]) / 2)
    SAR.append((sar[0] + sar[1]) / 2)

    count += 1
    print(' Processed %d/%d\tSDR=%f\tSIR=%f\tSAR=%f' % (count, len(music_filename), (sdr[0] + sdr[1]) / 2, (sir[0] + sir[1]) / 2, (sar[0] + sar[1]) / 2))
print('Mean SDR=%f, SIR=%f, SAR=%f' % (np.mean(SDR), np.mean(SIR), np.mean(SAR)))