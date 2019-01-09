import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import utils
import wave

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, wavdir, multi=1):
        self.wavdir = wavdir
        self.multi = multi
        self.wavfiles = self.load_wavfiles()
        self.count = len(self.wavfiles) * multi
        self.frequence_range, self.max_len = self.get_params()

    def load_wavfiles(self):
        wavfiles = []
        for file in os.listdir(self.wavdir):
            if file.split('.')[-1] == 'wav':
                wavfiles.append(os.path.join(self.wavdir, file))
        return wavfiles

    def get_params(self):
        max_len = 0
        for wavfile in self.wavfiles:
            f = wave.open(wavfile, 'rb')
            params = f.getparams()
            # 声道数，量化位数(byte单位)，采样频率(帧速率)，采样点数
            nchannels, sampwidth, framerate, nframes = params[:4]
            str_data = f.readframes(nframes)
            f.close()
            wave_data = np.frombuffer(str_data, dtype=np.short)
            wave_data.shape = -1, nchannels
            wave_data = wave_data.T
            temp = np.array(utils.Signal_to_Magnitude_and_Phase(wave_data[0], window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
            if max_len < temp.shape[0]:
                max_len = temp.shape[0]
        frequence_range = temp.shape[1]
        return frequence_range, max_len


    def __getitem__(self, index):
        read_index, k = divmod(index, self.multi)
        f = wave.open(self.wavfiles[read_index], 'rb')
        params = f.getparams()
        # 声道数，量化位数(byte单位)，采样频率(帧速率)，采样点数
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.frombuffer(str_data, dtype=np.short)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T

        self.processing_bgm = wave_data[0]
        step = int(len(self.processing_bgm) / self.multi)
        self.processing_vocal = utils.shift(wave_data[1], k * step)

        y1_feature = np.array(utils.Signal_to_Magnitude_and_Phase(self.processing_bgm, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
        y2_feature = np.array(utils.Signal_to_Magnitude_and_Phase(self.processing_vocal, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
        x_feature = np.array(utils.Signal_to_Magnitude_and_Phase(self.processing_bgm + self.processing_vocal, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)

        return torch.from_numpy(x_feature), torch.from_numpy(y1_feature), torch.from_numpy(y2_feature)

    # faster but maybe cannot work
    # def __getitem__(self, index):
    #     if self.processing_index == 0:
    #         f = wave.open(os.path.join(self.wavdir, self.wavfiles[index]), 'rb')
    #         params = f.getparams()
    #         # 声道数，量化位数(byte单位)，采样频率(帧速率)，采样点数
    #         nchannels, sampwidth, framerate, nframes = params[:4]
    #         str_data = f.readframes(nframes)
    #         f.close()
    #         wave_data = np.frombuffer(str_data, dtype=np.short)
    #         wave_data.shape = -1, nchannels
    #         wave_data = wave_data.T
    #
    #         self.processing_bgm = wave_data[0]
    #         self.processing_vocal = wave_data[1]
    #
    #         self.processing_y1_feature = np.array(utils.Signal_to_Magnitude_and_Phase(self.processing_bgm, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
    #     else:
    #         step = int(len(self.processing_bgm) / self.multi)
    #         self.processing_vocal = utils.shift(self.processing_vocal, step)
    #
    #     x = self.processing_bgm + self.processing_vocal
    #
    #     y1_feature = self.processing_y1_feature
    #     y2_feature = np.array(utils.Signal_to_Magnitude_and_Phase(self.processing_vocal, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
    #     x_feature = np.array(utils.Signal_to_Magnitude_and_Phase(x, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
    #     self.processing_index = divmod(self.processing_index + 1, self.multi)[1]
    #
    #     return x_feature, y1_feature, y2_feature

    def __len__(self):
        return self.count


class Loader():
    def __init__(self, wavdir, multi=1):
        self.wavdir = wavdir
        self.multi = multi
        self.hear_file = 'abjones_1_01.wav'
        self.total_music = self.count_music() * multi
        self.x_features, self.y1_features, self.y2_features, self.max_len, self.hear_flag = self.read_data()
        self.frequence_range = self.x_features[0].shape[1]

        self.hear_phase = self.get_hear_phase()

    def count_music(self):
        count = 0
        for wavfile in os.listdir(self.wavdir):
            if wavfile.split('.')[-1] != 'wav':
                continue
            else:
                count += 1
        return count

    def get_hear_phase(self):
        f = wave.open(os.path.join(self.wavdir, self.hear_file), 'rb')
        params = f.getparams()
        # 声道数，量化位数(byte单位)，采样频率(帧速率)，采样点数
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.frombuffer(str_data, dtype=np.short)
        wave_data.shape = -1, nchannels
        wave_data = wave_data.T

        bgm = wave_data[0]
        vocal = wave_data[1]
        mix = bgm + vocal

        # 因为只shift人声, 所以bgm的feature可以复用.
        return np.array(utils.Signal_to_Magnitude_and_Phase(mix, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)

    def read_data(self):
        x_features = []
        y1_features = []
        y2_features = []
        flag = 0
        count = 0
        max_len = 0
        for wavfile in os.listdir(self.wavdir):
            if wavfile.split('.')[-1] != 'wav':
                continue
            if flag == 0 and wavfile == self.hear_file:
                flag = count
            f = wave.open(os.path.join(self.wavdir, wavfile), 'rb')
            params = f.getparams()
            # 声道数，量化位数(byte单位)，采样频率(帧速率)，采样点数
            nchannels, sampwidth, framerate, nframes = params[:4]
            str_data = f.readframes(nframes)
            f.close()
            wave_data = np.frombuffer(str_data, dtype=np.short)
            wave_data.shape = -1, nchannels
            wave_data = wave_data.T

            bgm = wave_data[0]
            vocal = wave_data[1]
            mix = bgm + vocal

            # 因为只shift人声, 所以bgm的feature可以复用.
            y1_feature = np.array(utils.Signal_to_Magnitude_and_Phase(bgm, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T)
            if y1_feature.shape[0] > max_len:
                max_len = y1_feature.shape[0]

            step = int(len(bgm) / self.multi)
            for k in range(self.multi):
                y2 = utils.shift(vocal, frame=k * step)
                x = bgm + y2
                y1_features.append(y1_feature)
                y2_features.append(np.array(utils.Signal_to_Magnitude_and_Phase(y2, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T))
                x_features.append(np.array(utils.Signal_to_Magnitude_and_Phase(x, window=np.hamming(1024), nperseg=1024, noverlap=0.5 * 1024, save_stft=False)[0].T))
                count += 1
                print('Loaded %d Music\r' % count, end='')
        return x_features, y1_features, y2_features, max_len, flag