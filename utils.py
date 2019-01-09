import numpy as np
import random
from scipy import signal
import pyaudio
import wave


def Signal_to_Magnitude_and_Phase(x,
                                  fs=1.0,
                                  window='hann',
                                  nperseg=256,
                                  noverlap=None,
                                  nfft=None,
                                  detrend=False,
                                  return_onesided=True,
                                  boundary='zeros',
                                  padded=True,
                                  axis=-1,
                                  save_stft=False):
    _, _, stft = signal.stft(x, window=window, nperseg=nperseg, noverlap=noverlap,
                             nfft=nfft, detrend=detrend, return_onesided=return_onesided,
                             boundary=boundary, padded=padded, axis=axis)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    if save_stft == False:
        return magnitude, phase
    elif save_stft == True:
        return magnitude, phase, stft


def Magnitude_and_Phase_to_Signal(Magnitude, Phase,
                                  fs=1.0,
                                  window='hann',
                                  nperseg=None,
                                  noverlap=None,
                                  nfft=None,
                                  input_onesided=True,
                                  boundary=True,
                                  time_axis=-1,
                                  freq_axis=-2):
    # p.s. Please ensure that nperseg, noverlap are the same as stfting.
    Real = Magnitude * np.cos(Phase)
    Image = Magnitude * np.sin(Phase)
    stft = Real + Image * 1j
    _, istft = signal.istft(stft, fs=1.0, window=window, nperseg=nperseg, noverlap=noverlap,
                            nfft=nfft, input_onesided=input_onesided, boundary=boundary,
                            time_axis=time_axis, freq_axis=freq_axis)
    return istft.astype('int16')


def shift(signal, frame=None, t=0, framerate=16000, mode='standard'):  # 输入时间同时要输入帧率
    if mode == 'standard':
        if (frame):
            signal = np.roll(signal, frame)
        else:
            signal = np.roll(signal, t * framerate)
    elif mode == 'random':
        frame = random.randint(1, len(signal))
        signal = np.roll(signal, frame)
    return signal


def signal_pad(signal, frame=0, length=0, position='back', return_frame=False):  # 输入t就
    add = frame
    if position == 'back':
        signal = np.pad(signal, (0, frame), mode='constant').astype('int32')
        add += length - len(signal)
        signal = np.pad(signal, (0, length - len(signal)), mode='constant').astype('int32')
    elif position == 'front':
        signal = np.pad(signal, (frame, 0), mode='constant').astype('int32')
        add += length - len(signal)
        signal = np.pad(signal, (length - len(signal), 0), mode='constant').astype('int32')

    if return_frame:
        return signal, add
    else:
        return signal


def play_wave(wave, framerate=16000, nchannels=1, sampwidth=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(sampwidth),
                    channels=nchannels,
                    rate=framerate,
                    output=True)

    for i in range(0, wave.shape[0], 10):
        stream.write(wave[i:i + 10], num_frames=10)

    stream.stop_stream()
    stream.close()
    p.terminate()


def save_wave(signal, filename, nchannels=1, sampwidth=2, framerate=16000, nframes=0, comptype=None, compname=None):
    nframes = max(signal.shape)
    wf = wave.open(filename, 'wb')
    wf.setnchannels(nchannels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.setnframes(nframes)
    # wf.setcomptype(None)
    wf.writeframes(np.array(signal).astype('int16').tostring())
    wf.close()


# -------------------- eval --------------------
def get_bbs_component(s1, s2, pred_s1, pred_s2):
    s1 = np.matrix(s1)
    s2 = np.matrix(s2)
    pred_s1 = np.matrix(pred_s1)
    pred_s2 = np.matrix(pred_s2)
    # shape: 1 x n
    # target
    target_1 = (pred_s1 * s1.T) / (s1 * s1.T) * s1
    target_2 = (pred_s2 * s2.T) / (s2 * s2.T) * s2

    # interf
    Rss = np.matrix(np.array([[s1 * s1.T, s2 * s1.T], [s1 * s2.T, s2 * s2.T]]))
    iRss = Rss.I

    Pssj_1 = np.vstack((s1, s2)).T * iRss * np.vstack((pred_s1 * s1.T, pred_s1 * s2.T))
    Pssj_2 = np.vstack((s1, s2)).T * iRss * np.vstack((pred_s2 * s1.T, pred_s2 * s2.T))
    Pssj_1 = Pssj_1.T
    Pssj_2 = Pssj_2.T

    interf_1 = Pssj_1 - target_1  # 矩阵不能与ndarray相减, 但不会报错(
    interf_2 = Pssj_2 - target_2

    # noise = zeros
    noise_1 = 0
    noise_2 = 0

    # artif
    artif_1 = pred_s1 - target_1 - interf_1
    artif_2 = pred_s2 - target_2 - interf_2
    # return shape: 1 x n <matrix>
    return target_1, interf_1, target_2, interf_2, noise_1, noise_2, artif_1, artif_2


def SDR(s1, s2, pred_s1, pred_s2):
    target_1, interf_1, target_2, interf_2, noise_1, noise_2, artif_1, artif_2 = get_bbs_component(s1, s2, pred_s1, pred_s2)
    SDR1 = (target_1 * target_1.T) / ((interf_1 + noise_1 + artif_1) * (interf_1 + noise_1 + artif_1).T)
    SDR2 = (target_2 * target_2.T) / ((interf_2 + noise_2 + artif_2) * (interf_2 + noise_2 + artif_2).T)
    return 10 * np.log10(SDR1), 10 * np.log10(SDR2)


def SIR(s1, s2, pred_s1, pred_s2):
    target_1, target_2, interf_1, interf_2, _, _, _, _ = get_bbs_component(s1, s2, pred_s1, pred_s2)
    SIR1 = (target_1 * target_1.T) / (interf_1 * interf_1.T)
    SIR2 = (target_2 * target_2.T) / (interf_2 * interf_2.T)
    return 10 * np.log10(SIR1), 10 * np.log10(SIR2)


def SAR(s1, s2, pred_s1, pred_s2):
    target_1, interf_1, target_2, interf_2, noise_1, noise_2, artif_1, artif_2 = get_bbs_component(s1, s2, pred_s1, pred_s2)
    SAR1 = (target_1 + interf_1 + noise_1) * (target_1 + interf_1 + noise_1).T / (artif_1 * artif_1.T)
    SAR2 = (target_2 + interf_2 + noise_2) * (target_2 + interf_2 + noise_2).T / (artif_2 * artif_2.T)
    return 10 * np.log10(SAR1), 10 * np.log10(SAR2)


def distance(x, y):
    return np.sqrt(np.sum((x - y) * (x - y)))


def norm(x):
    return np.sqrt(np.dot(x, x))
