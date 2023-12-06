import math
import time
import wave
from datetime import datetime

import matplotlib.pyplot as plt
import numpy.fft
from numba import njit
from scipy.io import wavfile
import numpy as np
import pylab

# Прямое дискретное преобразование Фурье
def direct_ft(x):
    N = len(x)
    X = []
    for k in range(N):
        real = 0
        imag = 0
        for n in range(N):
            # степень (2pi/N*kn)
            angle = 2 * math.pi * k * n / N
            # e^angle=cos(angle)-isin(angle)
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            real += x[n] * cos_val
            imag -= x[n] * sin_val
        X.append(complex(real, imag))
    return X


# Обратное дискретное преобразование Фурье
def direct_ift(X):
    N = len(X)
    x = []
    for k in range(N):
        real = 0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            real += X[k].real * cos_val + X[k].imag * sin_val
        x.append(real / N)
    return x


# Считывание сигнала из файла
def get_signal(path):
    audio = wave.open(path,mode='r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = audio.getparams()
    content = audio.readframes(nframes)
    types = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }
    samples = np.fromstring(content, dtype=types[sampwidth])
    signal = []
    for i in range (0,len(samples),2):
        signal.append((samples[i]+samples[i+1])/2)
    time = []
    for i in range(1,nframes+1):
        time.append(i/framerate)
    time=np.array(time)
    signal = np.array(signal,dtype=np.complex128)
    return time, signal


def df(method):

    N = len(signal)
    delta_t = time[1] - time[0]
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])

    if method == 'direct_df':
        start = datetime.now()
        spectrum = direct_ft(signal)
        finish = datetime.now()
        print('Время работы ft_integral: ' + str(finish - start))

        start = datetime.now()
        restored_signal = direct_ift(spectrum)
        finish = datetime.now()
        print('Время работы ift_integral: ' + str(finish - start))
    elif method == 'fft':
        start = datetime.now()
        spectrum = numpy.fft.fft(signal)
        finish = datetime.now()
        print('Время работы fft: ' + str(finish - start))

        start = datetime.now()
        restored_signal = numpy.fft.ifft(spectrum)
        finish = datetime.now()
        print('Время работы ifft: ' + str(finish - start))
    elif method == 'fftshift':
        start = datetime.now()
        spectrum = numpy.fft.fftshift(signal)
        finish = datetime.now()
        print('Время работы fftshift: ' + str(finish - start))

        start = datetime.now()
        restored_signal = numpy.fft.ifftshift(spectrum)
        finish = datetime.now()

        print('Время работы ifftshift: ' + str(finish - start))

    ax.plot(time, restored_signal, 'r', label="Восстановленный")

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(omegas/ (2 * math.pi) , abs(spectrum), '', label="Спектр")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    time, signal = get_signal('signal 4 сек.wav')

    # График исходного сигнала
    fig = pylab.figure(1)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(time, signal, '', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Амплитуда')

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(time, signal, 'b', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Амплитуда')


    # Изменение шага дискретизации
    df('direct_df')
    # df('fft')
    # df('fftshift')
    plt.show()



