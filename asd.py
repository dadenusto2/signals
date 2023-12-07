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
            real += X[k] * np.exp(angle)
        x.append(real / N)
    return x

@njit
def idft(spectrum):
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)


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
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])

    if method == 'direct_df':
        start = datetime.now()
        spectrum = np.abs(direct_ft(signal))
        finish = datetime.now()
        print('Время работы direct_ft: ' + str(finish - start))

        start = datetime.now()
        restored_signal = direct_ift(spectrum)
        finish = datetime.now()
        print('Время работы direct_ift: ' + str(finish - start))
    elif method == 'fft':
        start = datetime.now()
        spectrum = numpy.fft.fft(signal)
        finish = datetime.now()
        print('Время работы fft: ' + str(finish - start))

        start = datetime.now()
        restored_signal = numpy.fft.ifft(spectrum)
        finish = datetime.now()
        print('Время работы ifft: ' + str(finish - start))
    else:
        start = datetime.now()
        spectrum = numpy.fft.fftshift(signal)
        finish = datetime.now()
        print('Время работы fftshift: ' + str(finish - start))

        start = datetime.now()
        restored_signal = numpy.fft.ifftshift(spectrum)
        finish = datetime.now()

        print('Время работы ifftshift: ' + str(finish - start))

    ax.plot(times, restored_signal, 'r', label="Восстановленный")
    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(omegas / (2 * math.pi), abs(spectrum), '', label="Спектр")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    data = np.genfromtxt('S1_P4_P6_hann5_100kHz_Ch2.txt', delimiter='', skip_header=7)
    signal = data[:, 1]  # второй столбец содержит значения сигнала

    # получение временной оси
    times = data[:, 0]

    # График исходного сигнала
    fig = pylab.figure(1)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(times, signal, '', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Амплитуда')

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(times, signal, 'b', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Амплитуда')


    # Изменение шага дискретизации
    df('direct_df')
    # df('fft')
    # df('fftshift')
    plt.show()



