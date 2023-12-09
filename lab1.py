import time
import wave
from datetime import datetime
from statistics import mean

import matplotlib.pyplot as plt
from numba import njit
from scipy.io import wavfile
import numpy as np
import pylab

# Прямое преобразованиие Фурье
def ft_integral(time, signal, omegas):
    delta = time[1] - time[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        X[i] = np.dot(signal, np.exp(-1j * omegas[i] * time))
    return X * delta


# Обратное преобразованиие Фурье
def ift_integral(time, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(time),))
    for i in range(len(time)):
        x[i] = np.dot(spectrum, np.exp(1j * omegas * time[i])).real
    return x * (delta_omega / np.pi)


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


def df(up, step,):
    f = np.arange(0, up, step)
    omegas = f*2*np.pi

    # сумма (прямоугольники)
    start = datetime.now()
    spectrum = ft_integral(time, signal, omegas)
    finish = datetime.now()
    print('Время работы ft_integral: ' + str(finish - start))

    start = datetime.now()
    restored_signal = ift_integral(time, spectrum, omegas)
    finish = datetime.now()
    print('Время работы ift_integral: ' + str(finish - start))

    ax.plot(time, restored_signal, 'r', label="Восстановленный")

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(f, abs(spectrum), '', label="Спектр")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.legend()
    ax1.legend()



if __name__ == "__main__":
    data = np.genfromtxt('sig.txt', delimiter='')
    signal = data[:, 1]  # второй столбец содержит значения сигнала
    # получение временной оси
    time = data[:, 0]

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
    # df(100000, 1)
    # df(100000, 10)
    # # Изменение верхнехней границы частот
    # df(200000, 10)
    mean_value = mean(signal)
    for i in range(len(signal)):
        signal[i] = signal[i] - mean_value
    df(200000, 10)
    plt.show()
