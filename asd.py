from time import time
import wave
import matplotlib.pyplot as plt
import numpy as np
import pylab
from numba import njit

@njit
def dft(amplitudes):
    N = len(amplitudes)
    X = np.zeros((N,), dtype=np.complex128)
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X[k] = np.dot(amplitudes, e)
    return X / np.sqrt(N)


@njit
def idft(spectrum):
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)

def get_data_from_file(str):
    wav = wave.open(str,mode='r')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    types = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }
    samples = np.fromstring(content, dtype=types[sampwidth])
    amplitudes = []
    for i in range (0,len(samples),2):
        amplitudes.append((samples[i]+samples[i+1])/2)
    t = []
    for i in range(1,nframes+1):
        t.append(i/framerate)
    t=np.array(t)
    amplitudes = np.array(amplitudes,dtype=np.complex128)
    return t, amplitudes

def task_2(type_ft):

    N = len(amplitudes)
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])

    if type_ft == 'dft':
        start = time()
        spectrum = dft(amplitudes)
        stop = time()
        print("Дискретное преобразование Фурье завершено за ", stop - start)

        start = time()
        restored_signal = idft(spectrum)
        stop = time()
        print("Обратное дискретное преобразование Фурье завершено за ", stop - start)
    else:
        start = time()
        spectrum = np.fft.fft(amplitudes, norm='ortho')
        stop = time()
        print("Быстрое преобразование Фурье завершено за ", stop - start)

        restored_signal = np.fft.ifft(spectrum, norm='ortho')
        stop = time()
        print("Обратное Быстрое преобразование Фурье завершено за ", stop - start)

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), 'b', label="Спектр")

    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.plot(times, restored_signal, 'r', label="Восстановленный сигнал", linestyle='--')

    ax.legend()
    ax1.legend()

if __name__ == "__main__":

    times, amplitudes = get_data_from_file('без названия.wav')

    fig = pylab.figure(1)
    ax = fig.add_subplot(3, 1, 1)
    ax.plot(times, amplitudes, 'g', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Напряжение')

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(times, amplitudes, 'g', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Напряжение')

    task_2('fft')

    plt.show()
