from statistics import mean
from time import time
import wave
import matplotlib.pyplot as plt
import numpy as np
import pylab
from numba import njit


def dft(amplitudes):
    N = len(amplitudes)
    X = np.zeros((N,), dtype=np.complex128)
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X[k] = np.dot(amplitudes, e)
    return X / np.sqrt(N)


def idft(spectrum):
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)


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
    ax1.plot((omegas / (2 * np.pi))[:len(abs(spectrum)[abs(spectrum)<0.001])], abs(spectrum)[abs(spectrum)<0.001], 'b', label="Спектр")

    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.plot(times, restored_signal, 'r', label="Восстановленный сигнал", linestyle='--')

    ax.legend()
    ax1.legend()

if __name__ == "__main__":
    data = np.genfromtxt('sig.txt', delimiter='', skip_header=7)
    amplitudes = data[:, 1]  # второй столбец содержит значения сигнала

    print((max(amplitudes) - min(amplitudes)) / 2)
    mean_value = mean(amplitudes)
    for i in range(len(amplitudes)):
        amplitudes[i] = amplitudes[i] - mean_value
    # получение временной оси
    times = data[:, 0]

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
