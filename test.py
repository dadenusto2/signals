import time
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pylab
from filters import *


def ft1(times, amplitudes, omegas):
    delta_t = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        X[i] = np.dot(amplitudes, np.exp(-1j * omegas[i] * times))
    return X * delta_t


def ift1(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times),))
    for i in range(len(times)):
        x[i] = np.dot(spectrum, np.exp(1j * omegas * times[i])).real
    return x * (delta_omega / np.pi)


def ft2(times, amplitudes, omegas):
    delta_t = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        edt = np.exp(1j * omegas[i] * delta_t)
        k = (2 - edt - (1 / edt)) / (delta_t * omegas[i] ** 2)
        X[i] = k * np.dot(amplitudes, np.exp(-1j * omegas[i] * times))
    return X


def ift2(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times),))
    for i in range(len(times)):
        e_d_omega = np.exp(1j * times[i] * delta_omega)
        k = (2 - e_d_omega - (1 / e_d_omega)) / (delta_omega * times[i] ** 2)
        x[i] = (k * np.dot(spectrum, np.exp(1j * times[i] * omegas))).real
    return x / np.pi


def first_task():
    omegas = np.arange(1e-2, 15000, 1)

    # сумма (прямоугольники)
    spectrum = ft1(time, signal, omegas)
    restored_signal = ift1(time, spectrum, omegas)

    # интеграл
    # spectrum = ft2(times, amplitudes, omegas)
    # restored_signal = ift2(times, spectrum, omegas)

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), '', label="Спектр")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.plot(time, restored_signal, 'r', label="Восстановленный", linestyle='--')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    data = np.genfromtxt('S1_P1_P2_hann5_200kHz_Ch2.txt', delimiter='', skip_header=7)
    signal_values = data[:, 1]  # второй столбец содержит значения сигнала

    # получение временной оси
    time = data[:, 0]

    # получение значения сигнала
    signal = data[:, 1]

    fig = pylab.figure(1)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(time, signal, 'g', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Напряжение')

    first_task()

    plt.show()

    print('Complete!')
