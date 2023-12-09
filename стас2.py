from statistics import mean
from time import time
from tqdm import tqdm
from numba import njit
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pylab


def load_from_wav():
    samplerate, data = wavfile.read('Signals/signal1.wav')
    print('Частота дискретизации аудио =', samplerate)
    print('Форма матрицы входных данных =', data.shape)
    length = data.shape[0] / samplerate
    print("Длительность аудио =", length)
    times = np.linspace(0., length, data.shape[0])
    amplitudes = np.array(data, dtype=np.complex128)
    return times, amplitudes


@njit
def FT_fast(times, amplitudes, omegas):
    """
        Прямое преобразование Фурье, оптимизированное Numba
    """
    delta_t = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        X[i] = np.dot(amplitudes, np.exp(-1j * omegas[i] * times))
    result = X * delta_t
    return result


@njit
def IFT_fast(times, spectrum, omegas):
    """
        Обратное преобразование Фурье, оптимизированное Numba
    """
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times),))
    for i in range(len(times)):
        x[i] = np.dot(spectrum, np.exp(1j * omegas * times[i])).real
    result = x * (delta_omega / np.pi)
    return result


def FT_tqdm(times, amplitudes, omegas):
    """
        Обратное преобразование Фурье с выводом прогресса в консоль (tqdm)
    """
    delta_t = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in tqdm(range(len(omegas))):
        X[i] = np.dot(amplitudes, np.exp(-1j * omegas[i] * times))
    result = X * delta_t
    return result


def IFT_tqdm(times, spectrum, omegas):
    """
        Обратное преобразование Фурье с выводом прогресса в консоль (tqdm)
    """
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times),))
    for i in tqdm(range(len(times))):
        x[i] = np.dot(spectrum, np.exp(1j * omegas * times[i])).real
    result = x * (delta_omega / np.pi)
    return result


def task1(times, amplitudes):
    max_frequency = 15000
    step = 1e-1

    f = np.arange(0, max_frequency, step)
    omegas = 2 * np.pi * f

    start = time()
    spectrum = FT_fast(times, amplitudes, omegas)
    stop = time()
    print("Преобразование Фурье завершено за ", stop - start)

    start = time()
    restored_signal = IFT_fast(times, spectrum, omegas)
    stop = time()
    print("Обратное преобразование Фурье завершено за ", stop - start)

    fig1 = pylab.figure(1)
    pl1 = fig1.add_subplot(1, 1, 1)
    pl1.plot(times, amplitudes, 'g', label="Исходный сигнал")
    pl1.set_xlabel("t, сек")
    pl1.set_ylabel('p(t)')
    pl1.legend()

    fig2 = pylab.figure(2)
    pl2 = fig2.add_subplot(1, 1, 1)
    pl2.plot(omegas / (2 * np.pi), abs(spectrum), '', label="Спектр сигнала")
    pl2.set_xlabel('f')
    pl2.set_ylabel('')
    pl2.legend()

    fig3 = pylab.figure(3)
    pl3 = fig3.add_subplot(1, 1, 1)
    pl3.plot(times, amplitudes, 'g', label="Исходный сигнал")
    pl3.plot(times, restored_signal, 'r', label="Восстановленный сигнал")
    pl3.set_xlabel("t, сек")
    pl3.set_ylabel('p(t)')
    pl3.legend()

    plt.show()



def DFT(amplitudes):
    N = len(amplitudes)
    X = np.zeros((N,), dtype=np.complex128)
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X[k] = np.dot(amplitudes, e)
    return X / np.sqrt(N)


def IDFT(spectrum):
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)


def task2(times, amplitudes, go_fast=False):
    N = len(amplitudes)
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])


    start = time()
    if not go_fast:
        spectrum = DFT(amplitudes)
    else:
        spectrum = np.fft.fft(amplitudes, norm='ortho')
    stop = time()
    print("Дискретное преобразование Фурье завершено за ", stop - start)

    start = time()
    if not go_fast:
        restored_signal = IDFT(spectrum)
    else:
        restored_signal = np.fft.ifft(spectrum, norm='ortho')
    stop = time()
    print("Обратное дискретное преобразование Фурье завершено за ", stop - start)


    fig1 = pylab.figure(1)
    pl1 = fig1.add_subplot(1, 1, 1)
    pl1.plot(times, amplitudes, '', label="Исходный сигнал")
    pl1.set_xlabel("Время")
    pl1.set_ylabel('Напряжение')
    pl1.legend()

    fig2 = pylab.figure(2)
    pl2 = fig2.add_subplot(1, 1, 1)
    pl2.plot((omegas / (2 * np.pi))[:len(abs(spectrum)[abs(spectrum)<0.001])], abs(spectrum)[abs(spectrum)<0.001], '', label="Спектр сигнала")
    pl2.set_xlabel('f')
    pl2.set_ylabel('')
    pl2.legend()

    fig3 = pylab.figure(3)
    pl3 = fig3.add_subplot(1, 1, 1)
    pl3.plot(times, amplitudes, 'g', label="Исходный сигнал")
    pl3.plot(times, restored_signal, 'r', label="Восстановленный сигнал")
    pl3.set_xlabel("t, сек")
    pl3.set_ylabel('p(t)')
    pl3.legend()

    plt.show()


if __name__ == "__main__":
    data = np.genfromtxt('sig.txt', delimiter='', skip_header=7)
    amplitudes = data[:, 1]  # второй столбец содержит значения сигнала

    print((max(amplitudes) - min(amplitudes)) / 2)
    mean_value = mean(amplitudes)
    for i in range(len(amplitudes)):
        amplitudes[i] = amplitudes[i] - mean_value
    # получение временной оси
    times = data[:, 0]

    task2(times, amplitudes, go_fast=True)


