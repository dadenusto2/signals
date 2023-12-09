import math
from datetime import datetime
from statistics import mean
import matplotlib.pyplot as plt
import numpy.fft
import numpy as np
import pylab

# Прямое дискретное преобразование Фурье
def direct_ft(signal):
    N = len(signal)
    X = np.zeros((N,), dtype=np.complex128)
    n = np.arange(N)
    for k in range(N):
        e = np.exp(-2j * np.pi * k * n / N)
        X[k] = np.dot(signal, e)
    return X / np.sqrt(N)


# Обратное дискретное преобразование Фурье
def direct_ift(spectrum):
    N = len(spectrum)
    restored_signal = np.zeros((N,), dtype=np.complex128)
    k = np.arange(N)
    for n in range(N):
        e = np.exp(2j * np.pi * k * n / N)
        restored_signal[n] = np.dot(spectrum, e)
    return restored_signal / np.sqrt(N)


def df(method, times, signal, cut=False):
    N = len(signal)
    delta_t = times[1] - times[0]
    delta_omega = 2 * np.pi / (N * delta_t)
    omegas = np.array([k * delta_omega for k in range(N)])
    print(omegas)

    if method == 'direct_df':
        start = datetime.now()
        spectrum = direct_ft(signal)
        finish = datetime.now()
        print('Время работы direct_ft: ' + str(finish - start))

        start = datetime.now()
        restored_signal = direct_ift(spectrum)
        finish = datetime.now()
        print('Время работы direct_ift: ' + str(finish - start))
    elif method == 'fft':
        start = datetime.now()
        spectrum = numpy.fft.fft(signal, norm='ortho')
        finish = datetime.now()
        print('Время работы fft: ' + str(finish - start))

        start = datetime.now()
        restored_signal = numpy.fft.ifft(spectrum, norm='ortho')
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
    print(max(spectrum))
    if not cut:
        ax1.plot(omegas / (2 * np.pi), abs(spectrum), '', label="Спектр")
    else:
        ax1.plot((omegas / (2 * np.pi))[:len(np.abs(spectrum)[np.abs(spectrum) < 0.001])],np.abs(spectrum)[np.abs(spectrum) < 0.001], '', label="Спектр")

    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    data = np.genfromtxt('sig.txt', delimiter='', )
    signal = data[:, 1]  # второй столбец содержит значения сигнала
    mean_value = mean(signal)
    for i in range(len(signal)):
        signal[i] = signal[i] - mean_value
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
    # df('direct_df', times, signal)
    # df('direct_df', True)
    # df('fft', times, signal)
    df('fft', times, signal, True)
    # df('fftshift', times, signal)
    plt.show()

