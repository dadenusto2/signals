import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import functions
from test import get_signal

def direct_ft(x):
    N = len(x)
    X = []
    for k in range(N):
        print(f'{k}/{N}')
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


# Обратное преобразование Фурье
def direct_ift(X):
    N = len(X)
    x = []
    for k in range(N):
        real = 0
        print(f'{k}/{N}')
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            real += X[k].real * cos_val + X[k].imag * sin_val
        x.append(real / N)
    return x

def ft(file):


    time, signal = get_signal('signal 4 сек.wav')


    plt.plot(time, signal)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.title('График сигнала')
    plt.show()

    # Прямое преобразование Фурье
    start = datetime.now()
    dft = functions.ft(signal)
    finish = datetime.now()
    print('Время работы direct_ft: ' + str(finish - start))

    # Обратное преобразование Фурье
    start = datetime.now()
    idft = functions.ift(signal)
    finish = datetime.now()
    print('Время работы direct_dft: ' + str(finish - start))

    # Дикретное преобразование Фурье
    start = datetime.now()
    discret_ft = functions.discret_ft(signal)
    finish = datetime.now()
    print('Время работы discret_ft: ' + str(finish - start))

    # Быстрое преобразование Фурье
    start = datetime.now()
    fft = np.fft.fft(signal)
    finish = datetime.now()
    print('Время работы fft: ' + str(finish - start))


    # алгоритмы вычисления прямого Фурье
    fft_algorithms = ['dft', 'discret_ft', 'fft', 'fftshift', 'rfft']
    for fft_algorithm in fft_algorithms:
        # получение спектра
        print(fft_algorithm)

        # получение частотной оси
        if fft_algorithm == 'dft':
            fft_values = dft
        elif fft_algorithm == 'fft':
            fft_values = fft
        elif fft_algorithm == 'fftshift':
            fft_values = fftshift
        elif fft_algorithm == 'rfft':
            fft_values = rfft
        else:
            fft_values = discret_ft

        if fft_algorithm == 'rfft':
            freq = np.fft.rfftfreq(signal.size, len(signal))
        else:
            freq = functions.frequency_axis(signal.size, len(signal))
        # построение графика
        plt.plot(freq, np.abs(fft_values), label=f'Алгоритм: {fft_algorithm}')

    plt.xlabel('Частота, Гц')
    plt.ylabel('Преобразования Фурье')
    plt.title('Значения при разных алгоритмах вычисления прямого Фурье')
    plt.legend()
    plt.show()


    # Частотный спектр сигнала — это распределение энергии сигнала по частотам
    # получение спектра
    spectrum_dft = np.abs(dft)
    spectrum_discret_ft = np.abs(discret_ft)
    spectrum_fft = np.abs(fft)
    spectrum_fftshift = np.abs(fftshift)
    spectrum_rfft = np.abs(rfft)

    freq = functions.frequency_axis(signal.size, len(signal))
    plt.plot(freq, spectrum_dft)
    plt.plot(freq, spectrum_discret_ft)
    plt.plot(freq, spectrum_fft)
    plt.plot(freq, spectrum_fftshift)
    plt.plot(np.fft.rfftfreq(signal.size, len(signal)), spectrum_rfft)

    plt.xlabel('Частота, Гц')
    plt.ylabel('Модулb спектра')
    plt.title('График модуля спектра при разныйх агоритмах')
    plt.show()

    # получение обратного преобразования Фурье
    reconstructed_signal = idft

    # построение графика
    plt.plot(time, reconstructed_signal)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.title('График восстановленного сигнала')
    plt.show()

    # построение графика сигнала
    plt.plot(time, signal, label='Исходный сигнал')

    # построение графика восстановленного сигнала
    plt.plot(time, reconstructed_signal, label='Восстановленный сигнал (обратное преобрзование Фурье)')

    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.title('Сравнение исходного и восстановленного сигналов')
    plt.legend()
    plt.show()

    # значения шага по времени
    dt_values = [0.01, 0.1, 1]

    for dt in dt_values:
        # загрузка данных
        data = np.genfromtxt(file, delimiter='', skip_header=7)

        # получение временной оси
        time = data[:, 0]

        # получение значения сигнала
        signal = data[:, 1]

        # получение спектра
        spectrum = np.abs(dft)

        # получение частотной оси
        freq = functions.frequency_axis(signal.size, dt)

        # построение графика
        plt.plot(freq, spectrum, label=f'Шаг по времени: {dt} с')

    plt.xlabel('Частота, Гц')
    plt.ylabel('Модуль спектра')
    plt.title('Влияние шага по времени на модуль спектра')
    plt.legend()
    plt.show()

    # построение графиков значения прямого преобразования Фурье
    # при разных алгоритмах и разных значениях шага дискретизации:
