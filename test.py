from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pylab


def ft_integral(times, signal, omegas):
    delta = times[1] - times[0]
    X = np.zeros((len(omegas),), dtype=np.complex128)
    for i in range(len(omegas)):
        X[i] = np.dot(signal, np.exp(-1j * omegas[i] * times))
    return X * delta


# Обратное преобразованиие Фурье
def ift_integral(times, spectrum, omegas):
    delta_omega = omegas[1] - omegas[0]
    x = np.zeros((len(times)))
    for i in range(len(times)):
        x[i] = np.dot(spectrum, np.exp(1j * omegas * times[i])).real
    return x * (delta_omega / np.pi)


def df(up, step):
    f = np.arange(0, up, step)
    omegas = f*2*np.pi

    # сумма (прямоугольники)
    start = datetime.now()
    spectrum = ft_integral(times, signal, omegas)
    finish = datetime.now()
    print('Время работы ft_integral: ' + str(finish - start))

    start = datetime.now()
    restored_signal = ift_integral(times, spectrum, omegas)
    finish = datetime.now()
    print('Время работы ift_integral: ' + str(finish - start))

    ax.plot(times, restored_signal, 'r', label="Восстановленный")

    ax1 = fig.add_subplot(3, 1, 3)
    ax1.plot(f, abs(spectrum), '', label="Спектр")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    # чтение данных из файла
    data = np.genfromtxt('S1_P4_P6_hann5_100kHz_Ch2.txt', delimiter='', skip_header=7)
    signal_values = data[:, 1]  # второй столбец содержит значения сигнала

    # получение временной оси
    times = data[:, 0]

    # получение значения сигнала
    signal = data[:, 1]
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
    # df(100000, 1)
    df(100000, 0.1)
    # plt.show()
    # df (3000, 0.1)

    # # Изменение верхнехней границы частот
    # df(150000, 0.1)
    # df(300000, 0.1)
    plt.show()




