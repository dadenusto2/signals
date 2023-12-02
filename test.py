import time
import wave

import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import pylab


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
    amplitudes = np.array(amplitudes)
    return t, amplitudes


def first_task():
    f= np.arange(0, 1000, 0.2)
    omegas = f*2*np.pi

    # сумма (прямоугольники)
    spectrum = ft1(times, amplitudes, omegas)
    restored_signal = ift1(times, spectrum, omegas)

    # интеграл
    # spectrum = ft2(times, amplitudes, omegas)
    # restored_signal = ift2(times, spectrum, omegas)

    ax1 = fig.add_subplot(2, 1, 2)
    ax1.plot(omegas / (2 * np.pi), abs(spectrum), '', label="Спектр")
    ax1.set_xlabel('Частота')
    ax1.set_ylabel('Амплитуда')

    ax.plot(times, restored_signal, 'r', label="Восстановленный", linestyle='--')

    ax.legend()
    ax1.legend()


if __name__ == "__main__":
    times, amplitudes = get_data_from_file('sample.wav')

    fig = pylab.figure(1)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(times, amplitudes, 'g', label="Исходный сигнал")
    ax.set_xlabel("Время")
    ax.set_ylabel('Напряжение')

    first_task()

    plt.show()

    print('Complete!')
