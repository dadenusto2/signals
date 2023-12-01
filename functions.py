import math

import numpy as np


# Прямое преобразование Фурье
def dft(x):
    N = len(x)
    X = []
    for k in range(N):
        # print(f'{k}/{N}')
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
def idft(X):
    N = len(X)
    x = []
    for k in range(N):
        real = 0
        # print(f'{k}/{N}')
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            real += X[k].real * cos_val - X[k].imag * sin_val
        x.append(real / N)
    return x


def discret_ft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X


# частотная ось
def frequency_axis(N, fs):
    df = fs / N  # Шаг частоты (разрешение)
    f = np.arange(0, N) * df  # Частотная ось
    return f