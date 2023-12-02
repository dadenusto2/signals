import math

import numpy as np


# Прямое преобразование Фурье
def ft(x):
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
def ift(X):
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


# частотная ось
def frequency_axis(N, fs):
    df = fs / N  # Шаг частоты (разрешение)
    f = np.arange(0, N) * df  # Частотная ось
    return f