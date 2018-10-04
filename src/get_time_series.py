import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os


def get_file(path):
    y1, fs = sf.read(path)
    y1 = y1 / np.max(y1)
    N1 = len(y1)
    T = 1 / fs
    time1 = T * np.arange(N1)

    freq1 = np.arange(N1 // 2) * fs / N1
    Y1 = np.fft.fft(y1)
    A1 = np.max(y1)
    Y1_adjusted = Y1[0:N1 // 2] / A1
    return freq1, Y1_adjusted


if __name__ == "__main__":

    path = os.path.realpath(os.path.join(os.path.dirname(__file__)))
    file1 = os.path.join("dataset", "dataset1", "all_down", "chunk827-164.55225.wav") # 'Violin-up-down\\all_down\\'


    freq1, Y1_adjusted = get_file(os.path.join(path, file1))

    #plt.plot(freq1, np.angle(Y1_adjusted) * np.abs(Y1_adjusted), label="up(y)", linewidth=2)
    plt.plot(np.angle(Y1_adjusted) * np.abs(Y1_adjusted), label="up(y)", linewidth=2)
    plt.show()


    print("done")