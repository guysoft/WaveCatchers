import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os

#path = 'C:\\Users\\Nadav\\Google Drive\\Data Projects\\WaveCatching\\'
#file1 = 'Violin-up-down\\all_down\\chunk827-164.55225.wav'
#file2 = 'Violin-up-down\\all_up\\chunk380-164.55225.wav'

path = os.path.realpath(os.path.join(os.path.dirname(__file__)))
file1 = os.path.join("dataset", "dataset1", "all_down", "chunk827-164.55225.wav") # 'Violin-up-down\\all_down\\'
file2 = os.path.join("dataset", "dataset1", "all_up", "chunk380-164.55225.wav") # 'Violin-up-down\\all_down\\'

y1, fs = sf.read(os.path.join(path, file1))
y1 = y1/np.max(y1)
N1 = len(y1)
T = 1/fs
time1 = T*np.arange(N1)
print("we")

y2, fs = sf.read(os.path.join(path, file2))
y2 = y2/np.max(y2)
N2 = len(y2)
T = 1/fs
time2 = T*np.arange(N2)

plt.plot(time1, y1, label="down")
plt.plot(time2, y2, label="up")
plt.show()

freq1 = np.arange(N1//2)*fs
Y1 = np.fft.fft(y1)
A1 = np.max(y1)
Y1_adjusted = Y1[0:N1//2]/A1

freq2 = np.arange(N2//2)*fs
Y2 = np.fft.fft(y2)
A2 = np.max(y2)
Y2_adjusted = Y2[0:N2//2]/A2

"""
plt.subplot(2,2,1)
plt.plot(freq1, np.abs(Y1_adjusted), label="up(y)", linewidth=2)
plt.plot(freq2, np.abs(Y2_adjusted), label="down(y)", linewidth=1)
plt.grid()
plt.legend()
"""

plt.subplot(2,2,2)
plt.plot(np.angle(Y1_adjusted) * np.abs(Y1_adjusted), label="up(y)", linewidth=2)
plt.plot(np.angle(Y2_adjusted) * np.abs(Y2_adjusted), label="down(y)", linewidth=1)
plt.grid()
plt.legend()

"""
plt.subplot(2,2,3)
p1 = np.sign(np.angle(Y1_adjusted)) * np.abs(Y1_adjusted)
p2 = np.sign(np.angle(Y2_adjusted)) * np.abs(Y2_adjusted)

plt.plot(freq1, p1, label="up(y)", linewidth=2)
plt.plot(freq2, p2, label="down(y)", linewidth=1)
"""

plt.grid()

plt.show()


print("done")