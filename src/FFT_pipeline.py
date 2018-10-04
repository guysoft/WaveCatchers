import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def my_fft(path):
    x, sampleRate = sf.read(path)
    N = len(x)
    
    cutoff = 60 #Hz. Cutoff frequency to remove electricity noise
    
    # Get frequency vector, filtered
    f = np.arange(N//2)*sampleRate/N
    index = [i for i,e in enumerate(f) if (e > cutoff) and (i < N//2)]
    f = f[index]
    
    # Get FFT complex amplitudes, normalized and filtered
    X = np.fft.fft(x)
    X = X[index]
    X = X/np.max(np.abs(X))
    
    return f, X;

def multiplied_fft(X):
    Y = np.abs(X) * np.sign(np.angle(X))
    return Y;

def plot_compare_fft(path_up, path_down):
    f1, X1 = my_fft(path_up)
    Y1 = multiplied_fft(X1)
    plt.plot(f1,Y1)
    
    f2, X2 = my_fft(path_down)
    Y2 = multiplied_fft(X2)
    plt.plot(f2,Y2)
    
    plt.show()
    

    
    
    