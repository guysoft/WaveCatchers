import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def my_fft(path):
    x, sampleRate = sf.read(path)
    N = len(x)
    
    x = x/np.max(x) # Normalize
    f = np.arange(N//2)*sampleRate
    X = np.fft.fft(x)
    X = X[0:N//2]
    
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
    
    plt.show();
    
    
    
    
    