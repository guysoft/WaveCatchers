import os
from os.path import isfile, join
import pandas as pd
from FFT_pipeline import my_fft, multiplied_fft

basic_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = join(path, "dataset", "dataset1")

freq, FFT = [my_fft(file) for file in os.listdir(dataset_path) if isfile(join(dataset_path, file))]
    
