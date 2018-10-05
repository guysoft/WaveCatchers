from __future__ import print_function, division
from train_svc import load_sound_file_for_network, load_dataset, FREQUENCY_UNIT_COUNT, labels, MODEL_PATH
import os
import numpy as np
import pickle
from get_time_series import get_file
import scipy.io.wavfile
import simpleaudio as sa

import numpy as np
import pyaudio
import samplerate as sr



if __name__ == "__main__":
    input_rate = 44100
    target_rate = 44100
    chunk = int(target_rate * 0.02)  # sample evrey 0.02 seconds

    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)

    test_file = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset1", "all_down", "chunk10-98.0.wav"))

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1,
                        rate=input_rate, input=True,
                        frames_per_buffer=chunk)

    x_data = np.zeros((1, 3*FREQUENCY_UNIT_COUNT))




    while True:
        raw_data = stream.read(chunk)
        data = np.fromstring(raw_data, dtype=np.int16)
        scipy.io.wavfile.write('xenencounter_23sin3.wav', input_rate, data)

        data = get_file('xenencounter_23sin3.wav')

        wave_obj = sa.WaveObject.from_wave_file('xenencounter_23sin3.wav')
        play_obj = wave_obj.play()
        play_obj.wait_done()

        good_length = min(FREQUENCY_UNIT_COUNT, len(data[1]))
        freq = data[0][:good_length]
        amplitude = np.abs(data[1])[:good_length]
        phase = np.angle(data[1])[:good_length]

        #        import code; code.interact(local=dict(globals(), **locals()))
        x_data[0] = np.concatenate((freq, amplitude, phase))

        y_pred = clf.predict(x_data)

        print(list(labels[x] for x in y_pred))






    stream.stop_stream()
    stream.close()
    audio.terminate()


    test = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "dataset4", "all_up", "chunk86-97.56637.wav"))


    #





    print("Done")