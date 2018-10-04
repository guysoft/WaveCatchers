#!/usr/bin/env/python3
from scipy.io import wavfile
#import pysptk
import pyreaper
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.playback import play
import os
import sys


def get_pitch(fs, x):
    pm_times, pm, f0_times, f0, corr = pyreaper.reaper(x, fs)
    values, counts = np.unique(f0, return_counts=True)
    return values[counts.argmax()]

# Set folders and input file here
#file_load = "/home/guy/workspace/wave_detector/dataset4/all_up2.flac"
#output_dir = "/home/guy/workspace/wave_detector/dataset4/all_up"

if len(sys.argv) < 2:
    print("Please provide a file to slice")
    sys.exit(1)
file_load = sys.argv[1]
output_dir = os.path.join(os.path.dirname(file_load), os.path.basename(file_load).split(".")[0])

print("writing file to output:" + output_dir)
os.system("mkdir -p " + output_dir)

#Split audio to bits

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

song = AudioSegment.from_file(file_load)

#split track where silence is 2 seconds or more and get chunks

chunks = split_on_silence(song, 
    # must be silent for at least 0.02 seconds or 20 ms
    min_silence_len=1,

    # consider it silent if quieter than -5 dBFS
    #Adjust this per requirement
    silence_thresh=-30
)

print("chunks: " + str(len(chunks)))

#Process each chunk per requirements
for i, chunk in enumerate(chunks):
    ##Create 0.5 seconds silence chunk
    #silence_chunk = AudioSegment.silent(duration=0)

    #Add  0.5 sec silence to beginning and end of audio chunk
    #audio_chunk = silence_chunk + chunk + silence_chunk
    audio_chunk = chunk

    #Normalize each audio chunk
    normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

    #Export audio chunk with new bitrate
    #print("exporting chunk{0}.mp3".format(i) )
    #normalized_chunk.export(".//chunk{0}.mp3".format(i), bitrate='192k', format="mp3")
    print("exporting chunk{0}.wav".format(i) )
    
    samples = normalized_chunk.get_array_of_samples()
    #play(normalized_chunk)
    pitch = get_pitch(normalized_chunk.frame_rate, np.array(samples))
    
    
    normalized_chunk = normalized_chunk.set_channels(1)
    normalized_chunk.export(os.path.join(output_dir, "chunk{0}-{1}.wav".format(i, str(pitch))), format="wav")



#fs, x = wavfile.read("/home/guy/workspace/wave_detector/audio_analysis/chunk1.wav")

